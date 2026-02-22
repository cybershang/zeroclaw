use super::persistence::{
    check_user_reply, clear_user_reply, create_task, get_task, log_action,
    update_task_status, update_task_step,
};
use super::react::{parse_react_response, REACT_SYSTEM_PROMPT};
use super::types::{
    ActionResultStatus, ActionSummary, BlockerReport, ErrorCategory, ErrorAction,
    IterativeConfig, Task, TaskOutcome, TaskStatus,
};
use super::detector::detect_stuck;

use crate::config::Config;
use crate::memory::{Memory, MemoryCategory};
use crate::providers::{ChatMessage, Provider, ToolCall};
use crate::tools::{self, Tool, ToolResult};
use anyhow::{anyhow, Context, Result};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

/// Run a task iteratively with autonomous execution
pub async fn run_iterative(
    config: &Config,
    goal: &str,
    iterative_config: &IterativeConfig,
) -> Result<TaskOutcome> {
    let task_id = format!("task_{}", Uuid::new_v4().to_string().split('-').next().unwrap_or(""));

    // Create task record
    let mut task = create_task(config, &task_id, goal).await?;

    // Initialize execution context
    let security = Arc::new(crate::security::SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));

    let runtime: Arc<dyn crate::runtime::RuntimeAdapter> =
        Arc::from(crate::runtime::create_runtime(&config.runtime)?);

    let memory: Arc<dyn Memory> = Arc::from(crate::memory::create_memory_with_storage(
        &config.memory,
        Some(&config.storage.provider.config),
        &config.workspace_dir,
        config.api_key.as_deref(),
    )?);

    // Build tools registry
    let tools_registry = tools::all_tools_with_runtime(
        Arc::new(config.clone()),
        &security,
        runtime,
        memory.clone(),
        None,
        None,
        &config.browser,
        &config.http_request,
        &config.workspace_dir,
        &config.agents,
        config.api_key.as_deref(),
        config,
    );

    // Create provider
    let provider_name = config.default_provider.as_deref().unwrap_or("openrouter");
    let provider: Box<dyn Provider> = crate::providers::create_routed_provider(
        provider_name,
        config.api_key.as_deref(),
        config.api_url.as_deref(),
        &config.reliability,
        &config.model_routes,
        config.default_model.as_deref().unwrap_or("claude-sonnet-4"),
    )?;

    // Build system prompt
    let mut system_prompt = String::from(REACT_SYSTEM_PROMPT);
    system_prompt.push_str(&format!("\n\nYour goal: {}", goal));

    // Initialize conversation history
    let mut history = vec![ChatMessage::system(&system_prompt)];

    // Store goal in memory
    let _ = memory
        .store(&task_id, goal, MemoryCategory::Core, Some(&task_id))
        .await;

    // Main execution loop
    let result = execution_loop(
        config,
        &mut task,
        iterative_config,
        provider.as_ref(),
        &tools_registry,
        &mut history,
    )
    .await;

    // Cleanup
    match &result {
        Ok(outcome) => {
            tracing::info!(task_id = %task_id, "Task completed: {:?}", outcome);
        }
        Err(e) => {
            tracing::error!(task_id = %task_id, "Task failed: {}", e);
            let _ = update_task_status(config, &task_id, &TaskStatus::Failed).await;
        }
    }

    result
}

/// Continue a waiting task (called when user replies)
pub async fn continue_task(
    config: &Config,
    task_id: &str,
    iterative_config: &IterativeConfig,
) -> Result<TaskOutcome> {
    let mut task = get_task(config, task_id)
        .await?
        .ok_or_else(|| anyhow!("Task not found: {}", task_id))?;

    // Check if task is in waiting state
    let user_reply = match &task.status {
        TaskStatus::WaitingForUserInput(_) => {
            check_user_reply(config, task_id)
                .await?
                .ok_or_else(|| anyhow!("Task is waiting but no user reply found"))?
        }
        _ => return Err(anyhow!("Task is not in waiting state")),
    };

    // Clear user reply
    clear_user_reply(config, task_id).await?;

    // Reconstruct execution context (simplified - in production would restore full state)
    let security = Arc::new(crate::security::SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));

    let runtime: Arc<dyn crate::runtime::RuntimeAdapter> =
        Arc::from(crate::runtime::create_runtime(&config.runtime)?);

    let memory: Arc<dyn Memory> = Arc::from(crate::memory::create_memory_with_storage(
        &config.memory,
        Some(&config.storage.provider.config),
        &config.workspace_dir,
        config.api_key.as_deref(),
    )?);

    let tools_registry = tools::all_tools_with_runtime(
        Arc::new(config.clone()),
        &security,
        runtime,
        memory.clone(),
        None,
        None,
        &config.browser,
        &config.http_request,
        &config.workspace_dir,
        &config.agents,
        config.api_key.as_deref(),
        config,
    );

    let provider_name = config.default_provider.as_deref().unwrap_or("openrouter");
    let provider: Box<dyn Provider> = crate::providers::create_routed_provider(
        provider_name,
        config.api_key.as_deref(),
        config.api_url.as_deref(),
        &config.reliability,
        &config.model_routes,
        config.default_model.as_deref().unwrap_or("claude-sonnet-4"),
    )?;

    // Rebuild history (simplified - should restore from checkpoint)
    let system_prompt = format!("{}", REACT_SYSTEM_PROMPT);
    let mut history = vec![
        ChatMessage::system(&system_prompt),
        ChatMessage::user(&format!("Task: {}\n\nUser feedback: {}", task.goal, user_reply)),
    ];

    // Resume execution
    execution_loop(
        config,
        &mut task,
        iterative_config,
        provider.as_ref(),
        &tools_registry,
        &mut history,
    )
    .await
}

/// Main execution loop
async fn execution_loop(
    config: &Config,
    task: &mut Task,
    iterative_config: &IterativeConfig,
    provider: &dyn Provider,
    tools_registry: &[Box<dyn Tool>],
    history: &mut Vec<ChatMessage>,
) -> Result<TaskOutcome> {
    let detection_config = &iterative_config.stuck_detection;

    loop {
        // Check for user reply if waiting
        if matches!(task.status, TaskStatus::WaitingForUserInput(_)) {
            if let Some(reply) = check_user_reply(config, &task.task_id).await? {
                clear_user_reply(config, &task.task_id).await?;
                history.push(ChatMessage::user(&format!(
                    "User feedback: {}",
                    reply
                )));
                task.status = TaskStatus::Running;
                update_task_status(config, &task.task_id, &task.status).await?;
            } else {
                // Still waiting
                sleep(Duration::from_secs(10)).await;
                continue;
            }
        }

        // Detect stuck condition
        if let Some(report) =
            detect_stuck(task, config, detection_config, provider).await?
        {
            // Notify user via channels
            notify_user_via_channels(config, task, &report).await?;

            // Update task status
            task.status = TaskStatus::WaitingForUserInput(report.clone());
            update_task_status(config, &task.task_id, &task.status).await?;

            return Ok(TaskOutcome::WaitingForUser {
                task_id: task.task_id.clone(),
                report,
            });
        }

        // Execute one step
        let step_result = execute_step(
            task,
            config,
            provider,
            tools_registry,
            history,
        )
        .await;

        match step_result {
            Ok(StepResult::Completed) => {
                update_task_status(config, &task.task_id, &TaskStatus::Completed).await?;
                return Ok(TaskOutcome::Success {
                    task_id: task.task_id.clone(),
                    total_steps: task.current_step,
                });
            }
            Ok(StepResult::Continue) => {
                task.current_step += 1;
                update_task_step(config, &task.task_id, task.current_step, Some(history.len()))
                    .await?;
            }
            Err(e) => {
                // Check if error is retryable
                let error_str = e.to_string();
                let category = ErrorCategory::classify(&error_str);

                match category.action() {
                    ErrorAction::AutoRetry => {
                        tracing::warn!("Retryable error: {}", e);
                        // Add error to history and continue
                        history.push(ChatMessage::user(&format!(
                            "Previous action failed: {}. Retrying...",
                            e
                        )));
                        continue;
                    }
                    ErrorAction::AskUser => {
                        // Notify user of error
                        let report = BlockerReport::llm_detected(
                            format!("Execution error: {}", e),
                            &[],
                            &[],
                        );
                        notify_user_via_channels(config, task, &report).await?;

                        task.status = TaskStatus::WaitingForUserInput(report.clone());
                        update_task_status(config, &task.task_id, &task.status).await?;

                        return Ok(TaskOutcome::WaitingForUser {
                            task_id: task.task_id.clone(),
                            report,
                        });
                    }
                }
            }
        }
    }
}

enum StepResult {
    Completed,
    Continue,
}

/// Execute a single step
async fn execute_step(
    task: &Task,
    config: &Config,
    provider: &dyn Provider,
    tools_registry: &[Box<dyn Tool>],
    history: &mut Vec<ChatMessage>,
) -> Result<StepResult> {
    // Get model response in ReAct format
    let model = config
        .default_model
        .as_deref()
        .unwrap_or("claude-sonnet-4");

    let response = provider
        .chat_with_history(history, model, config.default_temperature)
        .await?;

    // Parse ReAct response
    let react_output = parse_react_response(&response)
        .context("Failed to parse ReAct response")?;

    // Log thought
    let thought = react_output.thought.clone();

    // If no tool calls, task is complete
    if react_output.tool_calls.is_empty() {
        // Log completion
        log_action(
            config,
            &task.task_id,
            task.current_step,
            Some(&thought),
            "completion",
            "Task completed",
            ActionResultStatus::Success,
        )
        .await?;

        history.push(ChatMessage::assistant(&response));
        return Ok(StepResult::Completed);
    }

    // Execute tool calls
    let mut results = Vec::new();
    let mut action_summaries = Vec::new();

    for tool_call in &react_output.tool_calls {
        let tool_result = execute_tool_call(tools_registry, tool_call).await;

        let (summary, status) = match &tool_result {
            Ok(result) => {
                let summary = format!("{}: {}", tool_call.name, summarize_result(result));
                let status = if result.success {
                    ActionResultStatus::Success
                } else {
                    ActionResultStatus::Failure
                };
                (summary, status)
            }
            Err(e) => {
                let summary = format!("{}: error - {}", tool_call.name, e);
                (summary, ActionResultStatus::Failure)
            }
        };

        action_summaries.push(ActionSummary {
            step_number: task.current_step,
            action_type: tool_call.name.clone(),
            action_summary: summary.clone(),
            result_status: status.clone(),
            created_at: chrono::Utc::now(),
        });

        results.push((tool_call, tool_result, status));
    }

    // Log actions
    for summary in &action_summaries {
        log_action(
            config,
            &task.task_id,
            task.current_step,
            Some(&thought),
            &summary.action_type,
            &summary.action_summary,
            summary.result_status.clone(),
        )
        .await?;
    }

    // Build result message for LLM
    let mut result_message = String::new();
    for (tool_call, result, _) in results {
        match result {
            Ok(tool_result) => {
                result_message.push_str(&format!(
                    "Tool: {}\nResult: {}\nError: {}\n\n",
                    tool_call.name,
                    tool_result.output,
                    tool_result.error.unwrap_or_default()
                ));
            }
            Err(e) => {
                result_message.push_str(&format!(
                    "Tool: {}\nError: {}\n\n",
                    tool_call.name, e
                ));
            }
        }
    }

    // Update history
    history.push(ChatMessage::assistant(&response));
    history.push(ChatMessage::user(&result_message));

    Ok(StepResult::Continue)
}

/// Execute a single tool call
async fn execute_tool_call(
    tools_registry: &[Box<dyn Tool>],
    tool_call: &ToolCall,
) -> Result<ToolResult> {
    let tool = tools_registry
        .iter()
        .find(|t| t.name() == tool_call.name)
        .ok_or_else(|| anyhow!("Unknown tool: {}", tool_call.name))?;

    let args: serde_json::Value =
        serde_json::from_str(&tool_call.arguments).unwrap_or_else(|_| serde_json::json!({}));

    let result = tool.execute(args).await?;
    Ok(result)
}

/// Summarize tool result for logging
fn summarize_result(result: &ToolResult) -> String {
    if result.success {
        let output = &result.output;
        if output.len() > 100 {
            format!("{}... (truncated)", &output[..100])
        } else {
            output.clone()
        }
    } else {
        format!("Failed: {}", result.error.as_deref().unwrap_or("unknown error"))
    }
}

/// Notify user via configured channels
async fn notify_user_via_channels(
    _config: &Config,
    task: &Task,
    report: &BlockerReport,
) -> Result<()> {
    let message = format_blocker_message(task, report);

    // TODO: Integrate with existing channel system (Telegram, Lark, etc.)
    // For now, print to stderr for CLI usage
    tracing::info!("Task {} waiting for user input", task.task_id);
    
    eprintln!("\n{}", message);
    eprintln!("\n[Waiting for user reply...]\n");

    Ok(())
}

/// Format blocker report for user notification
fn format_blocker_message(task: &Task, report: &BlockerReport) -> String {
    let recent_actions = report
        .recent_actions
        .iter()
        .map(|a| format!("- {}: {}", a.action_type, a.action_summary))
        .collect::<Vec<_>>()
        .join("\n");

    let recent_thoughts = if report.recent_thoughts.is_empty() {
        "N/A".to_string()
    } else {
        report
            .recent_thoughts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("Step {}: {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "🤔 我在执行任务时遇到了一些困难，需要你的建议\n\n\
         **任务ID**: {}\n\
         **目标**: {}\n\
         **当前步数**: {}\n\
         **检测原因**: {:?}\n\n\
         **最近的思考**:\n{}\n\n\
         **最近的行动**:\n{}\n\n\
         **建议**: {}\n\n\
         请回复建议，例如:\n\
         - \"尝试 xxx 方法\"\n\
         - \"先检查 yyy 配置\"\n\
         - \"跳过这步，直接做 zzz\"\n\
         - \"停止任务\"",
        task.task_id,
        task.goal,
        task.current_step,
        report.blocker_type,
        recent_thoughts,
        recent_actions,
        report.suggestion_prompt
    )
}

/// List all tasks
pub async fn list_tasks(config: &Config, status_filter: Option<&str>) -> Result<Vec<Task>> {
    super::persistence::list_tasks(config, status_filter).await
}

/// Get task status
pub async fn get_task_status(config: &Config, task_id: &str) -> Result<Option<TaskStatus>> {
    let task = get_task(config, task_id).await?;
    Ok(task.map(|t| t.status))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_result() {
        let result = ToolResult {
            success: true,
            output: "a".repeat(200),
            error: None,
        };
        let summary = summarize_result(&result);
        assert!(summary.contains("truncated"));

        let result_fail = ToolResult {
            success: false,
            output: String::new(),
            error: Some("command not found".to_string()),
        };
        let summary = summarize_result(&result_fail);
        assert!(summary.contains("Failed"));
    }
}
