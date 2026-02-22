use super::persistence::{get_recent_actions, get_recent_thoughts};
use super::react::{build_assessment_prompt, parse_assessment_response};
use super::types::{
    ActionResultStatus, ActionSummary, BlockerReport, StuckDetectionConfig, Task,
};
use crate::config::Config;
use crate::providers::Provider;
use anyhow::Result;

/// Detect if the task is stuck and needs user intervention
pub async fn detect_stuck(
    task: &Task,
    config: &Config,
    detection_config: &StuckDetectionConfig,
    provider: &dyn Provider,
) -> Result<Option<BlockerReport>> {
    // 1. Check for duplicate actions (fast path)
    if let Some(report) = check_duplicate_actions(task, config, detection_config).await? {
        return Ok(Some(report));
    }

    // 2. Check for consecutive failures
    if let Some(report) = check_consecutive_failures(task, config, detection_config).await? {
        return Ok(Some(report));
    }

    // 3. Check for long-running task
    if task.current_step >= detection_config.long_running_threshold {
        return Ok(Some(BlockerReport::long_running(task.current_step)));
    }

    // 4. LLM-based assessment (every N steps)
    if task.current_step > 0 && task.current_step % detection_config.assess_interval == 0 {
        if let Some(report) =
            assess_with_llm(task, config, detection_config, provider).await?
        {
            return Ok(Some(report));
        }
    }

    Ok(None)
}

/// Check for repeated identical or similar actions
async fn check_duplicate_actions(
    task: &Task,
    config: &Config,
    detection_config: &StuckDetectionConfig,
) -> Result<Option<BlockerReport>> {
    let actions =
        get_recent_actions(config, &task.task_id, detection_config.duplicate_action_threshold)
            .await?;

    if actions.len() < detection_config.duplicate_action_threshold {
        return Ok(None);
    }

    // Check if all recent actions are the same type with the same summary
    let first = actions.first().unwrap();
    let all_same = actions.iter().all(|a| {
        a.action_type == first.action_type && a.action_summary == first.action_summary
    });

    if all_same {
        return Ok(Some(BlockerReport::repeating_actions(&actions)));
    }

    // Check for similar shell commands (e.g., retrying same build command)
    if first.action_type == "shell" {
        let command_pattern = extract_command_pattern(&first.action_summary);
        let similar_count = actions
            .iter()
            .filter(|a| {
                a.action_type == "shell"
                    && extract_command_pattern(&a.action_summary) == command_pattern
            })
            .count();

        if similar_count >= detection_config.duplicate_action_threshold {
            return Ok(Some(BlockerReport::repeating_actions(&actions)));
        }
    }

    Ok(None)
}

/// Extract command pattern for comparison (e.g., "cargo build" from "cargo build --release")
fn extract_command_pattern(command: &str) -> String {
    // Take first 2-3 words as the pattern
    let words: Vec<&str> = command.split_whitespace().take(3).collect();
    words.join(" ")
}

/// Check for consecutive failures
async fn check_consecutive_failures(
    task: &Task,
    config: &Config,
    detection_config: &StuckDetectionConfig,
) -> Result<Option<BlockerReport>> {
    let actions =
        get_recent_actions(config, &task.task_id, detection_config.consecutive_failure_threshold)
            .await?;

    if actions.len() < detection_config.consecutive_failure_threshold {
        return Ok(None);
    }

    let all_failed = actions
        .iter()
        .all(|a| matches!(a.result_status, ActionResultStatus::Failure));

    if all_failed {
        let errors: Vec<String> = actions
            .iter()
            .map(|a| format!("{}: {}", a.action_type, a.action_summary))
            .collect();

        return Ok(Some(BlockerReport {
            blocker_type: super::types::BlockerType::ConsecutiveFailures { errors },
            recent_thoughts: Vec::new(),
            recent_actions: actions,
            current_context: String::new(),
            suggestion_prompt: format!(
                "I've had {} consecutive failures. Please help me understand what went wrong.",
                detection_config.consecutive_failure_threshold
            ),
        }));
    }

    Ok(None)
}

/// Use LLM to assess progress
async fn assess_with_llm(
    task: &Task,
    config: &Config,
    _detection_config: &StuckDetectionConfig,
    provider: &dyn Provider,
) -> Result<Option<BlockerReport>> {
    // Get recent history
    let thoughts = get_recent_thoughts(config, &task.task_id, 5).await?;
    let actions = get_recent_actions(config, &task.task_id, 5).await?;

    if thoughts.len() < 2 {
        // Not enough history to assess
        return Ok(None);
    }

    // Build and send assessment prompt
    let prompt = build_assessment_prompt(&task.goal, &thoughts, &actions);

    // Use a lower temperature for more consistent assessment
    let response = provider
        .chat_with_system(None, &prompt, "assessment", 0.3)
        .await?;

    let assessment = parse_assessment_response(&response)?;

    // Only report if LLM thinks we need human help or are repeating
    if assessment.needs_human || assessment.is_repeating {
        return Ok(Some(BlockerReport::llm_detected(
            assessment.reason,
            &thoughts,
            &actions,
        )));
    }

    Ok(None)
}

/// Simple check if actions are similar (for quick comparison)
pub fn are_actions_similar(actions: &[ActionSummary]) -> bool {
    if actions.len() < 2 {
        return false;
    }

    let first = &actions[0];
    actions
        .iter()
        .skip(1)
        .all(|a| a.action_type == first.action_type && a.action_summary == first.action_summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterative::types::{ActionResultStatus, ActionSummary};
    use chrono::Utc;

    fn create_test_actions() -> Vec<ActionSummary> {
        vec![
            ActionSummary {
                step_number: 1,
                action_type: "shell".to_string(),
                action_summary: "cargo build".to_string(),
                result_status: ActionResultStatus::Failure,
                created_at: Utc::now(),
            },
            ActionSummary {
                step_number: 2,
                action_type: "shell".to_string(),
                action_summary: "cargo build".to_string(),
                result_status: ActionResultStatus::Failure,
                created_at: Utc::now(),
            },
            ActionSummary {
                step_number: 3,
                action_type: "shell".to_string(),
                action_summary: "cargo build".to_string(),
                result_status: ActionResultStatus::Failure,
                created_at: Utc::now(),
            },
        ]
    }

    #[test]
    fn test_extract_command_pattern() {
        assert_eq!(
            extract_command_pattern("cargo build --release"),
            "cargo build --release"
        );
        assert_eq!(extract_command_pattern("ls -la /home"), "ls -la /home");
    }

    #[test]
    fn test_are_actions_similar() {
        let actions = create_test_actions();
        assert!(are_actions_similar(&actions));

        let mut different = actions.clone();
        different[1].action_type = "file_read".to_string();
        assert!(!are_actions_similar(&different));
    }
}
