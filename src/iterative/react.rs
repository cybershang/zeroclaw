use super::types::ReActOutput;
use crate::providers::{ToolCall};
use anyhow::{anyhow, Context, Result};
use regex::Regex;

/// System prompt for ReAct mode
pub const REACT_SYSTEM_PROMPT: &str = r#"You are an autonomous coding agent. You must analyze the current situation and decide on the next action.

IMPORTANT: You MUST follow this exact format in your response:

<thought>
Analyze the current progress:
1. What was the result of the previous action?
2. Are we making progress toward the goal?
3. What should be done next?
4. Is this a new approach or are we repeating previous attempts?
</thought>

<action>
[Tool calls in JSON format]
</action>

Guidelines:
- Be thorough in your analysis. If something failed, explain why and how to fix it.
- If you've tried an approach multiple times without success, acknowledge this and consider alternatives.
- Tool calls should be specific and actionable.
- If the task is complete, output empty tool calls: []
"#;

/// Parse ReAct format response
pub fn parse_react_response(response: &str) -> Result<ReActOutput> {
    // Extract thought
    let thought = extract_thought(response)?;

    // Extract tool calls
    let tool_calls = extract_tool_calls(response)?;

    Ok(ReActOutput {
        thought,
        tool_calls,
        raw_response: response.to_string(),
    })
}

fn extract_thought(response: &str) -> Result<String> {
    // Try to extract <thought>...</thought>
    let thought_re = Regex::new(r"<thought>(.*?)</thought>")?;
    
    if let Some(caps) = thought_re.captures(response) {
        let thought = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        return Ok(thought.to_string());
    }

    // Fallback: if no thought tags, try to extract from the beginning
    if let Some(action_start) = response.find("<action>") {
        let thought = &response[..action_start].trim();
        if !thought.is_empty() {
            return Ok(thought.to_string());
        }
    }

    Err(anyhow!(
        "Could not extract <thought> from response. Expected format: <thought>...</thought>"
    ))
}

fn extract_tool_calls(response: &str) -> Result<Vec<ToolCall>> {
    // Try to extract <action>...</action>
    let action_re = Regex::new(r"<action>(.*?)</action>")?;

    let action_content = if let Some(caps) = action_re.captures(response) {
        caps.get(1).map(|m| m.as_str().trim()).unwrap_or("[]")
    } else {
        // Try to find JSON array directly
        find_json_array(response)?
    };

    // Parse tool calls from JSON
    parse_tool_calls_json(action_content)
}

fn find_json_array(response: &str) -> Result<&str> {
    // Find the first JSON array in the response
    let mut depth = 0;
    let mut start = None;

    for (i, ch) in response.char_indices() {
        match ch {
            '[' if depth == 0 => {
                depth = 1;
                start = Some(i);
            }
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        return Ok(&response[s..=i]);
                    }
                }
            }
            _ => {}
        }
    }

    Err(anyhow!("Could not find JSON array in response"))
}

fn parse_tool_calls_json(content: &str) -> Result<Vec<ToolCall>> {
    let value: serde_json::Value = serde_json::from_str(content)
        .context("Failed to parse tool calls JSON")?;

    let calls = match value {
        serde_json::Value::Array(arr) => arr,
        _ => return Err(anyhow!("Tool calls must be a JSON array")),
    };

    let mut tool_calls = Vec::new();

    for call in calls {
        let id = call
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let name = call
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Tool call missing 'name' field"))?
            .to_string();

        let arguments = call
            .get("arguments")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "{}".to_string());

        tool_calls.push(ToolCall {
            id,
            name,
            arguments,
        });
    }

    Ok(tool_calls)
}

/// Build assessment prompt for LLM progress evaluation
pub fn build_assessment_prompt(
    goal: &str,
    thoughts: &[String],
    actions: &[super::types::ActionSummary],
) -> String {
    let thoughts_str = thoughts
        .iter()
        .enumerate()
        .map(|(i, t)| format!("Step {}: {}", i + 1, t))
        .collect::<Vec<_>>()
        .join("\n\n");

    let actions_str = actions
        .iter()
        .map(|a| {
            format!(
                "- {} ({}): {} -> {:?}",
                a.action_type, a.step_number, a.action_summary, a.result_status
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"As a project supervisor, evaluate whether the following execution is making effective progress toward the goal.

Goal: {}

Recent thoughts:
{}

Recent actions:
{}

Analyze and respond in this exact format:

<assessment>
is_progressing: [true/false]
is_repeating: [true/false]  
needs_human: [true/false]
reason: [Brief explanation of your assessment]
</assessment>

Guidelines:
- is_progressing: Are we getting closer to the goal?
- is_repeating: Are we doing the same thing over and over without success?
- needs_human: Should we ask the user for guidance?
"#,
        goal, thoughts_str, actions_str
    )
}

/// Parse assessment response
pub fn parse_assessment_response(response: &str) -> Result<super::types::ProgressAssessment> {
    // Use (?s) flag to make . match newlines, and be more flexible with whitespace
    let assessment_re = Regex::new(
        r"(?s)is_progressing:\s*(true|false).*?is_repeating:\s*(true|false).*?needs_human:\s*(true|false).*?reason:\s*(.+?)(?:</assessment>|$)"
    )?;

    let lower = response.to_lowercase();
    if let Some(caps) = assessment_re.captures(&lower) {
        let is_progressing = caps.get(1).map(|m| m.as_str() == "true").unwrap_or(true);
        let is_repeating = caps.get(2).map(|m| m.as_str() == "true").unwrap_or(false);
        let needs_human = caps.get(3).map(|m| m.as_str() == "true").unwrap_or(false);
        let reason = caps
            .get(4)
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_else(|| "No reason provided".to_string());

        return Ok(super::types::ProgressAssessment {
            is_progressing,
            is_repeating,
            needs_human,
            reason,
        });
    }

    // Fallback: line-by-line parsing
    let mut is_progressing = true;
    let mut is_repeating = false;
    let mut needs_human = false;
    let mut reason = "Parsed from assessment response".to_string();

    for line in lower.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("is_progressing:") {
            is_progressing = val.trim() == "true";
        } else if let Some(val) = line.strip_prefix("is_repeating:") {
            is_repeating = val.trim() == "true";
        } else if let Some(val) = line.strip_prefix("needs_human:") {
            needs_human = val.trim() == "true";
        } else if let Some(val) = line.strip_prefix("reason:") {
            reason = val.trim().to_string();
        }
    }

    Ok(super::types::ProgressAssessment {
        is_progressing,
        is_repeating,
        needs_human,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterative::types::{ActionResultStatus, ActionSummary};

    #[test]
    fn test_parse_react_response() {
        let response = r#"
<thought>
The previous build failed because of missing dependencies. I should install them first.
</thought>

<action>
[{"id": "1", "name": "shell", "arguments": "{\"command\": \"cargo build\"}"}]
</action>
"#;

        let result = parse_react_response(response).unwrap();
        assert!(result.thought.contains("missing dependencies"));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "shell");
    }

    #[test]
    fn test_parse_assessment_response() {
        let response = r#"
<assessment>
is_progressing: true
is_repeating: false
needs_human: false
reason: Successfully compiled the code and tests are passing
</assessment>
"#;

        let result = parse_assessment_response(response).unwrap();
        assert!(result.is_progressing);
        assert!(!result.is_repeating);
        assert!(!result.needs_human);
    }

    #[test]
    fn test_build_assessment_prompt() {
        let thoughts = vec!["Checking file structure".to_string()];
        let actions = vec![ActionSummary {
            step_number: 1,
            action_type: "file_read".to_string(),
            action_summary: "Cargo.toml".to_string(),
            result_status: ActionResultStatus::Success,
            created_at: chrono::Utc::now(),
        }];

        let prompt = build_assessment_prompt("Build a Rust project", &thoughts, &actions);
        assert!(prompt.contains("Build a Rust project"));
        assert!(prompt.contains("Checking file structure"));
        assert!(prompt.contains("file_read"));
    }
}
