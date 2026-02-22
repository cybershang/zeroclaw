use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Task execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Running,
    WaitingForUserInput(BlockerReport),
    Completed,
    Failed,
}

/// Task structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub goal: String,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub current_step: usize,
    /// Checkpoint in conversation history (for context reconstruction)
    pub context_checkpoint: Option<usize>,
    /// User reply when waiting for input
    pub user_reply: Option<String>,
    pub replied_at: Option<DateTime<Utc>>,
}

// Re-export config types from config module
pub use crate::config::{IterativeConfig, StuckDetectionConfig};

/// Type of blocker detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data")]
pub enum BlockerType {
    /// Repeated same action multiple times
    DuplicateAction { action: String, count: usize },
    /// No meaningful progress detected by LLM
    NoProgress { reason: String, steps: usize },
    /// Consecutive failures
    ConsecutiveFailures { errors: Vec<String> },
    /// Long running without completion
    LongRunning { total_steps: usize },
    /// LLM detected stuck situation
    LlmDetected { reason: String },
}

/// Report sent to user when stuck is detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockerReport {
    pub blocker_type: BlockerType,
    pub recent_thoughts: Vec<String>,
    pub recent_actions: Vec<ActionSummary>,
    pub current_context: String,
    pub suggestion_prompt: String,
}

impl BlockerReport {
    pub fn repeating_actions(actions: &[ActionSummary]) -> Self {
        let action = actions.first().map(|a| a.action_type.clone()).unwrap_or_default();
        Self {
            blocker_type: BlockerType::DuplicateAction {
                action: action.clone(),
                count: actions.len(),
            },
            recent_thoughts: Vec::new(),
            recent_actions: actions.to_vec(),
            current_context: String::new(),
            suggestion_prompt: format!(
                "I've been repeatedly executing '{}' without success. Please advise on how to proceed.",
                action
            ),
        }
    }

    pub fn llm_detected(reason: String, thoughts: &[String], actions: &[ActionSummary]) -> Self {
        Self {
            blocker_type: BlockerType::LlmDetected {
                reason: reason.clone(),
            },
            recent_thoughts: thoughts.to_vec(),
            recent_actions: actions.to_vec(),
            current_context: String::new(),
            suggestion_prompt: format!(
                "I'm having difficulty making progress. My analysis: {}. Please advise.",
                reason
            ),
        }
    }

    pub fn long_running(total_steps: usize) -> Self {
        Self {
            blocker_type: BlockerType::LongRunning { total_steps },
            recent_thoughts: Vec::new(),
            recent_actions: Vec::new(),
            current_context: String::new(),
            suggestion_prompt: format!(
                "I've been working on this task for {} steps but haven't completed it yet. Should I continue or adjust my approach?",
                total_steps
            ),
        }
    }
}

/// Summary of an action for logging and detection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionSummary {
    pub step_number: usize,
    pub action_type: String,
    pub action_summary: String,
    pub result_status: ActionResultStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionResultStatus {
    Success,
    Failure,
    Skipped,
}

/// Progress assessment from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressAssessment {
    pub is_progressing: bool,
    pub is_repeating: bool,
    pub needs_human: bool,
    pub reason: String,
}

/// Error categorization for retry decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCategory {
    Network { retryable: bool },
    Permission { resource: String },
    InvalidInput { details: String },
    Unknown { message: String },
}

impl ErrorCategory {
    pub fn classify(error: &str) -> Self {
        let lower = error.to_lowercase();

        if lower.contains("timeout")
            || lower.contains("connection")
            || lower.contains("network")
            || lower.contains("econnrefused")
            || lower.contains("dns")
        {
            return ErrorCategory::Network { retryable: true };
        }

        if lower.contains("permission denied")
            || lower.contains("access denied")
            || lower.contains("not allowed")
            || lower.contains("forbidden")
            || lower.contains("unauthorized")
        {
            return ErrorCategory::Permission {
                resource: extract_resource_hint(&lower),
            };
        }

        if lower.contains("no such file")
            || lower.contains("command not found")
            || lower.contains("invalid")
            || lower.contains("not found")
            || lower.contains("does not exist")
        {
            return ErrorCategory::InvalidInput {
                details: error.to_string(),
            };
        }

        ErrorCategory::Unknown {
            message: error.to_string(),
        }
    }

    pub fn action(&self) -> ErrorAction {
        match self {
            ErrorCategory::Network { .. } => ErrorAction::AutoRetry,
            _ => ErrorAction::AskUser,
        }
    }
}

fn extract_resource_hint(error: &str) -> String {
    if error.contains("file") || error.contains("path") {
        "file/path".to_string()
    } else if error.contains("network") || error.contains("connection") {
        "network".to_string()
    } else if error.contains("api") || error.contains("token") {
        "api/token".to_string()
    } else {
        "resource".to_string()
    }
}

/// Action to take based on error category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorAction {
    AutoRetry,
    AskUser,
}

/// Task execution outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskOutcome {
    Success { task_id: String, total_steps: usize },
    Failed { task_id: String, reason: String },
    WaitingForUser { task_id: String, report: BlockerReport },
}

/// ReAct phase for thought/action separation
#[derive(Debug, Clone)]
pub enum ReActPhase {
    Reasoning { thought: String },
    Acting { tool_calls: Vec<crate::providers::ToolCall> },
}

/// Parsed ReAct output
#[derive(Debug, Clone)]
pub struct ReActOutput {
    pub thought: String,
    pub tool_calls: Vec<crate::providers::ToolCall>,
    pub raw_response: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        assert!(matches!(
            ErrorCategory::classify("connection timeout"),
            ErrorCategory::Network { .. }
        ));

        assert!(matches!(
            ErrorCategory::classify("permission denied: /etc/passwd"),
            ErrorCategory::Permission { .. }
        ));

        assert!(matches!(
            ErrorCategory::classify("no such file or directory"),
            ErrorCategory::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_error_action() {
        let network_err = ErrorCategory::Network { retryable: true };
        assert_eq!(network_err.action(), ErrorAction::AutoRetry);

        let perm_err = ErrorCategory::Permission {
            resource: "test".to_string(),
        };
        assert_eq!(perm_err.action(), ErrorAction::AskUser);
    }
}
