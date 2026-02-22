//! Iterative task execution module for autonomous operation.
//!
//! This module provides the capability to run long-running tasks with
//! automatic stuck detection and user notification via configured channels.

pub mod detector;
pub mod persistence;
pub mod react;
pub mod runner;
pub mod types;

pub use types::{
    BlockerReport, BlockerType, ErrorAction, ErrorCategory, ProgressAssessment,
    TaskOutcome, TaskStatus,
};

pub use runner::{run_iterative, continue_task, list_tasks};

use crate::config::Config;
use anyhow::Result;

/// Default configuration values
pub mod defaults {
    pub const DUPLICATE_ACTION_THRESHOLD: usize = 3;
    pub const ASSESS_INTERVAL: usize = 5;
    pub const LONG_RUNNING_THRESHOLD: usize = 50;
    pub const USER_TIMEOUT_MINUTES: u64 = 60;
}

/// Initialize the iterative module (create tables if needed)
pub async fn init(config: &Config) -> Result<()> {
    persistence::init_tables(config).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IterativeConfig;

    #[test]
    fn test_default_config() {
        let config = IterativeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.stuck_detection.duplicate_action_threshold, 3);
        assert_eq!(config.stuck_detection.assess_interval, 5);
    }
}
