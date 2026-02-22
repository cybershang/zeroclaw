use super::types::{
    ActionResultStatus, ActionSummary, Task, TaskStatus,
};
use crate::config::Config;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension};

/// Initialize database tables
pub async fn init_tables(config: &Config) -> Result<()> {
    let db_path = config.workspace_dir.join(".zeroclaw").join("iterative.db");
    tokio::fs::create_dir_all(db_path.parent().unwrap()).await?;

    let conn = rusqlite::Connection::open(&db_path)?;

    // Tasks table
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS iterative_tasks (
            task_id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            current_step INTEGER DEFAULT 0,
            context_checkpoint INTEGER,
            blocker_report TEXT,
            user_reply TEXT,
            replied_at INTEGER
        )
        "#,
        [],
    )?;

    // Action log table for stuck detection
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS task_action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES iterative_tasks(task_id),
            step_number INTEGER NOT NULL,
            thought TEXT,
            action_type TEXT NOT NULL,
            action_summary TEXT NOT NULL,
            result_status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        "#,
        [],
    )?;

    // Index for efficient querying
    conn.execute(
        r#"
        CREATE INDEX IF NOT EXISTS idx_task_log_task_id 
        ON task_action_log(task_id, step_number)
        "#,
        [],
    )?;

    Ok(())
}

/// Get database connection
fn get_conn(config: &Config) -> Result<rusqlite::Connection> {
    let db_path = config.workspace_dir.join(".zeroclaw").join("iterative.db");
    rusqlite::Connection::open(&db_path).context("Failed to open iterative database")
}

/// Create a new task
pub async fn create_task(config: &Config, task_id: &str, goal: &str) -> Result<Task> {
    let now = Utc::now();
    let task = Task {
        task_id: task_id.to_string(),
        goal: goal.to_string(),
        status: TaskStatus::Running,
        created_at: now,
        updated_at: now,
        current_step: 0,
        context_checkpoint: None,
        user_reply: None,
        replied_at: None,
    };

    let conn = get_conn(config)?;
    conn.execute(
        r#"
        INSERT INTO iterative_tasks 
        (task_id, goal, status, created_at, updated_at, current_step)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
        params![
            task.task_id,
            task.goal,
            serde_json::to_string(&task.status)?,
            task.created_at.timestamp(),
            task.updated_at.timestamp(),
            task.current_step
        ],
    )?;

    Ok(task)
}

/// Get task by ID
pub async fn get_task(config: &Config, task_id: &str) -> Result<Option<Task>> {
    let conn = get_conn(config)?;
    let mut stmt = conn.prepare(
        r#"
        SELECT task_id, goal, status, created_at, updated_at, 
               current_step, context_checkpoint, blocker_report, user_reply, replied_at
        FROM iterative_tasks WHERE task_id = ?1
        "#,
    )?;

    let task = stmt
        .query_row([task_id], |row| {
            let status_str: String = row.get(2)?;
            let mut status: TaskStatus =
                serde_json::from_str(&status_str).unwrap_or(TaskStatus::Running);
            
            // If status is WaitingForUserInput, try to load blocker_report
            if matches!(status, TaskStatus::WaitingForUserInput(_)) {
                let blocker_json: Option<String> = row.get(7)?;
                if let Some(json) = blocker_json {
                    if let Ok(report) = serde_json::from_str::<super::types::BlockerReport>(&json) {
                        status = TaskStatus::WaitingForUserInput(report);
                    }
                }
            }

            Ok(Task {
                task_id: row.get(0)?,
                goal: row.get(1)?,
                status,
                created_at: timestamp_to_datetime(row.get(3)?),
                updated_at: timestamp_to_datetime(row.get(4)?),
                current_step: row.get(5)?,
                context_checkpoint: row.get(6)?,
                user_reply: row.get(8)?,
                replied_at: row.get::<_, Option<i64>>(9)?.map(timestamp_to_datetime),
            })
        })
        .optional()?;

    Ok(task)
}

/// Update task status
pub async fn update_task_status(
    config: &Config,
    task_id: &str,
    status: &TaskStatus,
) -> Result<()> {
    let conn = get_conn(config)?;
    let now = Utc::now().timestamp();

    let status_str = serde_json::to_string(status)?;

    match status {
        TaskStatus::WaitingForUserInput(report) => {
            let report_str = serde_json::to_string(report)?;
            conn.execute(
                r#"
                UPDATE iterative_tasks 
                SET status = ?1, updated_at = ?2, blocker_report = ?3
                WHERE task_id = ?4
                "#,
                params![status_str, now, report_str, task_id],
            )?;
        }
        _ => {
            conn.execute(
                r#"
                UPDATE iterative_tasks 
                SET status = ?1, updated_at = ?2
                WHERE task_id = ?3
                "#,
                params![status_str, now, task_id],
            )?;
        }
    }

    Ok(())
}

/// Update task step counter
pub async fn update_task_step(
    config: &Config,
    task_id: &str,
    step: usize,
    checkpoint: Option<usize>,
) -> Result<()> {
    let conn = get_conn(config)?;
    let now = Utc::now().timestamp();

    conn.execute(
        r#"
        UPDATE iterative_tasks 
        SET current_step = ?1, context_checkpoint = ?2, updated_at = ?3
        WHERE task_id = ?4
        "#,
        params![step, checkpoint, now, task_id],
    )?;

    Ok(())
}

/// Store user reply
pub async fn store_user_reply(config: &Config, task_id: &str, reply: &str) -> Result<()> {
    let conn = get_conn(config)?;
    let now = Utc::now().timestamp();

    conn.execute(
        r#"
        UPDATE iterative_tasks 
        SET user_reply = ?1, replied_at = ?2, updated_at = ?3, status = 'running'
        WHERE task_id = ?4
        "#,
        params![reply, now, now, task_id],
    )?;

    Ok(())
}

/// Log an action
pub async fn log_action(
    config: &Config,
    task_id: &str,
    step_number: usize,
    thought: Option<&str>,
    action_type: &str,
    action_summary: &str,
    result_status: ActionResultStatus,
) -> Result<()> {
    let conn = get_conn(config)?;
    let now = Utc::now().timestamp();

    conn.execute(
        r#"
        INSERT INTO task_action_log 
        (task_id, step_number, thought, action_type, action_summary, result_status, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#,
        params![
            task_id,
            step_number,
            thought,
            action_type,
            action_summary,
            serde_json::to_string(&result_status)?,
            now
        ],
    )?;

    Ok(())
}

/// Get recent actions for stuck detection
pub async fn get_recent_actions(
    config: &Config,
    task_id: &str,
    limit: usize,
) -> Result<Vec<ActionSummary>> {
    let conn = get_conn(config)?;
    let mut stmt = conn.prepare(
        r#"
        SELECT step_number, action_type, action_summary, result_status, created_at
        FROM task_action_log
        WHERE task_id = ?1
        ORDER BY step_number DESC
        LIMIT ?2
        "#,
    )?;

    let actions: Result<Vec<_>> = stmt
        .query_map(params![task_id, limit], |row| {
            let status_str: String = row.get(3)?;
            let result_status: ActionResultStatus =
                serde_json::from_str(&status_str).unwrap_or(ActionResultStatus::Failure);

            Ok(ActionSummary {
                step_number: row.get(0)?,
                action_type: row.get(1)?,
                action_summary: row.get(2)?,
                result_status,
                created_at: timestamp_to_datetime(row.get(4)?),
            })
        })?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to collect actions");

    let mut actions = actions?;
    actions.reverse(); // Oldest first
    Ok(actions)
}

/// Get recent thoughts for LLM assessment
pub async fn get_recent_thoughts(
    config: &Config,
    task_id: &str,
    limit: usize,
) -> Result<Vec<String>> {
    let conn = get_conn(config)?;
    let mut stmt = conn.prepare(
        r#"
        SELECT thought
        FROM task_action_log
        WHERE task_id = ?1 AND thought IS NOT NULL
        ORDER BY step_number DESC
        LIMIT ?2
        "#,
    )?;

    let thoughts: Vec<String> = stmt
        .query_map(params![task_id, limit], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    let mut thoughts = thoughts;
    thoughts.reverse();
    Ok(thoughts)
}

/// List all tasks (with optional status filter)
pub async fn list_tasks(
    config: &Config,
    status_filter: Option<&str>,
) -> Result<Vec<Task>> {
    let conn = get_conn(config)?;

    let query = if status_filter.is_some() {
        r#"
        SELECT task_id, goal, status, created_at, updated_at, 
               current_step, context_checkpoint, blocker_report, user_reply, replied_at
        FROM iterative_tasks 
        WHERE status = ?1
        ORDER BY updated_at DESC
        "#
    } else {
        r#"
        SELECT task_id, goal, status, created_at, updated_at, 
               current_step, context_checkpoint, blocker_report, user_reply, replied_at
        FROM iterative_tasks 
        ORDER BY updated_at DESC
        "#
    };

    let mut stmt = conn.prepare(query)?;

    let rows: Vec<Task> = if let Some(status) = status_filter {
        stmt
            .query_map([status], row_to_task)?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to collect tasks")?
    } else {
        stmt
            .query_map([], row_to_task)?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to collect tasks")?
    };

    Ok(rows)
}

fn row_to_task(row: &rusqlite::Row) -> Result<Task, rusqlite::Error> {
    let status_str: String = row.get(2)?;
    let mut status: TaskStatus = serde_json::from_str(&status_str).unwrap_or(TaskStatus::Running);
    
    // If status is WaitingForUserInput, try to load blocker_report from column 7
    if matches!(status, TaskStatus::WaitingForUserInput(_)) {
        let blocker_json: Option<String> = row.get(7)?;
        if let Some(json) = blocker_json {
            if let Ok(report) = serde_json::from_str::<super::types::BlockerReport>(&json) {
                status = TaskStatus::WaitingForUserInput(report);
            }
        }
    }

    Ok(Task {
        task_id: row.get(0)?,
        goal: row.get(1)?,
        status,
        created_at: timestamp_to_datetime(row.get(3)?),
        updated_at: timestamp_to_datetime(row.get(4)?),
        current_step: row.get(5)?,
        context_checkpoint: row.get(6)?,
        user_reply: row.get(8)?,
        replied_at: row.get::<_, Option<i64>>(9)?.map(timestamp_to_datetime),
    })
}

fn timestamp_to_datetime(ts: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now())
}

/// Check and retrieve user reply for a waiting task
pub async fn check_user_reply(config: &Config, task_id: &str) -> Result<Option<String>> {
    let conn = get_conn(config)?;
    let mut stmt = conn.prepare(
        r#"
        SELECT user_reply FROM iterative_tasks 
        WHERE task_id = ?1 AND user_reply IS NOT NULL
        "#,
    )?;

    let reply: Option<String> = stmt.query_row([task_id], |row| row.get(0)).optional()?;

    Ok(reply)
}

/// Clear user reply after processing
pub async fn clear_user_reply(config: &Config, task_id: &str) -> Result<()> {
    let conn = get_conn(config)?;

    conn.execute(
        r#"
        UPDATE iterative_tasks 
        SET user_reply = NULL, replied_at = NULL
        WHERE task_id = ?1
        "#,
        [task_id],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config() -> Config {
        let tmp = TempDir::new().unwrap();
        Config {
            workspace_dir: tmp.path().to_path_buf(),
            ..Config::default()
        }
    }

    #[tokio::test]
    async fn test_create_and_get_task() {
        let config = test_config();
        init_tables(&config).await.unwrap();

        let task = create_task(&config, "test-1", "Test goal").await.unwrap();
        assert_eq!(task.task_id, "test-1");
        assert_eq!(task.goal, "Test goal");

        let retrieved = get_task(&config, "test-1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().goal, "Test goal");
    }

    #[tokio::test]
    async fn test_update_status() {
        let config = test_config();
        init_tables(&config).await.unwrap();

        create_task(&config, "test-2", "Test").await.unwrap();

        update_task_status(&config, "test-2", &TaskStatus::Completed)
            .await
            .unwrap();

        let task = get_task(&config, "test-2").await.unwrap().unwrap();
        assert!(matches!(task.status, TaskStatus::Completed));
    }

    #[tokio::test]
    async fn test_log_and_retrieve_actions() {
        let config = test_config();
        init_tables(&config).await.unwrap();

        create_task(&config, "test-3", "Test").await.unwrap();

        log_action(
            &config,
            "test-3",
            1,
            Some("Let me check the file"),
            "file_read",
            "src/main.rs",
            ActionResultStatus::Success,
        )
        .await
        .unwrap();

        let actions = get_recent_actions(&config, "test-3", 10).await.unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, "file_read");
    }
}
