//! `grun`: submit one job and block until it reaches a terminal state,
//! streaming its log to stdout and exiting with a status derived from the
//! outcome. The srun to gbatch's sbatch.
//!
//! Waiting is event-driven off the daemon's `/events` SSE stream, with a slow
//! poll of the job as a fallback so a dropped stream never strands the caller.

use anyhow::{Context, Result};
use clap::Parser;
use gflow::client::Client;
use gflow::config::load_config;
use gflow::core::job::{Job, JobState};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;
use tokio::signal::unix::{signal, SignalKind};
use tokio_stream::{Stream, StreamExt};

mod cli;

/// Exit status when the job failed and no payload exit code is known.
const EXIT_FAILED: i32 = 1;
/// Exit status for a job killed by its time limit (mirrors `timeout(1)`).
const EXIT_TIMEOUT: i32 = 124;
/// Exit status for a cancelled job (mirrors a SIGINT-terminated process).
const EXIT_CANCELLED: i32 = 130;

/// Fallback poll of the job state, in case the event stream is unavailable
/// or drops an event.
const POLL_INTERVAL: Duration = Duration::from_secs(5);
/// How often new log output is forwarded to stdout.
const TAIL_INTERVAL: Duration = Duration::from_millis(250);

/// Variables never forwarded to the job: the daemon owns the first two, the
/// rest describe the submitting shell rather than the job.
const ENV_BLOCKLIST: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "GFLOW_ARRAY_TASK_ID",
    "_",
    "OLDPWD",
    "PWD",
    "SHLVL",
    "TMUX",
    "TMUX_PANE",
];

pub async fn run(argv: Vec<OsString>) -> Result<()> {
    let args = cli::GSrun::parse_from(argv);
    let config = load_config(args.config.as_ref())?;
    let client = Client::build(&config).context("Failed to build client")?;

    let add_args = args.srun_args.to_add_args();
    let mut job =
        crate::multicall::gbatch::commands::add::build_job(&add_args, None, &client, None).await?;
    apply_env_export(&mut job, &args.srun_args);
    crate::multicall::gbatch::commands::add::validate_project(&mut job, &config)?;

    if args.srun_args.dry_run {
        print_dry_run(&job);
        return Ok(());
    }

    let response = client.add_job(job).await.context("Failed to submit job")?;
    let job_id = response.id;
    eprintln!("grun: submitted job {} ({})", job_id, response.run_name);

    let code = wait_for_job(&client, job_id).await?;
    io::stdout().flush().ok();
    std::process::exit(code);
}

/// Export the caller's environment into the job unless disabled. An exported
/// environment already carries the active conda/pixi/virtualenv (`PATH`,
/// `CONDA_PREFIX`, ...), so the conda env that `gbatch` auto-detects from
/// `CONDA_DEFAULT_ENV` is dropped: activating it again in the job shell is
/// redundant, and fails outright for non-conda tools (pixi sets
/// `CONDA_DEFAULT_ENV` too). An explicit `--conda-env` is always honoured.
fn apply_env_export(job: &mut Job, args: &cli::SrunArgs) {
    if args.no_export_env {
        return;
    }
    job.env = Some(exported_env());
    if args.conda_env.is_none() {
        job.conda_env = None;
    }
}

/// The current environment, minus the blocklist and anything that is not a
/// valid shell identifier (the tmux executor re-exports it through a shell).
fn exported_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| !ENV_BLOCKLIST.contains(&key.as_str()) && is_shell_identifier(key))
        .collect()
}

fn is_shell_identifier(key: &str) -> bool {
    let mut chars = key.chars();
    matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn print_dry_run(job: &Job) {
    let cmd = job
        .command
        .as_ref()
        .map(ToString::to_string)
        .or_else(|| job.script.as_ref().map(|s| s.to_string_lossy().to_string()))
        .unwrap_or_default();
    println!("Would submit and wait for 1 job:");
    println!("  {} (GPUs: {})", cmd, job.gpus);
    println!("  run_dir: {}", job.run_dir.display());
    println!(
        "  env: {} variable(s) exported",
        job.env.as_ref().map_or(0, BTreeMap::len)
    );
}

/// Outcome-derived exit status. `payload_exit_code` is the job process's own
/// exit code when the daemon reported it (process executor); a failed job
/// without one maps to a generic failure.
fn exit_code_for(state: JobState, payload_exit_code: Option<i32>) -> i32 {
    match state {
        JobState::Finished => 0,
        JobState::Failed => match payload_exit_code {
            Some(code) if code != 0 => code,
            _ => EXIT_FAILED,
        },
        JobState::Timeout => EXIT_TIMEOUT,
        JobState::Cancelled => EXIT_CANCELLED,
        JobState::Queued | JobState::Hold | JobState::Running => unreachable!("non-terminal state"),
    }
}

/// One parsed server-sent event that concerns the job being followed.
enum JobEvent {
    /// The job's state may have changed; re-fetch it.
    Changed,
    /// The job's process exited with this code (process executor only).
    ExecutionFinished { exit_code: Option<i32> },
}

type EventStream = Pin<Box<dyn Stream<Item = JobEvent> + Send>>;

/// Subscribe to `/events` and keep only the events about `job_id`. `None` when
/// the daemon cannot be subscribed to; polling then carries the wait alone.
async fn subscribe(client: &Client, job_id: u32) -> Option<EventStream> {
    let response = match client.subscribe_events().await {
        Ok(response) => response,
        Err(error) => {
            tracing::debug!(%error, "event stream unavailable, falling back to polling");
            return None;
        }
    };

    let mut buffer = String::new();
    let stream = response.bytes_stream().filter_map(move |chunk| {
        let chunk = chunk.ok()?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        let mut relevant = None;
        // Events are blank-line separated; parse every complete one in the buffer.
        while let Some(end) = buffer.find("\n\n") {
            let raw = buffer[..end].to_string();
            buffer.drain(..end + 2);
            if let Some(event) = parse_sse_event(&raw, job_id) {
                relevant = Some(event);
            }
        }
        relevant
    });
    Some(Box::pin(stream))
}

/// Parse one raw SSE event block into a [`JobEvent`] if it concerns `job_id`.
fn parse_sse_event(raw: &str, job_id: u32) -> Option<JobEvent> {
    let mut name = "";
    let mut data = String::new();
    for line in raw.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            name = value.trim();
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push_str(value.trim_start());
        }
    }
    let payload: serde_json::Value = serde_json::from_str(&data).ok()?;
    if payload.get("job_id")?.as_u64()? != u64::from(job_id) {
        return None;
    }
    match name {
        "job_state_changed" | "job_completed" | "job_timed_out" => Some(JobEvent::Changed),
        "job_execution_finished" => Some(JobEvent::ExecutionFinished {
            exit_code: payload
                .get("exit_code")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32),
        }),
        _ => None,
    }
}

/// Forwards the job's log file to stdout incrementally.
struct LogTail {
    path: Option<PathBuf>,
    file: Option<std::fs::File>,
    offset: u64,
}

impl LogTail {
    fn new() -> Self {
        Self {
            path: None,
            file: None,
            offset: 0,
        }
    }

    /// Copy any new log bytes to stdout. Resolves the log path lazily (the
    /// daemon only knows it once the job has a run name) and opens the file
    /// once it exists.
    async fn pump(&mut self, client: &Client, job_id: u32) -> Result<()> {
        if self.path.is_none() {
            self.path = client
                .get_job_log_path(job_id)
                .await
                .ok()
                .flatten()
                .map(PathBuf::from);
        }
        if self.file.is_none() {
            if let Some(path) = &self.path {
                if let Ok(file) = std::fs::File::open(path) {
                    self.file = Some(file);
                }
            }
        }
        let Some(file) = self.file.as_mut() else {
            return Ok(());
        };

        file.seek(SeekFrom::Start(self.offset))?;
        let mut new_bytes = Vec::new();
        file.read_to_end(&mut new_bytes)?;
        if !new_bytes.is_empty() {
            self.offset += new_bytes.len() as u64;
            let mut stdout = io::stdout().lock();
            stdout.write_all(&new_bytes)?;
            stdout.flush()?;
        }
        Ok(())
    }
}

/// Fetch the job; `Some(state)` once it is terminal.
async fn terminal_state(client: &Client, job_id: u32) -> Result<Option<JobState>> {
    let job = client
        .get_job(job_id)
        .await?
        .with_context(|| format!("Job {job_id} disappeared from the scheduler"))?;
    Ok(job.state.is_final().then_some(job.state))
}

async fn cancel(client: &Client, job_id: u32) {
    eprintln!("grun: interrupted, cancelling job {job_id}");
    if let Err(error) = client.cancel_job(job_id).await {
        eprintln!("grun: failed to cancel job {job_id}: {error}");
    }
}

/// Block until the job is terminal, streaming its log; returns the exit status.
async fn wait_for_job(client: &Client, job_id: u32) -> Result<i32> {
    let mut sigint = signal(SignalKind::interrupt())?;
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut events = subscribe(client, job_id).await;
    let mut poll = tokio::time::interval(POLL_INTERVAL);
    let mut tail_tick = tokio::time::interval(TAIL_INTERVAL);
    let mut tail = LogTail::new();
    let mut payload_exit_code = None;

    let outcome = loop {
        tokio::select! {
            _ = sigint.recv() => {
                cancel(client, job_id).await;
                break JobState::Cancelled;
            }
            _ = sigterm.recv() => {
                cancel(client, job_id).await;
                break JobState::Cancelled;
            }
            _ = poll.tick() => {
                if let Some(state) = terminal_state(client, job_id).await? {
                    break state;
                }
            }
            event = events.as_mut().expect("guarded by precondition").next(), if events.is_some() => {
                match event {
                    Some(JobEvent::Changed) => {
                        if let Some(state) = terminal_state(client, job_id).await? {
                            break state;
                        }
                    }
                    Some(JobEvent::ExecutionFinished { exit_code }) => payload_exit_code = exit_code,
                    // Stream closed: the poll branch carries the wait from here.
                    None => events = None,
                }
            }
            _ = tail_tick.tick() => {
                tail.pump(client, job_id).await?;
            }
        }
    };

    // The state flips after the process exits, but the tmux executor's
    // pipe-pane can still be flushing; give the log a moment, then drain it.
    tokio::time::sleep(TAIL_INTERVAL).await;
    tail.pump(client, job_id).await?;

    Ok(exit_code_for(outcome, payload_exit_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_follow_terminal_state() {
        assert_eq!(exit_code_for(JobState::Finished, None), 0);
        assert_eq!(exit_code_for(JobState::Failed, None), EXIT_FAILED);
        assert_eq!(exit_code_for(JobState::Failed, Some(3)), 3);
        assert_eq!(exit_code_for(JobState::Failed, Some(0)), EXIT_FAILED);
        assert_eq!(exit_code_for(JobState::Timeout, None), EXIT_TIMEOUT);
        assert_eq!(exit_code_for(JobState::Cancelled, Some(0)), EXIT_CANCELLED);
    }

    #[test]
    fn sse_events_filter_by_job_id() {
        let raw = "event: job_state_changed\ndata: {\"job_id\": 7, \"new_state\": \"Running\"}";
        assert!(matches!(parse_sse_event(raw, 7), Some(JobEvent::Changed)));
        assert!(parse_sse_event(raw, 8).is_none());

        let raw = "event: job_execution_finished\ndata: {\"job_id\": 7, \"exit_code\": 2}";
        assert!(matches!(
            parse_sse_event(raw, 7),
            Some(JobEvent::ExecutionFinished { exit_code: Some(2) })
        ));

        assert!(parse_sse_event("event: connected\ndata: {}", 7).is_none());
        assert!(parse_sse_event(": keep-alive", 7).is_none());
    }

    #[test]
    fn env_export_drops_auto_detected_conda_env_but_keeps_explicit() {
        let auto_detected = || Job {
            conda_env: Some("proteus:gpu".into()),
            ..Job::default()
        };
        let parse = |argv: &[&str]| cli::GSrun::try_parse_from(argv).unwrap().srun_args;

        let mut job = auto_detected();
        apply_env_export(&mut job, &parse(&["grun", "cmd"]));
        assert!(job.env.is_some());
        assert!(job.conda_env.is_none());

        let mut job = auto_detected();
        apply_env_export(&mut job, &parse(&["grun", "--conda-env", "myenv", "cmd"]));
        assert!(job.env.is_some());
        assert_eq!(job.conda_env.as_deref(), Some("proteus:gpu"));

        let mut job = auto_detected();
        apply_env_export(&mut job, &parse(&["grun", "--no-export-env", "cmd"]));
        assert!(job.env.is_none());
        assert_eq!(job.conda_env.as_deref(), Some("proteus:gpu"));
    }

    #[test]
    fn exported_env_drops_daemon_owned_and_invalid_keys() {
        assert!(is_shell_identifier("PATH"));
        assert!(is_shell_identifier("_x1"));
        assert!(!is_shell_identifier("1ABC"));
        assert!(!is_shell_identifier("BASH_FUNC_foo%%"));
        for key in ENV_BLOCKLIST {
            assert!(!exported_env().contains_key(*key));
        }
    }
}
