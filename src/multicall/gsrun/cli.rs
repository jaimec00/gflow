use crate::multicall::gbatch::cli::AddArgs;
use clap::Parser;
use gflow::build_info::version;

#[derive(Debug, Parser)]
#[command(
    name = "gsrun",
    author,
    version = version(),
    about = "Submits a job to the gflow scheduler and blocks until it finishes. Inspired by srun."
)]
#[command(styles=gflow::utils::STYLES)]
pub struct GSrun {
    #[command(flatten)]
    pub srun_args: SrunArgs,

    #[arg(long, global = true, help = "Path to the config file", hide = true)]
    pub config: Option<std::path::PathBuf>,
}

/// The `gbatch` submission flags that make sense for a single, attended job.
/// Array / parameter-sweep / retry flags are deliberately absent: `gsrun`
/// follows exactly one job.
#[derive(Debug, Parser, Clone)]
pub struct SrunArgs {
    /// The script or command to run (e.g., "script.sh" or "python train.py --epochs 100")
    #[arg(
        trailing_var_arg = true,
        allow_hyphen_values = true,
        required = true,
        value_hint = clap::ValueHint::CommandWithArguments
    )]
    pub script_or_command: Vec<String>,

    /// The conda environment to use
    #[arg(short, long, value_hint = clap::ValueHint::Other)]
    pub conda_env: Option<String>,

    /// The GPU count to request
    #[arg(short, long, visible_alias = "gres", name = "NUMS")]
    pub gpus: Option<u32>,

    /// Allow this job to share allocated GPU(s) with other shared jobs
    #[arg(long)]
    pub shared: bool,

    /// The priority of the job
    #[arg(short = 'p', long, visible_alias = "nice")]
    pub priority: Option<u8>,

    /// Job dependency; accepts a job ID or shorthand like "@" / "@~N"
    #[arg(short = 'd', long, visible_alias = "dependency", value_hint = clap::ValueHint::Other)]
    pub depends_on: Option<String>,

    /// Multiple job dependencies with AND logic (all must finish successfully)
    #[arg(long, value_hint = clap::ValueHint::Other, conflicts_with = "depends_on")]
    pub depends_on_all: Option<String>,

    /// Multiple job dependencies with OR logic (any one must finish successfully)
    #[arg(long, value_hint = clap::ValueHint::Other, conflicts_with_all = ["depends_on", "depends_on_all"])]
    pub depends_on_any: Option<String>,

    /// Disable auto-cancellation when dependency fails (default: enabled)
    #[arg(long)]
    pub no_auto_cancel: bool,

    /// Time limit for the job (formats: "HH:MM:SS", "MM:SS", "MM", or seconds as number)
    #[arg(
        short = 't',
        long,
        visible_aliases = ["time-limit", "timelimit"],
        value_hint = clap::ValueHint::Other
    )]
    pub time: Option<String>,

    /// Memory limit for the job (formats: "100G", "1024M", or "512" for MB)
    #[arg(
        short = 'm',
        long,
        visible_aliases = ["max-mem", "max-memory"],
        value_hint = clap::ValueHint::Other
    )]
    pub memory: Option<String>,

    /// Per-GPU memory limit for shared scheduling (formats: "24G", "16384M", or "8192" for MB)
    #[arg(
        long = "gpu-memory",
        visible_aliases = ["max-gpu-mem", "max-gpu-memory"],
        value_hint = clap::ValueHint::Other
    )]
    pub gpu_memory: Option<String>,

    /// Custom run name for the job
    #[arg(
        short = 'n',
        short_alias = 'J',
        long,
        visible_alias = "job-name",
        value_hint = clap::ValueHint::Other
    )]
    pub name: Option<String>,

    /// Project code for tracking and organization
    #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
    pub project: Option<String>,

    /// Do not forward the current environment to the job (by default gsrun
    /// exports its own environment into the job shell, like srun)
    #[arg(long)]
    pub no_export_env: bool,

    /// Preview the job that would be submitted without submitting it
    #[arg(long)]
    pub dry_run: bool,
}

impl SrunArgs {
    /// Map onto `gbatch`'s argument set so the two commands build jobs with
    /// the same code path.
    pub fn to_add_args(&self) -> AddArgs {
        AddArgs {
            script_or_command: self.script_or_command.clone(),
            begin: None,
            conda_env: self.conda_env.clone(),
            gpus: self.gpus,
            shared: self.shared,
            priority: self.priority,
            depends_on: self.depends_on.clone(),
            depends_on_all: self.depends_on_all.clone(),
            depends_on_any: self.depends_on_any.clone(),
            no_auto_cancel: self.no_auto_cancel,
            array: None,
            time: self.time.clone(),
            memory: self.memory.clone(),
            gpu_memory: self.gpu_memory.clone(),
            name: self.name.clone(),
            auto_close: true,
            param: Vec::new(),
            dry_run: self.dry_run,
            max_concurrent: None,
            max_retries: None,
            param_file: None,
            name_template: None,
            project: self.project.clone(),
            notify_email: Vec::new(),
            notify_on: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flags_before_double_dash_command() {
        let args = GSrun::try_parse_from([
            "gsrun", "--gpus", "1", "--time", "2:00:00", "--", "python", "train.py", "--lr", "0.1",
        ])
        .expect("should parse");
        assert_eq!(args.srun_args.gpus, Some(1));
        assert_eq!(args.srun_args.time.as_deref(), Some("2:00:00"));
        assert_eq!(
            args.srun_args.script_or_command,
            vec!["python", "train.py", "--lr", "0.1"]
        );
    }

    #[test]
    fn requires_a_command() {
        assert!(GSrun::try_parse_from(["gsrun", "--gpus", "1"]).is_err());
    }

    #[test]
    fn maps_onto_gbatch_args() {
        let args =
            GSrun::try_parse_from(["gsrun", "-g", "2", "--shared", "--gpu-memory", "8G", "cmd"])
                .expect("should parse");
        let add = args.srun_args.to_add_args();
        assert_eq!(add.gpus, Some(2));
        assert!(add.shared);
        assert_eq!(add.gpu_memory.as_deref(), Some("8G"));
        assert!(add.param.is_empty());
        assert!(add.array.is_none());
    }
}
