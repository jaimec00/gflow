# gsrun Reference

`gsrun` submits one job and blocks until it finishes: it streams the job's log to stdout and exits with a status derived from the outcome. It is the `srun` to `gbatch`'s `sbatch`, for callers that need a single foreground process to wait on (shell pipelines, CI steps, agent harnesses) rather than a job id to poll.

## Usage

```bash
gsrun [options] [--] <command...>
gsrun [options] script.sh
```

## Common Examples

```bash
# Run on one GPU and wait; the job's output arrives on stdout as it is written
gsrun --gpus 1 -- python train.py --lr 0.001

# Chain on the exit status like any other command
gsrun --gpus 1 -- python train.py && gsrun --gpus 1 -- python eval.py

# Share a GPU under a VRAM budget
gsrun --gpus 1 --shared --gpu-memory 8G -- python eval.py

# Preview the job (command, run dir, exported variable count) without submitting
gsrun --gpus 1 --dry-run -- python train.py
```

## Options

`gsrun` accepts the `gbatch` submission options that apply to a single attended job: `--gpus`, `--shared`, `--gpu-memory`, `--time`, `--memory`, `--priority`, `--name`, `--project`, `--conda-env`, and the dependency flags (`--depends-on`, `--depends-on-all`, `--depends-on-any`, `--no-auto-cancel`). Array, parameter-sweep, retry, and scheduled-start flags are not available: `gsrun` follows exactly one job. See the [gbatch reference](./gbatch-reference) for each option's format.

| Option | Description |
| --- | --- |
| `--no-export-env` | Do not forward the current environment to the job |
| `--dry-run` | Print what would be submitted and exit |

## Environment

By default `gsrun` exports its own environment into the job, as `srun` does, so the job sees the same `PATH`, virtualenv or conda activation, and variables as the shell that submitted it. `gbatch` jobs, by contrast, run with the daemon's environment.

`CUDA_VISIBLE_DEVICES` and `GFLOW_ARRAY_TASK_ID` are never forwarded; the daemon sets them per job. Variables describing the submitting shell (`PWD`, `OLDPWD`, `SHLVL`, `TMUX`, `TMUX_PANE`, `_`) and names that are not valid shell identifiers are dropped too.

## Exit Status

| Job outcome | Exit status |
| --- | --- |
| `Finished` | `0` |
| `Failed` | the job process's exit code when the daemon reported it (process executor), otherwise `1` |
| `Timeout` | `124` |
| `Cancelled` | `130` |

`SIGINT` or `SIGTERM` to `gsrun` cancels the job (queued or running) before exiting with `130`, so killing the waiting process releases its GPU.

## How It Waits

`gsrun` subscribes to the daemon's `/events` stream and re-checks the job on every event about it, with a fallback poll every few seconds in case the stream drops. The submission message (`gsrun: submitted job <id> (<name>)`) goes to stderr; stdout carries only the job's log, so the output can be piped.

The job is a regular scheduler job in every other respect: it appears in `gqueue`, can be cancelled with `gcancel`, and its log remains available through `gjob log`.
