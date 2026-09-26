# Bounded local research jobs

`scripts.research_job.guard.run_child(command, cwd, output_dir, timeout_seconds=300,
max_output_bytes=1<<30, reserve_bytes=10<<30, max_log_bytes=10<<20)` runs one
explicit argv list on macOS or Linux. It returns a dictionary and saves the same
result to `output_dir/status.json`; combined standard output and error go to
`output_dir/child.log`. The output directory must be absent, with an existing
parent. A repeated invocation with the same path raises `ValueError`, preserving
the previous run.

The guard starts a new process group, checks every 0.1 seconds, and kills the
group on timeout, log cap, output cap, or free-space reserve breach. It counts
regular files under both the output directory and a unique task-local
`.local-tools/tmp/research-*` directory. Symlinks and special files in these
trees are rejected. `TMPDIR`, `TMP`, and `TEMP` point to that task-local
directory, which is removed after the run. The status includes the return code,
duration, command, and one of `succeeded`, `nonzero_exit`, `timeout`,
`log_cap_exceeded`, `output_cap_exceeded`, `free_space_reserve_breached`,
or `guard_error`. A failed temporary cleanup is reported separately with its
path so the original child failure remains visible. It makes no peak memory
claim. Polling limits can be exceeded briefly between checks; this is not a
hard filesystem quota.

Callers must use trusted commands and pass output paths to those commands.
This guard is not an OS sandbox and cannot stop a command from deliberately
writing elsewhere. Keep source inputs separate and use a fresh run directory
for each attempt. Preflight errors occur before the run directory is created.

Example:

```python
from scripts.research_job.guard import run_child

result = run_child(
    ["/usr/bin/true"], cwd="/path/to/project", output_dir="/path/to/project/run-001"
)
if result["status"] != "succeeded":
    raise RuntimeError(result)
```
