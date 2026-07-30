# Remote Verification Subagent Contract

Apply this contract when a task's acceptance depends on executing or observing
a remote runtime: machine readiness, code parity followed by a smoke test,
service launch, request verification, accuracy, benchmark, profiling, or remote
package/setup validation.

## Delegation boundary

- When running as the root/main agent and a subagent slot is available, delegate
  the bounded remote execution and evidence collection to one subagent before
  touching the remote runtime.
- Do not delegate local planning, source edits, Git decisions, or the final
  user-facing conclusion. The parent owns those decisions.
- When already running as a subagent, execute the remote workflow directly. Do
  not recursively spawn another verification agent.
- If collaboration tools or a slot are unavailable, execute directly and state
  that fallback in the handoff; do not block a verification solely on agent
  availability.

## Spawn contract

1. Use the configured `[agents]` defaults for model and reasoning effort unless
   the user explicitly requests an override. Do not hard-code a different model
   in a skill.
2. Use a bounded context fork and a self-contained task message. Include:
   - exact host/session/container and runtime paths
   - the command or skill entry point to run
   - current service/process state
   - whether remote-code-parity is required
   - allowed restart/cleanup scope and destructive-action boundaries
   - success criteria and required log/metric evidence
3. Use an internal subagent, not a user-owned task/thread.
4. Have the subagent read the applicable domain skill, `remote-toolbox`, and
   `remote-code-parity` when local code must be reflected remotely.

## Execution and retry contract

- The subagent owns remote launch, readiness waits, requests/workloads, cleanup,
  and evidence collection for the delegated run.
- The subagent must not make source changes merely to get a failing verification
  to pass. It reports the confirmed exception and holds the retry.
- The parent preserves any required fix locally, validates it, establishes
  remote parity, then reuses the same subagent with a follow-up task for retry.
- Keep long-running progress visible to the parent at meaningful checkpoints.

## Required handoff evidence

Return a compact result containing:

- resolved target and code/snapshot identity
- launch/readiness status and relevant log paths
- exact workload/request outcome, timing, and usage where applicable
- domain metrics and the log lines that prove success or failure
- any result that succeeded only partially, without upgrading it to full success
- final external state: service/job running or stopped, and cleanup performed
- remaining hardware, distributed, or backend assumptions not verified
