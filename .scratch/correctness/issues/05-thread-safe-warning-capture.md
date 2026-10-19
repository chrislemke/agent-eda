# 05: Thread-safe warning capture

What to build: analyses can be called from several threads at once. Each result carries its own limitations, and the caller's warning settings stay untouched.

Read these sections of `.scratch/correctness/spec.md`: Keys, limits and threads, and test group 4. This ticket covers user story 15. Findings: CORE-1.

Blocked by: none

Status: ready-for-agent

- [ ] Both capture steps run under one module-level lock: `_run.watch_all_warnings` and `_registry._absorb_warnings`.
- [ ] After 400 calls of `summarize_numeric`, `describe_dataset`, `summarize_categorical` and `compute_association_matrix` through an 8-thread pool, `warnings.filters` and `warnings.showwarning` are what they were before, and a warning raised afterwards in the caller is shown.
- [ ] Clean calls and calls whose data raises a RuntimeWarning, mixed in threads, give `numerical_warning` to every noisy result and to no clean one. Nothing reaches stderr.
- [ ] The test passes on the minimum versions too.
- [ ] The README and AGENTS.md's Implementation section say that concurrent calls are safe and capture their warnings one at a time.
- [ ] The finish checks pass.

## Comments
