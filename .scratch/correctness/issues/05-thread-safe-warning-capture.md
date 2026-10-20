# 05: Thread-safe warning capture

What to build: analyses can be called from several threads at once. Each result carries its own limitations, and the caller's warning settings stay untouched.

Read these sections of `.scratch/correctness/spec.md`: Keys, limits and threads, and test group 4. This ticket covers user story 15. Findings: CORE-1.

Blocked by: none

Status: done

- [x] Both capture steps run under one module-level lock: `_run.watch_all_warnings` and `_registry._absorb_warnings`.
- [x] After 400 calls of `summarize_numeric`, `describe_dataset`, `summarize_categorical` and `compute_association_matrix` through an 8-thread pool, `warnings.filters` and `warnings.showwarning` are what they were before, and a warning raised afterwards in the caller is shown.
- [x] Clean calls and calls whose data raises a RuntimeWarning, mixed in threads, give `numerical_warning` to every noisy result and to no clean one. Nothing reaches stderr.
- [x] The test passes on the minimum versions too.
- [x] The README and AGENTS.md's Implementation section say that concurrent calls are safe and capture their warnings one at a time.
- [x] The finish checks pass.

## Comments

- One reentrant lock, `_run._CAPTURE_LOCK`, guards both capture steps through a new helper `_run.capture_warnings()` (a `catch_warnings(record=True)` under the lock). `watch_all_warnings` and `_registry._absorb_warnings` both use it. The lock is an `RLock` because a capture can start inside another: `watch_warnings` in `_numeric` runs inside the registry wrapper's capture.
- Consequence: the registry wrapper holds the lock for the whole analysis call, so analyses called from several threads run one after another. The spec asks for this ("captures run one after another"); the benchmark is unchanged (summarize_numeric 1.98 s, all 12 benchmark tests pass).
- `tests/test_concurrent_calls.py` runs 400 calls of the four analyses in an 8-thread pool, a third of them noisy (a value whose hash warns RuntimeWarning, or float64 overflow for `summarize_numeric`). It checks `warnings.filters`, `warnings.showwarning`, `warnings._showwarnmsg_impl`, stderr, a later caller warning and each result's `numerical_warning`. Both tests fail without the lock (checked by swapping in a null context) and pass with it. The tests don't depend on versions, so the minimum-version runs pass too (1954 and 2261 passed).
- `tests/test_readme.py` checks that the README and AGENTS.md say concurrent calls are safe and capture one at a time. The README gets a Features bullet and a paragraph under "Reading a result".
- Not covered: the plain `warnings.catch_warnings()` in `_registry._spec` (import time, before callers exist) and `_data_files` (MCP server) stay unlocked; they run at registration or in the single-threaded server.
