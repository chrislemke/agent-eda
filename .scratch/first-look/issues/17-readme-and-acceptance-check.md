# 17: README and 0.1 acceptance check

**What to build:** Someone who lands on the repo sees what 0.1 does and how to use it, in examples that run as written. A final pass then checks the slice against the spec as a whole, now that all 11 analyses, discovery, the Project with masking, packaging and CI are in place.

Read these sections of `.scratch/first-look/spec.md`: Docs and records, plus the search acceptance cases and test group 9 under Testing Decisions. Covers user stories 115 and 123.

**Blocked by:** 05-ci-workflow, 09-project-files, 11-detect-missing-sentinels, 13-validate-column-types, 14-find-candidate-keys, 15-profile-target, 16-compute-association-matrix

**Status:** ready-for-agent

- [ ] The README's planned usage becomes real usage that runs as written, and its status line says what 0.1 contains.
- [ ] The README shows the first-look order and how to switch on masking, and calls masking a guardrail, not anonymization or access control.
- [ ] A test runs the README's usage examples.
- [ ] Every row of the spec's search acceptance table passes.
- [ ] `ae.guide()` names all 11 analyses in the first-look order and stays within about 60 lines.
- [ ] All 11 analyses finish in under 5 seconds on the benchmark with default arguments. The timings and the machine go under `## Comments` in this ticket.
- [ ] All 11 catalogue entries are marked implemented. Between them they note every difference the spec lists: no `context` argument, literal methods, defaults in compute_association_matrix instead of required type and method mappings, `max_key_size` defaulting to 2, weights deferred, "convertible" instead of "parseable", and shares instead of rates.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
