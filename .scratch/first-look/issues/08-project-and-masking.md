# 08: Project and opt-in masking

**What to build:** An agent handling personal data declares sensitive columns in a Project and switches masking on, either for every call through the Project or for one call with `mask_sensitive=True`. Numbered placeholders then replace the raw values of those columns in everything the agent reads, while counts, shares and statistics stay.

This ticket adds `ae.Project` and `ae.DataPolicy` and brings masking to summarize_categorical and find_duplicate_rows. The later analyses that can show raw values build on it. Masking is a guardrail against accidental exposure, not anonymization and not access control, per ADR 0001.

Read these sections of `.scratch/first-look/spec.md`: Project, data policy and masking, and test groups 4 and 7 under Testing Decisions. Covers user stories 67 and 70 to 78.

**Blocked by:** 06-summarize-categorical, 07-find-duplicate-rows-and-row-ids

**Status:** ready-for-agent

- [ ] `ae.Project` is a frozen pydantic model that rejects unknown fields, with an optional `name` and a `data_policy`. `ae.DataPolicy` has `sensitive_columns`, default none, and `mask_sensitive`, default false. Setting `mask_sensitive` without sensitive columns raises `InvalidArgumentError`. Dataset roles stay out of this slice.
- [ ] Invalid Project or DataPolicy arguments raise `InvalidArgumentError` in the package's message format, never a raw pydantic error.
- [ ] summarize_categorical and find_duplicate_rows accept `project=` and `mask_sensitive=`. A per-call True or False wins. Otherwise the Project's setting applies, and without a Project nothing is masked. A per-call True without a Project that declares sensitive columns raises `InvalidArgumentError`.
- [ ] Masking stays off by default, even when sensitive columns are declared.
- [ ] With masking on, numbered placeholders per column, in result order, replace the category levels of sensitive columns in summarize_categorical and the row IDs taken from a sensitive `id_column` in find_duplicate_rows. The placeholders, `<masked 1>`, `<masked 2>` and so on, reveal neither content nor length. Counts, shares and numeric statistics stay.
- [ ] Index labels are never masked. The docs tell the agent to pass a non-sensitive `id_column` when the index holds sensitive values.
- [ ] Masking covers the printed text, `to_dict()`, `to_json()` and `to_frame()` alike. The result's `masking` field records whether masking was applied and to which columns.
- [ ] `ae.list_analyses()` and `ae.describe_analysis()` report masking support for both analyses, and `describe_analysis()` says that they can show raw values.
- [ ] The property suite gains a no-leak property for every analysis that supports masking: with masking on, no raw value from a sensitive column appears in the printed text, `to_json()` or `to_frame()`.
- [ ] The docstrings and the Project docs call masking a guardrail against accidental exposure, not anonymization and not access control.
- [ ] Tests cover a Project built in code, every combination of the masking resolution, and the error for masking without sensitive columns.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
