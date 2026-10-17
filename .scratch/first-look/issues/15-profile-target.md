# 15: profile_target

**What to build:** An agent names the target column and the task, and `ae.profile_target()` describes the target for that task. For classification that means class counts, shares and imbalance, plus missing labels, labels outside the expected set and, when the agent gives a threshold, rare classes. For regression it means the numeric summaries. The analysis flags targets that don't fit the task, such as a single class or a constant regression target, before the agent trains anything. It never infers the task from the dtype, so a numeric class label isn't mistaken for a regression target.

The analysis lives in `agent_eda.data.targets` and reuses summarize_numeric's implementation for regression.

Read these sections of `.scratch/first-look/spec.md`: the profile_target entry under The first-look analyses, Row IDs, and Project, data policy and masking. Covers user stories 106 to 108.

**Blocked by:** 08-project-and-masking

**Status:** ready-for-agent

- [ ] `target_column` and `task` are required. `task` is a literal of `binary_classification`, `multiclass_classification` and `regression`. `expected_labels` and `rare_share` are optional with no defaults, and `rare_share` lies between 0 and 1. `id_column`, `project` and `mask_sensitive` work as in find_duplicate_rows.
- [ ] For classification, the result gives the class counts and shares over valid labels, the number of classes, the majority and minority shares, the imbalance ratio, the missing labels, the labels outside `expected_labels` with counts and example row IDs, and the rare classes when `rare_share` is given.
- [ ] For regression, the result gives summarize_numeric's fields for the target, from the same implementation, plus the count of non-finite values.
- [ ] Without `expected_labels` or `rare_share`, the checks that need them don't run, and the result says they weren't checked.
- [ ] A single class, a binary task with other than two classes, labels outside `expected_labels` and a constant regression target each give a violation. A classification target whose values are all distinct gives a suspected finding. Missing labels and rare classes give descriptive findings.
- [ ] A regression task on a non-numeric column raises `ColumnTypeError` suggesting a classification task or a conversion. An unknown `target_column` raises `ColumnNotFoundError` with suggestions.
- [ ] With masking on and a sensitive target column, class labels and unexpected labels become placeholders, and so do row IDs from a sensitive `id_column`. Counts and shares stay.
- [ ] The declared statuses are `completed`, `insufficient_data` and `failed`.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold, including the no-leak property.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("class imbalance")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted, among them weights deferred and `rare_share` in place of `rare_threshold`.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
