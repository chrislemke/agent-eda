# 04: Discovery: describe_analysis, search_analyses and guide

**What to build:** An agent new to the package calls `ae.guide()` and gets one page on what the package does, how to call an analysis, how to read its result and which analyses exist, in first-look order. `ae.search_analyses()` finds the right analysis from the agent's own wording, such as "nulls per column", and `ae.describe_analysis()` tells the agent everything it needs for a correct first call.

All three come from the registry and the docstrings, so there's no second copy to keep in sync, and none of them loads SciPy.

Read these sections of `.scratch/first-look/spec.md`: Discovery, Docstrings, and test groups 3, 5 and 8 plus the search acceptance cases under Testing Decisions. Covers user stories 9 to 14 and 16 to 18.

**Blocked by:** 02-summarize-numeric-and-column-rules

**Status:** ready-for-agent

- [ ] `ae.describe_analysis(name)` accepts the short or the fully qualified name and returns a typed description that prints compactly. It shows the signature; each parameter's type, default, allowed values and description from the docstring's `Args:`; the declared statuses with their reasons from `Statuses:`; the finding codes; the result fields and their meanings; masking support; the work-limit parameters; and the example. Its record includes a JSON Schema of the parameters other than `data`.
- [ ] For summarize_numeric, the description lists the allowed `quantile_method` values and the valid range of `trim_fraction`.
- [ ] An unknown or misspelled name raises `UnknownAnalysisError`, a `LookupError`, suggesting the closest analysis names.
- [ ] `ae.search_analyses(query, limit=5)` ranks analyses by keyword matches against names, summaries, tags and when-to-use text, expanded with a small curated synonym list. The list covers at least nulls and NaN for missing values, dupes for duplicates, correlation for association and label for target. It returns a typed listing with each match's summary and the terms that matched.
- [ ] The same query always returns the same ranked list, and ties go to the alphabetically first name. Search uses no embeddings and no network.
- [ ] A query without matches returns an empty listing that says so and points to `ae.guide()`.
- [ ] `ae.guide()` prints one page of plain text, at most about 60 lines, and returns None. It covers what the package is, the call pattern, how to read results and statuses, a reminder to print results, every registered analysis grouped by module with its one-line summary, and the spec's first-look order, limited to the analyses registered so far.
- [ ] Search and the guide read each analysis's summary line and when-to-use text from its docstring.
- [ ] In a fresh subprocess, calling each discovery function leaves SciPy, statsmodels and scikit-learn unimported.
- [ ] The contract test also checks, for every analysis, that `describe_analysis()` works, that searching for its name returns it first, and that every result field has a description.
- [ ] A search acceptance test holds the spec's table, starting with the rows for the analyses that exist: "dataset overview" returns describe_dataset first and "mean and standard deviation" returns summarize_numeric first. Each later analysis ticket adds its rows.
- [ ] A guide test checks that the guide names every registered analysis and stays within its size.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
