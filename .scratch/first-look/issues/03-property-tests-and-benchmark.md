# 03: Property tests and benchmark for every analysis

**What to build:** A maintainer runs `uv run pytest`, and every registered analysis runs on generated DataFrames full of edge cases. A new analysis then meets the contract or fails loudly, without anyone writing new property tests for it. A separate opt-in benchmark times every analysis on a deterministic DataFrame of 1M rows and 100 mixed columns.

Both loop over `ae.list_analyses()` and use one call recipe per analysis, kept in the tests. The fixtures, the strategy and the recipes written here become the pattern for later slices.

Read these sections of `.scratch/first-look/spec.md`: Performance, and test groups 4 and 9, What makes a good test here and Warning policy under Testing Decisions. Covers user stories 44, 115, 118 and 119.

**Blocked by:** 02-summarize-numeric-and-column-rules

**Status:** ready-for-agent

- [ ] Hypothesis joins the development dependencies.
- [ ] A mixed-dtype DataFrame strategy generates floats with NaN and infinity, nullable integers, bool and boolean, object strings including empty and whitespace-only ones, pandas 3 strings, categoricals with unused categories, naive and time-zone-aware datetimes, and pyarrow-backed columns. It also generates zero rows, one row, zero columns, constant columns and all-missing columns. It works on pandas 2.2 and on 3.x.
- [ ] For every analysis in `ae.list_analyses()` and valid generated input, these properties hold:
  - no unexpected exception
  - the status is one of the analysis's declared statuses
  - `to_json()` is strict JSON
  - two identical calls give identical JSON
  - the input DataFrame keeps its values, dtypes, index, columns and `attrs`
  - no warning escapes
  - the printed text stays within 40 lines and 120 characters per line
- [ ] Each analysis has a call recipe in the tests that builds valid arguments for a generated DataFrame. A registered analysis without a recipe fails the suite with a message that says to add one.
- [ ] Hypothesis runs derandomized or with a fixed seed, so a run never depends on luck, and the default suite stays fast enough for the pre-push hook.
- [ ] Each bug the properties find in describe_dataset or summarize_numeric is fixed and gets a regression test through the public API.
- [ ] A deterministic benchmark DataFrame has 1,000,000 rows and 100 mixed columns: floats with missing and infinite values, nullable integers, strings and categoricals with cardinality from 2 up to all-distinct, booleans and datetimes, with about 5% of values missing overall.
- [ ] The benchmark times every registered analysis with default arguments, using the same recipes. The default test run skips it, and a documented command runs it.
- [ ] describe_dataset and summarize_numeric each finish in under 5 seconds on the benchmark. The timings and the machine go under `## Comments` in this ticket, since the target machine is a 2-vCPU, 8 GB AgentCore session.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
