# 14: find_candidate_keys

**What to build:** An agent calls `ae.find_candidate_keys(df)` and learns which columns, alone or combined, uniquely identify rows, with missing and duplicate counts for each candidate. When no exact key exists, the closest candidates expose a key broken by a few duplicates. The result says whether the search was exhaustive, so a cut-off search is never mistaken for proof that no key exists. The analysis lives in `agent_eda.data.structure`.

The 5-second target is the hard part. With 100 columns and the default `max_key_size` of 2, the search covers up to 5,050 column sets on 1M rows, far too many to check one by one on all rows. Rejecting candidates on an evenly spaced subset of rows, as the spec allows, has to do most of the work.

Read these sections of `.scratch/first-look/spec.md`: the find_candidate_keys entry under The first-look analyses, and Calling convention. Covers user stories 58, 103 to 105 and 116.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [x] `max_key_size` defaults to 2 and `max_combinations` to 10,000, and the result records both.
- [x] The search covers single columns, then combinations in increasing size, and keeps only minimal keys. It skips supersets of any key it has found.
- [x] Each candidate reports its columns, distinct key count, duplicate rows, rows with a missing key value, uniqueness share and whether it is minimal.
- [x] Keys with missing values are listed separately and are not primary keys.
- [x] Without an exact key, the result lists the 10 most unique column sets.
- [x] The result says whether the search was exhaustive. Hitting `max_combinations` stops the search with `budget_exceeded`, keeps what was found, marks it non-exhaustive, and states the estimate, the limit and how to raise the limit or narrow `columns`.
- [x] A deterministic, evenly spaced subset of rows may reject a candidate, because a duplicate in the subset proves it isn't unique. Uniqueness is only ever confirmed on all rows. A test uses a column set that is unique on the subset but not on all rows.
- [x] Distinct and duplicate counts match pandas on the same columns.
- [x] `ae.describe_analysis()` lists `max_combinations` as a work-limit parameter.
- [x] The result shows no raw data values.
- [x] The declared statuses are `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable` and `failed`.
- [x] The docstring has the seven sections in order and passes the contract tests.
- [x] The property suite has a call recipe for it, and the properties hold.
- [x] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [x] `ae.search_analyses("primary key")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [x] Its catalogue entry is marked implemented, with its differences from the catalogue noted, among them `max_key_size` defaulting to 2.
- [x] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments

What this ticket settled that the spec left open:

- **Missing values compare equal.** A set is a key when no two rows share its combination of values, and None, NaN, NaT and pd.NA count as one value, as pandas' `factorize` and `DataFrame.duplicated` over several columns compare them. SQL's rule, where a missing value never conflicts, would make an all-missing column a key, and a column that is 99% missing too, and only a threshold could keep them out. So a single-column key holds at most one missing value, and most keys with missing values are combinations such as (a, b) with gaps in b. One catch for tests: for a single column, `DataFrame.duplicated` calls `Series.duplicated`, which tells None and NaN apart in an object column. The pandas reference in the tests adds a constant column, so pandas compares through `factorize` there too.
- **The counts.** `distinct_count` is the rows `drop_duplicates` keeps. `duplicate_row_count` is the rows `duplicated(keep=False)` marks, the rows an agent should look at; rows minus `distinct_count` gives the excess copies. `missing_row_count` counts rows with a missing value in any column of the set, and `uniqueness_share` is `distinct_count` over the rows. Every reported count comes from all rows, also for the most unique sets.
- **"Without an exact key"** means without any key, with or without missing values. The ranking lists only minimal sets, most combinations first, then fewer columns, then column order. A set is minimal when each column adds combinations, so (order_id, day) never follows order_id with the same count. `is_minimal` is therefore true for every listed set. I kept it because the ticket asks for it and an agent shouldn't have to assume minimality.
- **The open question from `performance.md`: ranking the 10 most unique sets.** The counts are always exact, and the order is exact whenever bounds prove it. Sets are counted on all rows highest upper bound first, and counting stops once the 10th best count beats every bound left. The bounds are pigeonhole limits within each column's groups of rows, rows missing in every column of the set (they share one combination), and repeats on the evenly spaced subsets (a repeat there is a repeat on all rows). A count only hashes the rows whose value another row shares in every column, a few percent of the rows for near-unique columns. On ticket 03's benchmark frame without its three key columns, the bounds prove the ranking after 49 counts. The best pair is (f14, f17) with 997,143 combinations, the same pair an unbounded exact count found. Where bounds can't separate the sets, such as 40 uniform integer columns without missing values, counting stops after hashing 100 times the data's rows (at least 100,000,000) and a `ranking_not_proven` limitation says how many sets it counted. That cap keeps the ranking's cost linear in the rows, so it needs no `max_*` parameter. On that frame the listed pairs still match an exact count of every pair. A ranking by subset estimates, the other option the research named, would have needed a limitation every time.
- **Subsets.** Each set is checked on 16,384 evenly spaced rows, row i * n // 16,384, in three steps: every 32nd of those rows, every 4th, then all of them. In data of more than 65,536 rows, a set of several columns without a repeat there is also checked on every 4th row before all rows. That check sees a sixteenth of the pairs of rows that share a combination at a quarter of the cost, and it cut the 40-column case from 7.7 s to 3.8 s (278 checks on all rows down to 17). Single columns skip it, since their codes on all rows cost the same. `search.rejected_on_subset_count` counts rejections on either subset. `search.checked_on_all_rows_count` counts the sets checked on all rows, the only way to confirm a key.
- **The budget.** `max_combinations` counts the sets the search examines. A set of k columns is generated only when every subset of k - 1 columns was examined and isn't a key, so supersets of keys are never generated and never count. The estimate in the reason is `search.combination_count`, the sum of C(p, k) for k up to `max_key_size`. It is an upper bound, since skipped supersets lower the real need. Raising the limit to it always suffices. The reason code is `too_many_combinations`. `exhaustive` is false after a stop, and also when no search ran.
- **Shared parts for compute_association_matrix (ticket 16).** `_run.budget_exceeded_reason()` builds the reason as "<needs> <required> <unit>, more than <parameter>=<limit> allows; <outcome>. Raise <parameter> to <required>, or <narrow>." For keys it reads "The complete search covers up to 10 column sets, more than max_combinations=5 allows; it stopped after examining 5 and kept what it found. Raise max_combinations to 10, or pass fewer columns." A new contract test requires an analysis to declare `budget_exceeded` exactly when it declares work-limit parameters, each an int with a default and a valid range. `check_contract` now also requires a `budget_exceeded` result's reason to name a work-limit parameter. The text template wraps the `Reason:` line instead of cutting it at 120 characters, because a reason ends with the fix. Short reasons print as before.
- **Statuses.** `insufficient_data` for no rows (`no_rows`) and for one row (`one_row`), since every column identifies a single row. `not_applicable` when the data has no columns (`no_columns`) or every selected column holds unhashable values (`no_hashable_columns`). `failed` is declared, but no code path returns it.
- **Columns.** Every dtype is accepted, by name or by default. A column whose values can't be hashed (lists, dicts, and nested Arrow types such as list, struct and map) is left out with an `unhashable_values` limitation and named in `left_out_columns`, while `parameters["columns"]` keeps the selection. Only the values tell, so a column named in the call doesn't raise. `infer_dtype` reads object columns, and only the ones it calls "mixed" get hashed whole to find out.
- **Findings** are all descriptive: `keys_found`, `keys_with_missing_values` and `no_key_found`. The last names the most unique set and its duplicate rows. The claims stay short so the printed line fits 120 characters.
- **Search.** Tags "primary key", "unique key", "composite key", "unique id", "identifier" and "uniqueness". "duplicate rows" finds this analysis first until find_duplicate_rows ships, whose name outscores it.
- **Hypothesis draws constants from the package's code.** Hypothesis 6.17x picks literals from local modules for 5% of its choices, so derandomized examples shift whenever the code gains a constant. After this ticket's constants, `test_generated_frames_cover_every_dtype_family_and_shape` drew 200 frames without a float column holding NaN. It now searches with `find()` for each feature its 200 frames missed before failing. Every analysis ticket adds constants, so the parallel tickets may meet the same failure.
- **Glossary gap.** "Key", "minimal" and "primary key" aren't in GLOSSARY.md. The docstring defines them; `/domain-modeling` should decide whether they belong there.

Measurements, on the Apple M3 Max with 2 threads, while other agents kept the load average at 6 to 10:

| Frame (1,000,000 rows) | latest (pandas 3.0.6, numpy 2.5.4) | minimum (pandas 2.2.0, numpy 1.26.0) |
| --- | --- | --- |
| Benchmark, default arguments (`uv run pytest -m benchmark`) | 0.35 s, completed | 0.50 s, completed |
| Benchmark without its key columns t17, t18 and t19 | 1.61 s, ranking proven | 2.12 s, ranking proven |
| Benchmark with `max_key_size=3` | 1.40 s, budget_exceeded at 10,000 sets | 1.73 s, budget_exceeded |
| 40 uniform integer columns, no missing values | 3.82 s, ranking not proven | 3.92 s, ranking not proven |

The research prototype took 1.84 s and 2.72 s on its own benchmark frame, because it factorized all 100 columns. The search factorizes only the 16,384-row subset of each column, plus the whole columns that pass it (t17, t18 and t19 here). The benchmark run's peak memory was 1.80 GiB and 2.22 GiB, against ticket 03's 1.77 GiB and 2.23 GiB.

Checks:

- `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass. The minimum-versions run passed 557 tests with numpy 1.26.0 and pandas 2.2.0, and `uv run --isolated --python 3.13 pytest` and the same on 3.14 passed.
- The `thorough` profile (1,000 examples) passed this analysis's properties and the pandas comparison on both version sets.
- A scratch script compared keys and rankings with brute force (pandas counts of every set) on 340 random frames, 280 of 300 to 40,000 rows and 60 of 70,000 to 100,000, where the every-4th-row check runs. Keys always agreed, and every proven ranking matched the brute-force top 10.
- Checking the 40-column frame against an exact count of every pair caught a bug: when the work limit stopped the ranking, sets it had already counted could miss the list. They now enter the list as soon as they are counted, and the work-limit test checks that all 10 listed sets are pairs.
- Not run locally: Linux x86_64 and ARM64, and the 2-vCPU target machine. Ticket 05's CI covers the first two.
