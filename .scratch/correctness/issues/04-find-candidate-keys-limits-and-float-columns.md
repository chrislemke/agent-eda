# 04: find_candidate_keys: limits and float columns

What to build: `find_candidate_keys` finishes for any `max_key_size`, its budget message suggests a fix that works, and float measurements are no longer offered as primary keys unless named.

Read these sections of `.scratch/correctness/spec.md`: Keys, limits and threads, and test group 4. This ticket covers user stories 8 and 14. Findings: AN2-4, CORE-3, MCP-3, STD-38 (the key-search loop), AN2-10, AX-11, STD-15.

Blocked by: 01

Status: done

- [x] `find_candidate_keys(pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]}), max_key_size=sys.maxsize)` finishes in under one second.
- [x] Over MCP, a call with `"max_key_size": 100000000000000000000` returns, and a `ping` sent after it is answered.
- [x] On 40 columns with `max_key_size=40`, the `budget_exceeded` reason suggests the largest `max_key_size` that fits `max_combinations`. It never suggests an impossible `max_combinations`.
- [x] Non-integral float columns are left out by default, with a limitation that names them. Naming them in `columns` includes them. Float columns holding only whole numbers stay candidates.
- [x] The docstring and the README describe the float rule. The Comments record that this changes first-look ticket 14.
- [x] The docstring no longer claims its counts are what `df.duplicated` gives for an object column holding None and NaN (STD-15).
- [x] The finish checks pass.

## Comments

- **Key-size loop.** `combination_count` now sums `math.comb` only up to `min(max_key_size, columns)`, so `max_key_size=sys.maxsize` and `10**20` finish at once. The search loop already stopped at the column count. `max_key_size` has no schema maximum: any size is valid and acts as the column count, which the docstring says.
- **Budget message.** `budget_exceeded_reason` takes `lower=(parameter, value)` and `raise_up_to`. `find_candidate_keys` passes the largest `max_key_size` whose sets fit `max_combinations` (none when even the single columns don't fit) and `raise_up_to=1,000,000`: a `max_combinations` above that is never suggested, as it is beyond what the search can run. Both fixes appear when both are possible, such as "Lower max_key_size to 1, raise max_combinations to 10, or pass fewer columns." On 40 columns with `max_key_size=40` it says "Lower max_key_size to 2, or pass fewer columns." The 1,000,000 cap is my choice; the spec left "impossible" open. Supersets of keys aren't counted by the search, so the suggested size fits with room to spare.
- **Float rule.** With `columns=None`, a column of a float dtype (NumPy, nullable `Float32/64`, Arrow floats) is left out when some finite value has a fraction. Whole-number floats with missing values stay candidates, as do all-missing float columns. Infinities are neither whole nor fractional: a column with only whole numbers and infinities stays. The columns are named in a new result field `non_integral_float_columns` and a new limitation code `non_integral_float_columns`; they are not in `parameters["columns"]` or `support.columns` (unlike unhashable columns, which stay there), since they were never selected. When that leaves no column, the status is `not_applicable` with code `no_columns` and a message about the floats. Naming a float column in `columns` searches it and records no limitation.
- **This changes first-look ticket 14**, which accepts every dtype by default (its Columns bullet). Floats with fractions are now left out by default; ticket 14's text is untouched, as the roadmap owner handles it.
- **STD-15.** The docstring and the field descriptions of `distinct_count` and `duplicate_row_count` no longer cite `df.drop_duplicates` or `df.duplicated`. The docstring says None, NaN, NaT and NA are one value here, though pandas' `duplicated` tells None from NaN in an object column.
- **README.** A paragraph under the value-identity paragraph describes the float rule and the key-size rules (the generated Reference keeps only each argument's first sentence); the Reference lists the new field.
- **Tests.** The two existing tests that quoted the old budget message now quote the new one. The MCP test sends a `ping` after the call; `send_ping` is deprecated in the SDK, so it carries a `# ty: ignore[deprecated]` with its reason.
- **Checks.** All Finish checks ran locally and pass: ruff, ty, pytest (2,800 passed), both minimum-version runs, the stdio smoke test and the benchmark. Nothing is GitHub-only.
