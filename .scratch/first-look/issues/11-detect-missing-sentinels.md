# 11: detect_missing_sentinels

**What to build:** An agent calls `ae.detect_missing_sentinels(df)` and gets the values that look like stand-ins for missing data, such as -999, "N/A" or 1900-01-01, each with its evidence and example row IDs. Suspected values stay apart from the sentinel values the agent declared, and the data never changes, so legitimate zeros, negatives and rare codes stay valid until the agent confirms otherwise.

The analysis lives in `agent_eda.data.missingness`.

Read these sections of `.scratch/first-look/spec.md`: the detect_missing_sentinels entry under The first-look analyses, Row IDs, and Project, data policy and masking. Covers user stories 93 and 94.

**Blocked by:** 08-project-and-masking, 10-profile-missing-values

**Status:** ready-for-agent

- [ ] Candidates come only from fixed, documented rules with no tunable thresholds: values from the built-in list, and repeated numeric values at the column's minimum or maximum whose distance to the nearest other value exceeds the range of all other values.
- [ ] The built-in list covers:
  - the numeric codes -9, -99, -999, -9999, 99, 999, 9999, 99999 and 999999, when they sit at the column's minimum or maximum
  - text, compared case-insensitively: "", whitespace only, "NA", "N/A", "NULL", "None", "nan", "-", "?", "unknown" and "missing"
  - the dates 1900-01-01, 1970-01-01, 2099-12-31 and 9999-12-31
- [ ] Each candidate reports its column, value, count, share, evidence codes, up to 10 example row IDs with the total, and whether it is declared or suspected.
- [ ] Declared values give descriptive findings, and matches from the built-in list or the gap rule give suspected ones. A declared value is never also reported as suspected.
- [ ] Declared `sentinel_values` follow the same rules as in profile_missing_values, from one shared implementation, including the error for a value that can never match its column's type.
- [ ] A value that matches no rule, such as a legitimate 0, a negative number or a rare code, is never a candidate.
- [ ] The record caps candidates at 100 with the total, and `to_frame()` lists all of them.
- [ ] `id_column`, `project` and `mask_sensitive` work as in find_duplicate_rows. With masking on, candidate values in sensitive columns and row IDs from a sensitive `id_column` become placeholders.
- [ ] The declared statuses are `completed`, `insufficient_data`, `not_applicable` and `failed`.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold, including the no-leak property.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("-999 placeholder codes")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
