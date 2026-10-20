# 12: Argument messages

What to build: every argument mistake gets a message that says what's wrong, what's allowed and a fix that works. That includes a path passed as data, an `id_column` that names the index, an unknown method, and wrong keywords to discovery functions. Table-driven tests on both surfaces cover every message.

Read these sections of `.scratch/correctness/spec.md`: Argument messages, and test group 9. This ticket covers user story 27. Findings: AX-4, CORE-7, AX-5, AX-21, AX-25, CORE-21, AN2-13, STD-35, STD-38 (id_column message), STD-46, TEST-5, and round 3, Q7d.

Blocked by: 11

Status: done

- [x] Wrong kinds of data get a fix that works:
  - `ae.describe_dataset("churn.csv")` and a `pathlib.Path` suggest `pd.read_csv(path)` or `pd.read_parquet(path)`;
  - a pyarrow RecordBatch or Table suggests `.to_pandas()`;
  - `None` says the variable holds no data.
- [x] `find_duplicate_rows(df.set_index("customer_id"), id_column="customer_id")` says the index labels are already the row IDs.
- [x] `find_candidate_keys(df, project=p)` and `summarize_numeric(df, mask_sensitive=True)` say that the analysis shows no raw values and so takes no project.
- [x] These argument mistakes get the messages the spec describes:
  - `columns=[None]` points to `columns=None`;
  - a set inside a mapping gets a fix for that key;
  - messages say "an object" and "an Interval";
  - for `expected_labels=[np.nan, np.nan]`, the NaN is reported first;
  - `task="binary"` suggests "binary_classification".
- [x] `sentinel_values={"email": [...]}` with `id_column="email"` and no `columns` no longer says to "Add it to columns". It says that leaving the columns argument out leaves out the id_column (STD-38).
- [x] No message shows a type annotation or a pydantic phrase. That covers `schema={"age": int}`, `schema=["age"]` and `quantiles="0.5"`.
- [x] Closest-name suggestions find case-only typos and name the misspelled key (STD-35).
- [x] An unknown `method` raises instead of computing Kendall, and the dead defaults are gone (STD-46).
- [x] `ae.describe_analysis(nme="x")`, `ae.search_analyses(question="x")`, `ae.Project.load(paht="x")` and `project.save(paht="x")` raise `InvalidArgumentError` (`invalid_call`) with the closest name. The Comments record that this changes first-look ticket 09.
- [x] Table-driven tests, one for Python and one for MCP, assert each message's code and key fragments, including every path TEST-5 lists.
- [x] The finish checks pass.

## Comments
- Tests: `tests/test_argument_messages.py` (Python) and `tests/test_mcp_argument_messages.py` (MCP) are tables of mistakes, each with the error code and key fragments, and they also check that no message shows a typing annotation or a pydantic phrase. The MCP table also checks that no message holds Python syntax. TEST-5's unreached paths are covered where a caller can reach them: data by position and keyword, "The data has no columns.", an unknown MCP tool, the `table_path` errors (directory, existing file, missing directory, other format, not a string, unwritable directory), the `data_path` directory, `ColumnSpec`'s logical type given twice, a duration finer than the column unit, a project file whose name can't be read, and the float, date and table values of a project file. The write-time `table_path` races (`FileExistsError` and `FileNotFoundError` between the check and the write) can't be reached through the public API without a race, so they stay untested.
- Settled where the spec left it open:
  - Argument types in messages are words now: "a number with 0 <= trim_fraction < 0.5", "a string, or None", "a list of numbers", "a mapping from column names to ...". Over MCP, None is written "null". One renderer, `_annotations.allowed_text`, replaces `_registry._allowed` and `_plain`; `describe_analysis` keeps `type_text`. Existing tests that quoted "float with ...", "int with ..." or "str | None" now quote the words. `tests/test_declarations.py` still checks that Project's signature and message agree, now in words ("a DataPolicy, a dict or None").
  - Pydantic's own sentence ("Input should be ...") is dropped from every argument message, since the allowed values say the same.
  - A single value given for a list option, such as `quantiles="0.5"`, is named for the option ("got '0.5' for quantiles"), not for the list's first item.
  - `project`, `project_path` and `mask_sensitive` on an analysis without them say "takes no project: it shows no raw values, so there is nothing to mask. Leave project out." The message is data-driven: it follows the registry, so `profile_missing_values` stops saying it when ticket 06 gives it masking.
  - `closest_names` moved to `_text.py`, where `results.py` and `__init__.py` can import it, and now also suggests names that start with the given text of 3 characters or more ("binary" gives "binary_classification"). Every suggestion goes through it.
  - A declaration with a misspelled field names the misspelled key even when a required field is missing as a result (STD-35). `ColumnSpec(logical_tpe="integer")` now says "ColumnSpec has no field 'logical_tpe'. Did you mean 'logical_type'?".
  - The sentinel message for an `id_column` names the id_column only when the call passed no `columns`; a call that passed `columns` without the column still says to add it to `columns`.
  - STD-46: `numeric_pair` takes the `NumericMethod` Literal, now defined in `_association.py`, and raises for any other value, so a mistyped method can't compute Kendall. The unreachable `rank_pair is None` and `buffer=None` paths are gone. `read_limitations` takes a `ReadEffects` (the three flags that `NumericColumn` extends), so `compute_association_matrix` no longer builds an empty stand-in column. Callers can't reach the unknown-method raise through the public API, which rejects it first with `invalid_argument`, so it isn't tested apart from that.
  - Not done, as the ticket doesn't list it: CORE-21's footer "res.to_frame() has all 0 columns" for empty results.
- Changes first-look ticket 09: `ae.describe_analysis(nme="x")`, `ae.search_analyses(question="x")`, `ae.guide(x=1)`, `ae.list_analyses(x=1)`, `ae.Project.load(paht="x")` and `project.save(paht="x")` now raise `InvalidArgumentError` with code `invalid_call` and the closest name, not a bare `TypeError`. The Python test of that is in `tests/test_argument_messages.py`; no earlier test asserted the `TypeError`.
- MCP: the unknown-tool test asserts only the code (-32602) and the fragment "There is no MCP tool named", since ticket 08 changes that message to add suggestions. The `data_path` directory and `table_path` rows assert fragments of today's messages, which ticket 08 reworks, so whichever of the two merges second may need to adjust a fragment.
- Checks: ruff, format, ty, `uv run pytest` (2420 passed, 2 skipped), both minimum-version runs (Python 3.12, numpy 1.26, pandas 2.2; with and without the mcp extra), the MCP stdio smoke and the benchmark all pass locally. No check was left for GitHub.
