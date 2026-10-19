# 12: Argument messages

What to build: every argument mistake gets a message that says what's wrong, what's allowed and a fix that works. That includes a path passed as data, an `id_column` that names the index, an unknown method, and wrong keywords to discovery functions. Table-driven tests on both surfaces cover every message.

Read these sections of `.scratch/correctness/spec.md`: Argument messages, and test group 9. This ticket covers user story 27. Findings: AX-4, CORE-7, AX-5, AX-21, AX-25, CORE-21, AN2-13, STD-35, STD-38 (id_column message), STD-46, TEST-5, and round 3, Q7d.

Blocked by: 11

Status: ready-for-agent

- [ ] Wrong kinds of data get a fix that works:
  - `ae.describe_dataset("churn.csv")` and a `pathlib.Path` suggest `pd.read_csv(path)` or `pd.read_parquet(path)`;
  - a pyarrow RecordBatch or Table suggests `.to_pandas()`;
  - `None` says the variable holds no data.
- [ ] `find_duplicate_rows(df.set_index("customer_id"), id_column="customer_id")` says the index labels are already the row IDs.
- [ ] `find_candidate_keys(df, project=p)` and `summarize_numeric(df, mask_sensitive=True)` say that the analysis shows no raw values and so takes no project.
- [ ] These argument mistakes get the messages the spec describes:
  - `columns=[None]` points to `columns=None`;
  - a set inside a mapping gets a fix for that key;
  - messages say "an object" and "an Interval";
  - for `expected_labels=[np.nan, np.nan]`, the NaN is reported first;
  - `task="binary"` suggests "binary_classification".
- [ ] `sentinel_values={"email": [...]}` with `id_column="email"` and no `columns` no longer says to "Add it to columns". It says that leaving the columns argument out leaves out the id_column (STD-38).
- [ ] No message shows a type annotation or a pydantic phrase. That covers `schema={"age": int}`, `schema=["age"]` and `quantiles="0.5"`.
- [ ] Closest-name suggestions find case-only typos and name the misspelled key (STD-35).
- [ ] An unknown `method` raises instead of computing Kendall, and the dead defaults are gone (STD-46).
- [ ] `ae.describe_analysis(nme="x")`, `ae.search_analyses(question="x")`, `ae.Project.load(paht="x")` and `project.save(paht="x")` raise `InvalidArgumentError` (`invalid_call`) with the closest name. The Comments record that this changes first-look ticket 09.
- [ ] Table-driven tests, one for Python and one for MCP, assert each message's code and key fragments, including every path TEST-5 lists.
- [ ] The finish checks pass.

## Comments
