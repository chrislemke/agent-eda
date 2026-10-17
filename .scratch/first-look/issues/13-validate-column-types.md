# 13: validate_column_types

**What to build:** An agent declares the logical type it expects for each column and calls `ae.validate_column_types(df, schema=...)`. For each declared column it learns how many values already have that type, how many convert exactly, how many are invalid and how many are missing, with example row IDs and values for the invalid ones. The package never changes the data to find this out. A declared column missing from the data comes back as a violation, so checking a broken file still completes.

The analysis shares infer_column_semantics's parse rules, so the two never disagree on what parses. It lives in `agent_eda.data.validation`.

Read these sections of `.scratch/first-look/spec.md`: the validate_column_types entry under The first-look analyses, Row IDs, and Project, data policy and masking. Covers user stories 86 to 89.

**Blocked by:** 08-project-and-masking, 12-infer-column-semantics

**Status:** ready-for-agent

- [ ] `schema` is required and maps column names to a logical type or a `ColumnSpec`. `id_column`, `project` and `mask_sensitive` work as in find_duplicate_rows.
- [ ] Per declared column, the result gives the expected logical type, the observed dtype, whether the column is present, and four counts: compatible values, already stored as that type; convertible values, which convert exactly without changing meaning, such as "42" or 42.0 for integer, or text in ISO 8601 or the declared `datetime_format` for datetime; invalid values; and missing values.
- [ ] Invalid values come with up to 10 example row IDs and 5 distinct example values, each with its total.
- [ ] The result lists the undeclared columns.
- [ ] Invalid values, a declared column missing from the data and missing values in a column declared non-nullable each give a violation. A clean column gives a passed finding, and a column without values to check gives an unknown one.
- [ ] Nothing is coerced, and the data never changes.
- [ ] The declared statuses are `completed` and `failed`. The status is `completed` also when declared columns are missing from the data.
- [ ] validate_column_types and infer_column_semantics share one implementation of the parse rules, so they agree on which text parses as a number, an integer, an ISO 8601 date or a boolean.
- [ ] With masking on, example values from sensitive columns and row IDs from a sensitive `id_column` become placeholders.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold, including the no-leak property.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("check data types against a schema")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted, among them "convertible" instead of "parseable".
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
