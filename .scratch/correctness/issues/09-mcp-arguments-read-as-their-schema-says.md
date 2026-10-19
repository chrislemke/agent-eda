# 09: MCP arguments read as their schema says

What to build: the MCP server reads arguments the way their JSON Schema describes them, so ISO 8601 strings work wherever the schema offers `date-time`, `date` or `duration`, and sentinel values for date and duration columns can be declared over MCP.

Read these sections of `.scratch/correctness/spec.md`: MCP arguments, and test group 7. This ticket covers user story 23. Findings: DES-3, SPEC-4, MCP-6, STD-10.

Blocked by: none

Status: done

- [x] The MCP server validates arguments in JSON mode through the registry's arguments model.
- [x] On a Parquet file with datetime64, timedelta64 and date columns, these are accepted over MCP and give the same `sentinel_count` as the Python call with `pd.Timestamp`, `pd.Timedelta` and `datetime.date`:
  - `{"born": ["1900-01-01"]}` and `{"born": ["1900-01-01T00:00:00"]}`;
  - `{"wait": ["PT0S"]}`;
  - the date column's ISO date.
- [x] A time-zone mismatch, such as `"1900-01-01T00:00:00Z"` for a naive column, gets an error with a JSON fix.
- [x] Python callers are unchanged: a `str` declared for a datetime column still gets the Python fix.
- [x] A contract test sends every MCP tool an example of each `format` its input schema offers, and gets no `invalid_argument`.
- [x] A value that `detect_missing_sentinels` suggests over MCP can be declared over MCP exactly as it was shown.
- [x] The finish checks pass.

## Comments

- Validation: `_registry.validate_json` runs the options through the arguments model with `model_validate_json`, after the same `_normalize` a Python call gets, and returns the options the caller passed. The MCP server calls it after reading the data (so a data-file error still comes before an argument error) and passes the validated values on; the analysis validates them again in Python mode, which a datetime, date or timedelta object passes.
- Pydantic's smart union picks `str` for an ISO string in `bool | int | float | str | datetime | date | timedelta`, so JSON mode alone doesn't read it as a datetime. `_sentinels.read_iso(data, sentinel_values)` does, per column, and only over MCP. A string is read as ISO 8601 when its column holds datetimes, dates or durations and no text (a column of Python objects that mixes text and dates keeps its strings as text; there the MCP caller can't declare the date). The text is read with `_parse.read_as` (ticket 11's one parser), so MCP and the suggestions agree. Durations must also start with `P` (after an optional sign), because `read_as` reads them as pandas does and `"5"` or `"1 day"` would otherwise become durations. A time zone is kept as written, so `"...Z"` for a naive column fails the same check as a tz-aware `pd.Timestamp`.
- The sentinel error message is worded per surface (`worded`): over MCP it shows the value and the "did you mean" as JSON and the fix as `{"sentinel_values": {"born": ["1900-01-01T00:00:00"]}}`; the Python message is unchanged. The other Python leaks in MCP messages stay with ticket 10.
- The format contract test (`tests/test_mcp_arguments.py`) walks every tool's input schema for `format`, builds an argument holding an example of each, and fails on a format without an example, so a new `format` can't ship untested. Only the two sentinel tools offer formats today (`date-time`, `date`, `duration`).
- Not run: the benchmark (no analysis code changed). All other Finish checks ran locally and pass.
