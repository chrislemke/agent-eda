# 09: MCP arguments read as their schema says

What to build: the MCP server reads arguments the way their JSON Schema describes them, so ISO 8601 strings work wherever the schema offers `date-time`, `date` or `duration`, and sentinel values for date and duration columns can be declared over MCP.

Read these sections of `.scratch/correctness/spec.md`: MCP arguments, and test group 7. This ticket covers user story 23. Findings: DES-3, SPEC-4, MCP-6, STD-10.

Blocked by: none

Status: ready-for-agent

- [ ] The MCP server validates arguments in JSON mode through the registry's arguments model.
- [ ] On a Parquet file with datetime64, timedelta64 and date columns, these are accepted over MCP and give the same `sentinel_count` as the Python call with `pd.Timestamp`, `pd.Timedelta` and `datetime.date`:
  - `{"born": ["1900-01-01"]}` and `{"born": ["1900-01-01T00:00:00"]}`;
  - `{"wait": ["PT0S"]}`;
  - the date column's ISO date.
- [ ] A time-zone mismatch, such as `"1900-01-01T00:00:00Z"` for a naive column, gets an error with a JSON fix.
- [ ] Python callers are unchanged: a `str` declared for a datetime column still gets the Python fix.
- [ ] A contract test sends every MCP tool an example of each `format` its input schema offers, and gets no `invalid_argument`.
- [ ] A value that `detect_missing_sentinels` suggests over MCP can be declared over MCP exactly as it was shown.
- [ ] The finish checks pass.

## Comments
