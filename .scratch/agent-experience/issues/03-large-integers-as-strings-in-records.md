# 03: Large integers as strings in records

What to build: every JSON record writes integers beyond ±2⁵³ as strings, so that Node-based MCP clients stop rounding IDs. Result objects keep the exact Python ints.

Read these sections of `.scratch/agent-experience/spec.md`: Records (the integers item), and test group 2. This ticket covers user story 10. Findings: MCP-9.

Blocked by: 02

Status: done

- [x] With `id_column` "order_id" holding 1234567890123456789, `find_duplicate_rows`' record holds `"1234567890123456789"`, and parsing the MCP response with Node's `JSON.parse`, or an equivalent check of the bytes, gives the exact ID.
- [x] Row IDs, labels, values and parameters all follow the rule. Integers within ±9,007,199,254,740,991 stay numbers.
- [x] Result objects keep Python ints. The field descriptions and the output schema say that integers beyond ±2⁵³ are strings.
- [x] The README's "Reading a result" section and the CHANGELOG say so.
- [x] The finish checks pass.

## Comments

- One shared function, `record_integers` in `_written_form.py`, walks the dict that `to_dict()` builds, for results and discovery records alike, so `to_json()` and the MCP structured content follow. It rewrites every int outside -9,007,199,254,740,991..9,007,199,254,740,991 as its decimal string, wherever it sits; booleans stay booleans. Result objects and `__eq__` are untouched.
- Field descriptions of the row ID, label, value and `parameters` fields say it in one sentence ("The record writes an integer beyond ±2^53 as a string."). The output schema's root `description` says it once for the whole record. Fields typed `str | bool | int | float` or Any already accept strings, so records validate against the output schema; a test checks three analyses.
- The Node check in `tests/test_record_integers.py` runs only where `node` is installed (skipped otherwise); the bytes check through the in-process MCP client always runs when the mcp extra is installed. Benchmark ran (14 passed).
- README: one sentence in "Reading a result"; two sentences elsewhere trimmed to stay under the 60,000-character limit (59,972).
