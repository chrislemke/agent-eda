# 03: Large integers as strings in records

What to build: every JSON record writes integers beyond ±2⁵³ as strings, so that Node-based MCP clients stop rounding IDs. Result objects keep the exact Python ints.

Read these sections of `.scratch/agent-experience/spec.md`: Records (the integers item), and test group 2. This ticket covers user story 10. Findings: MCP-9.

Blocked by: 02

Status: ready-for-agent

- [ ] With `id_column` "order_id" holding 1234567890123456789, `find_duplicate_rows`' record holds `"1234567890123456789"`, and parsing the MCP response with Node's `JSON.parse`, or an equivalent check of the bytes, gives the exact ID.
- [ ] Row IDs, labels, values and parameters all follow the rule. Integers within ±9,007,199,254,740,991 stay numbers.
- [ ] Result objects keep Python ints. The field descriptions and the output schema say that integers beyond ±2⁵³ are strings.
- [ ] The README's "Reading a result" section and the CHANGELOG say so.
- [ ] The finish checks pass.

## Comments
