# 06: One value-reading engine

What to build: `infer_column_semantics` and `validate_column_types` read values through one engine, `read_values(series) -> ValueReading`, and `ColumnSpec` moves to the declarations module.

Read these sections of `.scratch/deepening/spec.md`: One value-reading engine, and test group 1. This ticket covers user stories 6 and 9. Findings: DES-17.

Blocked by: 04

Status: ready-for-agent

- [ ] `read_values` returns a `ValueReading` with:
  - the distinct values and their counts;
  - the value kinds and storage type;
  - a verdict per logical type.
- [ ] `infer_column_semantics` computes its shares from `ValueReading`, and `validate_column_types` its compatible, convertible and invalid counts. Neither walks a column on its own.
- [ ] `ColumnSpec` lives in the declarations module. `ae.ColumnSpec` and `agent_eda.data.validation.ColumnSpec` still work.
- [ ] The benchmark times for both analyses stay within 10% of the times before the change, on the latest and the minimum versions. The Comments record them.
- [ ] The baseline comparison passes unchanged.
- [ ] The finish checks pass.

## Comments
