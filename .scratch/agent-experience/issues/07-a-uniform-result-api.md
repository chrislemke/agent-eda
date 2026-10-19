# 07: A uniform result API

What to build:
- `to_frame(table=...)` on every result;
- helpful errors for mistyped column keys;
- pydantic's deprecated methods pointing to the right ones;
- a useful `help(res)` and a Rich display that keeps to the text template;
- methods that carry one analysis's result into the next;
- public submodules that always resolve.

Read these sections of `.scratch/agent-experience/spec.md`: Result API, and test group 5. This ticket covers user stories 16 to 20. Findings: AX-6, AX-7, AX-8, AX-23, AX-24, CORE-20, and SPEC's questionable decision on Rich display.

Blocked by: 05

Status: ready-for-agent

- [ ] Every result accepts `to_frame(table=...)`. A result with one table accepts its name, and any other name gets the error that `compute_association_matrix` gives today.
- [ ] `res.columns["Age"]` raises a `KeyError` that names the closest names, such as 'age'. `describe_dataset` results support lookup by name with the same message, and the docs say why `columns` is a list there.
- [ ] `res.json()`, `res.dict()`, `res.schema()`, `res.parse_obj()` and `res.copy()` raise an error naming `to_json()`, `to_dict()`, `record_schema()` or `model_copy()`.
- [ ] `help(type(res))` shows each field's description and points to `ae.describe_analysis`. `pydoc.render_doc` of a result is under 10,000 characters.
- [ ] Rich renders the capped text template.
- [ ] `InferColumnSemanticsResult.inferred_schema()` returns a mapping that `validate_column_types` accepts, leaving out columns without a type.
- [ ] `DetectMissingSentinelsResult.sentinel_values(columns=None)` returns `sentinel_values` that `profile_missing_values` accepts. Masked candidates are left out, with a note.
- [ ] Both docstrings show the flow in their Examples.
- [ ] `ae.data`, `ae.statistics`, `ae.discovery`, `ae.project` and `ae.mcp` resolve on a fresh import. `ae.mcp` gives the install command when the extra is missing. `ae.AgentEDAError` raises an AttributeError that points to `ae.errors.AgentEDAError`.
- [ ] The finish checks pass.

## Comments
