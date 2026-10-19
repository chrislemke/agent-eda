# Spec: deepening

Status: ready-for-agent

## Problem Statement

agent-eda ships 11 analyses and plans about 270 more (`docs/planned-analyses.md`). The October 2026 audit's design and standards reviews (`.scratch/audit/findings.md`, DES and STD) found that the code makes each new analysis expensive. It also lets shared rules drift apart.

- **Every analysis lives in two places.** `results.py` is 3,329 lines: a shared base, the discovery records, and 51 per-analysis types that each serve one analysis. Each analysis ticket added 173 to 438 lines to it, and parallel tickets collided there. Adding an analysis touches at least 8 files and 12 sites (DES-5, STD-56).
- **Each analysis hand-builds the result envelope:** the analysis name, `method=None`, a `parameters` dict restating every argument, support, execution. Result text then reads its own arguments back out of that untyped dict, behind 24 isinstance guards. The invariants that support counts add up and field paths resolve hold by discipline only (DES-6, STD-53).
- **Python-versus-MCP wording is opt-in at each raise site.** There are 24 `worded(...)` lambdas, about 40 surface branches in 11 modules, and parallel fix tables in `profile_target`. The three leaks the audit found came from sites that forgot (DES-7, STD-57, DES-15).
- **The package root mixes three kinds of module.** It holds 26 private modules: shared helpers, four single-analysis engines, and five modules of the MCP adapter. `_registry` doubles as the argument-message library, `_run` is a grab-bag, there are two type renderers and four closest-name implementations, and private names cross module boundaries (DES-9, STD-35, STD-36, STD-49, STD-50).
- **The same pieces are built over and over.**
  - Shared finding codes such as `constant_columns` are built three times.
  - Result classes re-implement the same text conventions.
  - Each multi-table result names its tables in four places.
  - `profile_target` keeps four parallel structures per finding code (DES-16, DES-18, STD-51, STD-54, STD-55).
- **Two engines walk a column the same way.** `infer_column_semantics` and `validate_column_types` do, held together only by an agreement property. `data/validation.py` does three jobs (DES-17).
- **Knowledge is declared twice.**
  - Statuses and finding codes appear in the decorator and in the docstring (DES-14).
  - Each cap is written in up to 12 markers, 6 analysis constants and 39 sentences (DES-13).
- **The test suite repeats itself.**
  - Helpers are copied: 8 copies of `call`, 5 of `list_tools`, 3 of the reference read (TEST-14, STD-59).
  - A footer rewording fails 34 tests (TEST-13).
  - The packaging test starts 9 interpreters (TEST-17).
  - Some names are unclear (STD-58).

The correctness spec already put several shared rules in one place: value identity, constancy, declarations, parsing for suggestions and MCP argument reading. This spec does the restructuring that changes no behaviour.

## Solution

- `agent_eda.results` becomes a lazy facade, and each analysis's result types live next to it.
- The registry builds the result envelope and gives result text a typed view of its parameters.
- Error fixes become data, worded by one renderer per surface.
- Modules move to where their scope is visible.
- Shared findings and text conventions get one builder each, and `infer_column_semantics` and `validate_column_types` read values through one engine.
- Statuses, finding codes and caps have one source each.
- The tests share helpers and pin whole texts only in one golden test per template.

Public names, import paths, results and messages stay as they are. Every change here must leave the records of every recipe byte-identical.

## User Stories

1. As a coding agent adding an analysis, I want its computation, result types and text template in one place, so that I read and change one module.
2. As a coding agent, I want the result envelope built for me, so that a new parameter is recorded without a second edit, and support counts can't disagree.
3. As a coding agent writing a caller-mistake message, I want to state the fix once as data, so that it is right for Python and MCP callers alike.
4. As a coding agent, I want each module's location to tell me its scope: shared helper, single-analysis engine or MCP adapter.
5. As a coding agent, I want shared finding codes built by one function, so that `constant_columns` means the same in every analysis.
6. As a coding agent adding a `validate_*` analysis, I want one engine that reads a column's values, so that inference and validation can't disagree.
7. As a coding agent, I want statuses and finding codes declared once in the docstring, and each cap written once, so that changing one is a one-place edit.
8. As a coding agent editing tests, I want shared helpers and few whole-text assertions, so that a harmless rewording fails one test, not 34.
9. As a user, I want none of this to change a single byte of any result.

## Implementation Decisions

### Results next to their analyses

- `agent_eda.results` becomes a lazy facade, like `agent_eda/__init__.py`. Its export table maps each public result name to the module that defines it.
- The shared base (`ResultModel`, `Finding`, `Limitation`, `Support`, `Masking`, `AnalysisResult`, `MaskableResult` and the caps) moves to a private `_result_base.py`. The discovery records move next to discovery. Each analysis's result types move into a private sibling of the analysis module.
- `ae.results.<Name>` keeps working, and so does `from agent_eda.results import <Name>`. A pickle written by 0.1.0 still loads: a test keeps one such pickle per result class as a fixture.
- A sync test checks the facade's export table against the classes that subclass `AnalysisResult`.

### The registry builds the envelope

- The registration wrapper fills in `analysis`, `method`, `parameters` and `execution`. `parameters` comes from the arguments model's JSON dump, plus overrides for resolved or masked values, such as the resolved `columns`, masked `expected_labels` and masked `sentinel_values`.
- An analysis returns its own fields plus status, reason, support, findings and limitations.
- For a masking analysis, the wrapper refuses to record by default a parameter that holds data values, so a forgotten override can't leak.
- `ColumnSupport.of(column, eligible, excluded_by_reason)` derives the dependent counts.
- A typed parameters view, generated from the arguments model, replaces the `parameters.get(...)` calls in result text.
- `check_contract` asserts that support adds up.

### Error fixes as data

- `_wording` gets a small vocabulary of fixes: `Pass(name=value)`, `LeaveOut(name)`, `SeeArguments(owner)`, `PythonOnly(code)` and `DeclareSensitive(columns)`.
- An error carries what is wrong, what is allowed and its fixes. One renderer writes them for Python (`str(error)`) and for MCP, where `PythonOnly` is dropped and the rest is written as JSON.
- Raise sites stop branching on the surface, and `worded` and its lambdas go.
- `ColumnRule` takes these fixes and can be chosen per task. `profile_target` then selects its target with the task's rule, and its `_check_target` and parallel fix tables go.
- A property over the renderer replaces most of `MISTAKES`. Messages stay as the correctness spec left them, which a comparison of every message in the message tables checks.

### Module layout

- **The MCP adapter** (`_mcp_server`, `_mcp_instructions`, `_data_files`, `_local_files`, `_table_files`) moves into a private `agent_eda/_mcp/` package. `agent_eda.mcp` stays the public entry point.
- **Single-analysis engines move next to their analysis as private siblings:** `_association`, `_keys`, `_conversions` and `_sentinel_rules`.
- **`_registry` splits.** The registry stays. Argument normalizing and the message helpers (`invalid_message`, `unknown_name_message`, `missing_message`, value text, `python_scalar`, `check_distinct`) move to `_arguments`, with one type renderer (STD-36) and one closest-name helper (STD-35).
- **`_run` splits** into status helpers and a warnings module, which also takes `_absorb_warnings`.
- **`_sentinels` gets a name for what it holds:** values declared by the caller and checked against a column.
- **MCP schema fragments** become immutable, and the file-format knowledge sits in one place (STD-49, STD-50).
- No module imports another's underscore name, except from the modules meant as internal interfaces. A test checks this.

### Shared builders

- `_findings.py` has one builder per shared finding code: `constant_columns`, `all_missing_columns`, `mixed_value_types` and `infinite_values`. Each builder owns the claim, the assessment and the definition. The differences each analysis documents are passed as explicit arguments.
- The result base gains `shown(value, value_type, column)`, which quotes a value unless it is masked, and `row_id_source()`.
- A class-level `TABLES` declaration drives the names `to_frame` accepts, the table check and the footer for each surface.
- `profile_target`'s four parallel structures per finding code collapse into one.
- `summarize_numeric`'s defaults are no longer restated as literals.
- The small declarations every analysis module repeats move to a shared place (STD-51, STD-54, STD-55).

### One value-reading engine

- One engine reads a column's values: `read_values(series)` returns a `ValueReading` with the distinct values and their counts, the value kinds and storage type, and a verdict per logical type. `infer_column_semantics` computes its shares from it, and `validate_column_types` its compatible, convertible and invalid counts.
- `ColumnSpec` moves to the declarations module that correctness ticket 11 created.
- The agreement property stays, but now follows from the design.
- The fast paths tickets 12 and 13 tuned are kept, and the benchmark shows it.

### Single sources

- **Statuses and finding codes.** Registration reads them from the parsed docstring, as `_registry.spec_of` already does for unregistered functions. It raises for a `Statuses:` entry outside `Status`, or a malformed code. The decorator keeps tags, masking, work limits and label tolerance.
- **Caps.** The cap marker holds the only number. Analyses that truncate ask the result class for the limit, each capped field's description gets its cap clause from its marker, and `check_contract` asserts that a capped field holds `min(cap, total)` entries.

### Test suite upkeep

- One helper module serves the MCP tests: the client call, the tool listing, file writing, the Python reference run and an independent reference read.
- One golden test per text template remains. Other text tests assert structure, codes, fields and key fragments.
- `tests/test_readme.py` selects the MCP JSON block by its content (TEST-12).
- The packaging import checks run in one interpreter (TEST-17).
- Names that STD-58 found unclear are renamed.

## Testing Decisions

### The seam

The public API, unchanged. The proof that this spec changes no behaviour is a byte comparison:
1. Before the first ticket, a script records `to_json()`, `str(res)` and every `to_frame()` table for every recipe on `EDGE_FRAMES` and a fixed set of generated frames, plus every message in the message tables.
2. Every ticket compares against that baseline.
3. The baseline changes only where a ticket says why.

### What makes a good test here

Tests stay at the public seam. A refactor that needs a test to reach past the interface is the wrong shape. The new sync tests check the facade's export table against the result classes, and check that no module imports a private name from another.

### Test groups

1. **Byte equality** with the baseline, for every ticket.
2. **The results facade:** every name, every import path, the 0.1.0 pickles.
3. **The envelope:** support adds up; parameters are recorded without hand edits; a masking analysis can't record data values by default.
4. **The renderer:** a property over every fix type and surface.
5. **Layout:** no private imports across modules; `agent_eda.mcp` and the console script unchanged.
6. **Single sources:** docstring-declared statuses and codes, and caps from one marker.

## Out of Scope

- DataFrames passed by role (`AnalysisSpec.frames`, DES-8): this is built with the first analysis that takes `before=`/`after=` or `sample=`/`population=`. One adapter makes a hypothetical seam.
- Splitting public modules or renaming public names: public paths stay as 0.1.0 shipped them.
- Generated `llms.txt` and `SKILL.md`.
- Performance work beyond keeping the benchmark where it is.

## Further Notes

- The plan was settled in a grilling session on 2026-10-19. The decision behind this spec is round 2, Q1, and the ticket split is round 3, Q5.
- This spec runs after 0.1.0, so it must keep every public name and path stable. That is why the results move behind a facade, and why the byte baseline gates every ticket.
- Finding IDs refer to `.scratch/audit/findings.md`. The STD details are in `.scratch/audit/standards-review.md`.
