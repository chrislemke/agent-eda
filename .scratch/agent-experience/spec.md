# Spec: what agents read

Status: ready-for-agent

## Problem Statement

Agent-friendliness is agent-eda's top priority, and agents meet it through what they read:
- the printed result (Code Interpreter returns only printed text);
- the JSON record (Claude Code shows the model only an MCP call's structured content);
- the names and statuses they branch on;
- the result API they script against;
- discovery and search;
- the MCP tool definitions their client loads.

The October 2026 audit (`.scratch/audit/findings.md`) played an agent through three messy datasets and measured every surface. The numbers were mostly right; what agents read often wasn't.

- **Printed text hides what matters.**
  - Lines are cut at 120 characters with "…", and the cut part often holds the fix, a pair's names or the "and 113 more" count, though results use at most 25 of their 40 lines (AX-1, AN2-11).
  - Tables list columns in DataFrame order, so a 120-column table shows 10 healthy columns and none of the four wrecked by -999 (AX-3).
  - A violation can be cut from the text behind suspected findings (AN1-8).
  - `summarize_categorical` flags case variants it never shows (AX-9).
  - Shares and percentages are mixed (AX-18).
  - "Checked, nothing found" is sometimes a finding and sometimes "Findings: none" (AX-19).
- **Records are heavy and partly unreadable.**
  - Capped lists drop the entries that findings point at, so field paths resolve on the object but not in `to_dict()` or over MCP (AX-2, AN2-8).
  - A first look on a 1,000 × 30 CSV costs about 47K tokens of structured content against about 4K of text. One wide `summarize_categorical` record was 383 KB, and `top_n` doesn't shrink it (MCP-10, AX-26).
  - Constant prose such as `definitions` repeats in every record.
  - Integers above 2⁵³ are rounded by Node-based MCP clients (MCP-9).
  - Status reasons, limitations and field descriptions carry Python such as `df.reset_index(drop=True)` and `columns=None` into MCP callers' context (STD-39, SPEC-10, MCP questionable decisions, AX-27).
- **Names and statuses disagree.**
  - One concept goes by different names: `schema` and `declared_schema`, `cardinality`, `distinct_count` and `class_count`, `count` and `non_missing_count`, `skipped_columns` and `left_out_columns` (AX-17, STD-28).
  - Every analysis declares `failed`, which none returns, and the guide lists statuses no analysis can return (AX-20, TEST questionable decisions).
  - Status reason codes are undocumented free text, with six spellings of "nothing to work on" (DES-11, STD-26, STD-27).
  - `describe_dataset` lives in `data.provenance`, a module planned for lineage and contract checks (DES questionable decisions).
- **The result API surprises.**
  - Half the results reject `to_frame(table=...)`.
  - `res.columns["Age"]` raises a bare KeyError, and `describe_dataset.columns` is a list where every other result has a mapping (AX-7, AX-8).
  - pydantic's deprecated `json()`, `dict()` and `schema()` return records without totals (AX-6).
  - `help(res)` is 29K characters of pydantic internals (AX-23).
  - Nothing carries one analysis's result into the next: an agent hand-writes `schema` and `sentinel_values` (AX-24).
  - `ae.data` and `ae.project` work only after something imported them (CORE-20).
- **Discovery doesn't scale and search misleads.**
  - `guide()` and the MCP instructions list every analysis and sit at their 60-line cap, so the 12th analysis fails two tests (DES-2, DOC-4).
  - Search ranks generic words ("how many rows" finds `find_duplicate_rows`), lacks common synonyms ("PII", "missingness", "skewed"), and its text grows without bound with the query (AX-10, CORE-15, CORE-16, SPEC-11, STD-34).
  - `describe_dataset`'s docstring names all 10 other analyses (DES-12, DOC-13).
- **MCP tool definitions are large.** Descriptions plus input schemas come to about 100K characters, about 25K tokens. The `table_path` description alone repeats 1,122 characters in 11 tools, and pydantic titles add more (MCP-18, DOC-13).

## Solution

- Printed text wraps instead of cutting. Findings come in order of assessment, and table rows in order of interest.
- Records keep what findings point at and drop repeated prose and per-column noise. Integers above 2⁵³ become strings, and no record text holds Python.
- One name per concept, with the renames made before 0.1.0 without shims. `failed` is removed from the analyses' declarations, and status reason codes become a documented list.
- The result API is uniform, helpful on mistakes, and carries results into the next analysis.
- `guide()` and the MCP instructions give an overview that doesn't grow with every analysis. Search understands agents' words and says when a match is weak.
- MCP tool definitions shrink by about a third.

## User Stories

### Printed text

1. As an agent in Code Interpreter, I want long lines wrapped, not cut, so that I always get the fix, the names and the counts.
2. As an agent, I want violations listed before suspected findings, and suspected findings before the rest, so that the printed text never hides the one violation.
3. As an agent with a wide table, I want the rows with findings or limitations shown first, so that the print shows the columns that need me.
4. As an agent told that a categorical column has case variants, I want to see the variants, so that I know what to merge.
5. As an agent comparing analyses, I want shares written the same way everywhere.
6. As an agent counting findings, I want "checked, nothing found" expressed one way in every analysis.

### Records

7. As an agent reading `to_dict()` or MCP structured content, I want every field path in a finding to resolve in the record.
8. As an agent in Claude Code, I want records that hold what I need without per-column boilerplate, so that a first look doesn't fill my context.
9. As an agent, I want `top_n` to bound what the record holds for levels, and `table_path` or `to_frame()` to give me the rest.
10. As an agent in a Node-based MCP client, I want large integer IDs to arrive exactly.
11. As an MCP caller, I want record text without Python.

### Names and statuses

12. As an agent, I want one name per concept across all results and arguments, so that code I wrote for one result works on the next.
13. As an agent, I want only statuses that can occur declared and listed, so that I write no dead branches.
14. As an agent, I want status reason codes and limitation codes from a documented list, so that I can branch on them.
15. As an agent, I want `describe_dataset` in the module named for what it does.

### Result API

16. As an agent, I want `to_frame(table=...)` accepted by every result, and a mistyped column key to suggest the closest names.
17. As an agent fluent in pydantic, I want the deprecated `json()`, `dict()` and `schema()` to point me to `to_json()`, `to_dict()` and `record_schema()`, not to return a different record.
18. As an agent, I want `help(res)` to show the fields and what they mean.
19. As an agent, I want `infer_column_semantics`' result to hand me a schema, and `detect_missing_sentinels`' result to hand me `sentinel_values`, so that the first look flows.
20. As an agent, I want `ae.data`, `ae.statistics` and `ae.discovery` to always resolve.

### Discovery and MCP definitions

21. As an agent, I want `guide()` to give an overview that stays one page however many analyses exist, and to point me to `list_analyses` and `search_analyses` for the rest.
22. As an agent searching in my own words, including "PII", "missingness", "skewed", "first look" and "zero variance", I want the right analysis first, and weak matches marked as weak.
23. As an agent, I want each analysis's "When to use" to name at most three alternatives, so that descriptions stay short and search isn't polluted.
24. As an MCP client user, I want tool definitions about a third smaller, so that the tool list costs less context.
25. As an agent, I want docstrings and field descriptions that agree with the code.

## Implementation Decisions

### Printed text

- **Wrapping.**
  - Finding, limitation, fact and reason lines wrap with a hanging indent, as `Reason:` and `More:` already do. No line is cut with "…".
  - The 40-line cap is the only hard limit, and it counts what it leaves out. Table cells keep their widths; a cell too wide is the only place text is shortened, and the record keeps the whole value.
  - This reverses first-look ticket 01's decision to cut.
- **Order.** Findings are ordered violation, suspected, descriptive, passed, then unknown. Within an assessment the analysis's own order stays.
- **Tables by interest.** Table rows put columns with a finding or limitation first, in DataFrame order, then the rest in DataFrame order. `profile_missing_values` and `validate_column_types` keep their own sorting.
- **`summarize_categorical`'s levels.** The top levels get their own lines within the 40-line budget, and a case-variant finding names its variants in its text.
- **Shares.** Printed text writes every share as a fraction under its field name (`missing_share 0.0598`). Percentages are no longer printed, so every number matches its field.
- **"Nothing found".** Findings report what was found. A check against a declaration that holds gives a `passed` finding, such as `valid_columns`. A search that finds nothing gives no finding, and a fact line says what was searched. `find_duplicate_rows`' `no_duplicate_rows` goes.
- **Wording.**
  - "Left out" lines say why each column was left out (AX-15).
  - `infer_column_semantics`' schema fact line accounts for every declared column (AN1-10).
  - An empty table's footer says it has no rows (CORE-21).

### Records

- **Capped lists keep what findings point at.** When a capped list (100 entries) must drop entries, it keeps those that a finding or limitation points at, then the rest in order. Every field path therefore resolves in `to_dict()`. Field paths name entries by key wherever the list is keyed, such as pairs named by their two columns.
- **`summarize_categorical`'s levels.** The record keeps `top_n` levels per column, with `levels_total`. `to_frame()` and `table_path` keep every level.
- **Support.** `support.columns` lists only columns where something was excluded. The shared counts sit once at the top, with the rule stated in the field description.
- **Constant prose leaves the record.** `definitions`, `share_denominator`, `level_order` and `missing_values_compare_equal` move into field descriptions, which `describe_analysis` and the output schema carry. The JSON keeps only values that vary with the data or the call.
- **Parameters.**
  - Nested lists in `parameters` are capped with totals.
  - `project` is recorded whole, without a `project_total`.
  - `to_frame()` gives every row ID wherever the record caps them: `detect_missing_sentinels`, `validate_column_types` and `profile_target`.
- `to_dict()`'s docstring matches what the record keeps (STD-18).
- ADR 0006's measured sizes are updated with before and after numbers, for the first look on 1,000 × 30 and for a wide `summarize_categorical`.
- **Integers beyond ±2⁵³ (9,007,199,254,740,991) become strings in every JSON record,** in row IDs, labels, values and parameters. Field descriptions say so, and result objects keep the exact Python ints.
- **No Python in record text.**
  - Status reasons, limitation reasons, finding claims and field descriptions name arguments and fixes without code. For example, "make the index unique" replaces `df.reset_index(drop=True)`, and "leaving the columns argument out" replaces `columns=None`.
  - The printed text and error messages keep surface-specific fixes.
  - Python-facing field descriptions stop naming `table_path` (AX-27).
  - The guard test from correctness ticket 10 extends to output schemas and to records produced by every recipe.
  - This reverses mcp-server ticket 07's choice to keep Python wording in records.

### Names, statuses and codes

- **Renames, made before 0.1.0 without shims** (round 2, Q6):

  | Before | After |
  | --- | --- |
  | `infer_column_semantics(declared_schema=)` | `schema=` |
  | `cardinality` | `distinct_count` |
  | `summarize_numeric`'s `count` | `non_missing_count` |
  | `skipped_columns` | `left_out_columns` (entries keep their reasons) |
  | `find_candidate_keys`' `duplicate_row_count` | `duplicate_group_row_count` |
  | finding code `declared_columns_missing` | `missing_declared_columns` |

  The ticket checks every other result field against the glossary and renames what still clashes, recording each rename. The CHANGELOG lists them.
- **`describe_dataset` moves to `agent_eda.data.structure`** (round 3, Q7a). Planned analyses and tests follow.
- **`failed`.** It is removed from the 11 analyses' declarations and `Statuses:` sections. The status stays in `Status` for analyses with anticipated numerical failures.
- **Which statuses are listed.** `guide()`, the README and the MCP instructions list only the statuses some registered analysis can return, computed from the registry.
- **Status reason codes** become a `StatusReasonCode` Literal, with one comment per code. The six spellings of "nothing to work on" merge into one family: `no_rows`, `no_columns` and `no_values` with a qualifier. Each analysis's `Statuses:` entries name their codes, `describe_analysis` shows them, and `_check_result` checks returned codes against declared ones.
- **Limitation codes** get discoverable meanings: `describe_analysis` and the generated README Reference show each code's comment.
- **`excluded_by_reason`** is typed, with one spelling per reason (STD-27).
- **Nulls and limitations.**
  - Checks that didn't run leave a limitation, as their siblings do (STD-29).
  - `infer_column_semantics`' null type gets a limitation (STD-31).
  - Fields that are null because of the task carry a limitation naming them (`profile_target`).
  - A share that wasn't computed is null, not 0.0 (`find_duplicate_rows`' `duplicate_share` under `not_applicable`).
- **No rows.** Data without rows gives `insufficient_data` in every analysis, `validate_column_types` included.

### Result API

- **`to_frame(table=...)`** works on every result: results with one table accept its name as a one-value Literal. Mappings keyed by column name raise a `KeyError` that suggests the closest names. `describe_dataset`'s positional `columns` gets a by-name lookup with the same suggestions, and the docs say why it is a list (repeated labels).
- **pydantic's deprecated methods** (`json`, `dict`, `schema`, `parse_obj`, `copy`) raise an error naming the method to use instead.
- **`help(res)`** shows a generated class docstring with each field's description, and points to `ae.describe_analysis`.
- **Rich display** shows the capped text template.
- **Carrying results forward:**
  - `InferColumnSemanticsResult.inferred_schema()` returns a schema `validate_column_types` accepts, leaving out columns without a type.
  - `DetectMissingSentinelsResult.sentinel_values(...)` returns `sentinel_values` for the candidates, optionally only some.
- **Public submodules.** `ae.data`, `ae.statistics`, `ae.discovery`, `ae.project` and `ae.mcp` resolve on a fresh import. A misspelled exception name such as `ae.AgentEDAError` points to `ae.errors`.

### Discovery and search

- **One overview builder** serves `guide()` and the MCP instructions. It lists each area (`data`, `statistics` and so on) with a count of its analyses and one line on what it covers, taken from the subpackage docstring. Then it gives the first-look order, and how to reach the rest: `list_analyses` and `search_analyses`, or their MCP tools.
- **The MCP instructions** drop Python module headings. A test checks that the overview doesn't grow with the number of analyses. This changes both specs' Discovery decision.
- **Docstrings.**
  - `When to use:` names the trigger and at most three alternatives, and `describe_dataset` stops listing every analysis.
  - See Also points to each analysis's natural next step (STD-30), and names only analyses, not `list_analyses` (STD-21).
  - AGENTS.md's neighbour-update rule changes to match.
- **Search.**
  - Generic words weigh little: column(s), values, rows, data, count. One-letter tokens are dropped.
  - New synonyms and tags:

    | Agent's words | Leads to |
    | --- | --- |
    | rows, row count, zero variance, constant | `describe_dataset` |
    | skewed, percentiles, outliers | `summarize_numeric` |
    | capitalization, whitespace | `summarize_categorical` |
    | missingness, nulls, null counts | `profile_missing_values` |
    | PII, privacy, anonymize | masking |
    | first look, eda | `guide()` |

  - Weak matches are marked as weak.
  - The query echo is cut to 40 characters, and the listed terms are capped with a count (CORE-15).
  - The acceptance table grows to cover every query in AX-10, CORE-16, SPEC-11 and STD-34.
- **The guide's advice on CSVs** says that `pd.read_csv`'s default NA handling turns "N/A" and "NULL" into missing values, and how to keep them (AX-22).

### MCP tool definitions

- The `data_path`, `table_path` and `project_path` descriptions shrink to one sentence each, and the rules move to the instructions.
- pydantic titles are dropped from input and output schemas.
- With the shorter `When to use` sections, descriptions plus input schemas shrink by at least 30%. The ticket records the sizes before and after.
- `readOnlyHint` stays false.

### Docstrings and descriptions that disagree with the code

- The cases STD-25 lists are fixed, along with:
  - STD-45: conversion advice for the same column kind;
  - STD-48: small record inconsistencies;
  - STD-32: `datetime_format` depends on the process locale, and the docs say so;
  - STD-33: the `schema` JSON Schema requires at least one entry.

## Testing Decisions

### The seam

The public API and the MCP protocol, as today. Text tests assert structure: wrapping, order, the 40-line cap and counts. They don't pin whole texts; one golden test per template is enough.

### What makes a good test here

Measure the costs the spec targets and record them:
- record sizes;
- the size of the tool definitions;
- the lines the overview takes as analyses are added, simulated with stand-in analyses through the registry's test hook.

### Test groups

1. **Text:** wrapping keeps every word; violations come first; interest order holds; variants are shown; the share format.
2. **Record:** field paths resolve in `to_dict()`; levels respect `top_n`; support is compact; no constant prose; big integers are strings; no Python (the guard).
3. **Names:** no old name remains, checked by a test against the rename table.
4. **Statuses:** declared equals reachable for every analysis; reason codes come from the Literal; listings are computed.
5. **Result API:** `to_frame(table=)`, the KeyError suggestions, deprecated methods, `help`, the bridges, submodules.
6. **Discovery:** overview size with 11 and with 60 stand-in analyses; the search acceptance table; the bounded echo.
7. **MCP definitions:** size thresholds and the absence of titles.

## Out of Scope

- MCP presets or subsets of tools, until the tool list becomes a problem (ADR 0006).
- Measuring text width in display cells (CORE-11): the 120-character limit counts characters, and the README says so. Agents read tokens, not cells.
- `readOnlyHint: true` on analysis tools, since `table_path` writes a file.
- Renaming `trim_fraction`: it is an argument, not a result field, and "fraction" is how trimmed means are named.
- Masking placeholders numbered across columns: they stay numbered per column.
- MCP records with MCP argument names: records keep `data` and `project`, so that both surfaces stay equal.
- Sending serialized JSON instead of the printed text in MCP responses: both stay (ADR 0006).

## Further Notes

- The plan was settled in a grilling session on 2026-10-19. The decisions behind this spec are round 2, Q2 to Q7 and Q10, and round 3, Q4 and Q7a.
- The renames happen before 0.1.0 and need no shims. AGENTS.md's two-release rule applies from 0.1.0.
- Each ticket adds its caller-visible changes to `CHANGELOG.md` (release ticket 04 creates it) and updates the README.
- Earlier decisions this spec changes:
  - first-look ticket 01 (cut lines, and field paths that index the object only);
  - the first-look and mcp-server specs' Discovery sections;
  - mcp-server ticket 07 (Python wording in records);
  - the first-look spec's "every analysis can also return failed".
