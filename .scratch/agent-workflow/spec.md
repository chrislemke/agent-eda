# Spec: agent workflow and documents

Status: done

## Problem Statement

Coding agents build agent-eda ticket by ticket, and agents call it. The October 2026 audit (`.scratch/audit/findings.md`) found that the documents those agents work from steer them wrong.

- **Picking up work.** AGENTS.md, which every agent loads on every turn, named the finished `mcp-server` feature as the current one, and its rule for the next ticket matched mcp-server ticket 01 again. Nothing said what comes after the last ticket. Finished tickets still read `ready-for-agent`, so any skill that reads the tracker sees 25 open tickets (DOC-1, DOC-9, SPEC-18).
- **Running the checks.**
  - With uv 0.10, the version on the maintainer's PATH, `uv sync` removes the mcp extra. `ty check` then fails on 29 imports, and the MCP tests skip with a reason that points to `pip install` (DOC-2).
  - "The four checks under Commands and the minimum-versions run" is ambiguous. The two minimum-versions commands, and the warning that they need `--isolated`, live only in the README (DOC-8).
  - The pre-push hook skips pushes that change only the README, CI or pyproject, which tests check (TEST-16). Coverage ignores code that runs in subprocesses (TEST-8).
  - An LSP plugin for a type checker the project doesn't use is enabled (DOC-14).
- **Adding an analysis.**
  - It touches about 12 places, and no document lists them. Forgetting the `@analysis` decorator passes every test. The three export lists in `__init__.py` and the documented error codes can drift silently (DOC-5, DES-19).
  - AGENTS.md has no rules for what MCP callers read (DOC-10).
  - Following AGENTS.md's "move it to the Analyses table" fails `tests/test_readme.py` for any analysis outside the first look (DOC-3).
- **The README.**
  - It is about 36K tokens, so an agent's Read shows only the first 631 of 1,066 lines.
  - 46% of it is the backlog of planned analyses, and 31% a hand-written Reference that repeats docstrings AGENTS.md calls "the only copy".
  - The development notes AGENTS.md relies on come after line 1,000 (DOC-6, DOC-7).
  - Several signatures and claims are wrong (SPEC-8, SPEC-20, CORE-22, AN1-12, SPEC-16). Its relative links break on PyPI (REL-4).
  - Backlog rows contradict the ADRs (DOC-18, SPEC-17).
- **Vocabulary.** The glossary lacks terms that decide behaviour, such as limitation, first look, level and key, and some code and tests use words it avoids (DOC-16, SPEC-19, STD-23, STD-24).

## Solution

- `.scratch/ROADMAP.md` lists the specs in order. AGENTS.md says how to find the current spec and ticket without naming one, and a finished ticket says `Status: done`.
- `pyproject.toml` requires uv 0.13 or later. AGENTS.md holds the only copy of the commands and an exact list of finish checks.
- A checklist for adding an analysis, plus tests for the steps that fail silently today. AGENTS.md states the rules for what MCP callers read.
- The README becomes the user guide, readable in one Read and correct on PyPI:
  - the backlog moves to `docs/planned-analyses.md`;
  - development notes move to `CONTRIBUTING.md`;
  - the Reference is generated from the docstrings;
  - every link is absolute.
- The glossary defines every term the analyses rely on.

## User Stories

### Picking up work

1. As a coding agent starting a session, I want AGENTS.md to tell me how to find the current spec and ticket without naming one, so that it never sends me to finished work.
2. As a coding agent, I want a finished ticket to say `Status: done`, so that every skill that reads the tracker can tell open work from finished work.
3. As a maintainer, I want the order of the specs in one roadmap file, so that I can reorder work by editing one line.
4. As a coding agent that finishes the last open ticket, I want to be told to ask the maintainer what comes next, so that I don't invent work.
5. As a coding agent, I want tickets marked `ready-for-human` skipped, so that I never publish a release or change account settings on my own.

### Running the checks

6. As a coding agent on any machine, I want uv to stop with one clear message when it is older than the project needs, so that the mcp extra is never removed without a word.
7. As a coding agent, I want AGENTS.md to list every finish check as a command I can copy, both minimum-versions runs and the MCP smoke test included, so that I finish a ticket with the checks CI runs.
8. As a coding agent, I want the warning that the minimum-versions runs need `--isolated` next to their commands, so that I never rewrite `uv.lock` by accident.
9. As a coding agent, I want skipped MCP tests to name the fix that works in this repository, so that I don't `pip install` the package into the project.
10. As a maintainer, I want the pre-push hook to run the tests on every push, so that a README or workflow edit can't push a failing document test.
11. As a maintainer, I want coverage to count code that runs in subprocesses, so that the report shows only real gaps.
12. As a coding agent, I want only the plugins the project uses enabled, so that I'm not offered a type checker that errors and isn't the project's.

### Adding an analysis

13. As a coding agent adding or renaming an analysis, I want one checklist of every place to touch and what catches each miss, so that I don't learn the steps one failing test at a time.
14. As a coding agent, I want a failing test when I export an analysis but forget to register it, so that it can't be left out of discovery, the contract tests and the MCP server.
15. As a coding agent, I want a failing test when the export lists in `__init__.py` disagree, or when an error code is raised but not documented, so that these can't drift.
16. As a coding agent writing messages and docstrings, I want AGENTS.md to state what MCP callers read and how it must be worded, so that my analysis is right on both surfaces.
17. As a coding agent, I want AGENTS.md to point at the test helpers that generate edge-case data and add an analysis to the property suites, so that I reuse them.
18. As a coding agent that checks a change through the repository's MCP server, I want to know that it serves the code from when the session started, so that I don't misread my own change.

### README

19. As a person or agent new to agent-eda, I want the README to be the user guide (install, quick start, analyses, results, projects, MCP and reference), short enough to read in one go.
20. As a visitor on PyPI, I want every link in the description to work.
21. As a coding agent, I want the Reference generated from the docstrings, with a failing test when it is stale, so that the docstring stays the only hand-written copy.
22. As a coding agent shipping an analysis outside the first look, I want a table for it in the README and a test that every analysis is listed exactly once.
23. As a maintainer, I want the backlog in its own file, with every row consistent with the ADRs and the glossary.
24. As a contributor, I want the development notes in `CONTRIBUTING.md`, pointing to AGENTS.md for the commands.

### Vocabulary

25. As a coding agent naming something, I want the glossary to define limitation, first look, level, native missing value, compatible, convertible and invalid values, key, primary key and candidate key, target, class and label, and sentinel candidate.
26. As a reader of code, tests and docs, I want the words the glossary avoids gone from them.

## Implementation Decisions

### Roadmap and tickets

- `.scratch/ROADMAP.md`, created with this spec, lists the specs in the order they are worked on. AGENTS.md names no spec.
- **Choosing the next ticket.** The current spec is the first in the roadmap with a `ready-for-agent` ticket that isn't `done` and whose `Blocked by:` tickets are all done. Within it, an agent takes the lowest-numbered such ticket. A blocked ticket doesn't hold agents in its spec: release's "Prepare 0.1.0" waits for two later specs. Tickets marked `ready-for-human` or `needs-info` are skipped. When no spec has such a ticket, the agent asks the maintainer which spec comes next.
- **Ticket header lines.** Tickets carry a plain `Status:` line and a plain `Blocked by:` line: `Blocked by: none`, `Blocked by: 01, 03`, or another spec named in words, such as `Blocked by: every ticket of correctness`.
- **Closing a ticket.** Finishing a ticket sets `Status: done`. `docs/agents/triage-labels.md` adds `done`. `docs/agents/issue-tracker.md` gets a "Closing a ticket" section that matches AGENTS.md. Its Wayfinding section says it applies only to `/wayfinder`.
- **The finished tickets.** The 25 tickets and 2 specs of first-look and mcp-server get `Status: done`. Boxes that later commits superseded aren't unticked. Each gets one line under Comments naming the decision or commit that superseded it.

### Commands and checks

- `[tool.uv] required-version = ">=0.13"`. CI and the uv-lock hook already run 0.13.1.
- AGENTS.md's Commands section is the only copy of the commands. It gets a "Finish checks" list:
  - `ruff check`, `ruff format --check` and `ty check`;
  - `pytest`;
  - both minimum-versions runs, with the note that `--isolated` keeps `uv.lock` unchanged;
  - the MCP smoke test;
  - the benchmark, after changing an analysis whose cost grows with rows or columns.

  `tests/test_ci_workflow.py` reads the minimum-versions commands from AGENTS.md. The README and `CONTRIBUTING.md` point to AGENTS.md for the commands.
- The MCP test modules' skip reason names `uv sync` (uv 0.13 or later) or `uv run --extra mcp pytest` inside this repository, and `pip install 'agent-eda[mcp]'` outside it.
- The pre-push pytest hook runs on every push and fails when it collects no tests.
- `[tool.coverage.run] patch = ["subprocess"]`.
- `.claude/settings.json` drops pyright-lsp. Any document that asks for a review names the skill with its plugin prefix, such as `mattpocock-skills:code-review`.
- AGENTS.md states the performance target as the benchmark checks it: every analysis runs in under 5 seconds on 1,000,000 rows × 100 columns.

### Adding an analysis

- The checklist lives in `docs/agents/new-analysis.md`. AGENTS.md points to it in one line: "Before adding or renaming an analysis, follow docs/agents/new-analysis.md". Only that branch of work needs it.
- For each step, the checklist names what catches a miss.
- **Guard tests.** Each of these fails:
  - a public function defined in a module under `agent_eda.data` or `agent_eda.statistics` that isn't a registered analysis;
  - export lists in `__init__.py` (`__all__`, `_EXPORTS` and the `TYPE_CHECKING` imports) that name different sets;
  - an `AgentEDAError` subclass whose `code` is missing from the `ae.errors` docstring or the README's errors table;
  - a documented code that no class has.
- **An "MCP surface" rule in AGENTS.md:**
  - the docstring's sections become the MCP tool description;
  - a message an MCP tool can reach gives a fix an MCP caller can make;
  - every parameter type needs a JSON Schema, and a DataFrame passed by role needs a `<role>_path`;
  - nothing an MCP caller reads holds Python.

  The ADR trigger list adds ADR 0006, for the MCP server, data reading and anything an MCP caller reads.
- AGENTS.md's Tests section points to `tests/recipes.py` (a `Recipe` adds an analysis to the property, masking, benchmark and MCP contract tests) and `tests/strategies.py` (`data_frames()`, `EDGE_FRAMES`). It also says the repository's MCP server serves the code from when the session started.

### README

- The README's sections, in order:
  1. Features
  2. Installation
  3. Quick start
  4. Analyses: the first-look table, then an "Other analyses" table by module
  5. Discovery
  6. Reading a result
  7. Projects and masking
  8. Use over MCP
  9. Reference
  10. Documentation: links to CONTRIBUTING.md, AGENTS.md, GLOSSARY.md, the ADRs and the planned analyses
  11. License
- **The backlog moves to `docs/planned-analyses.md`.** The `tests/test_readme.py` checks on it move with it (each name once, no shipped analysis listed). Rows that conflict with the rules are rewritten:
  - `load_dataset` reads no remote source;
  - `run_analysis` and `run_standard_eda` take no run context (ADR 0004);
  - the artifact references go;
  - "config" and "split" are reworded (glossary);
  - `test_functional_dependencies` runs no hypothesis test, so it gets a name without `test_`.
- **Development moves to `CONTRIBUTING.md`.** It keeps its notes, and its commands come by pointer to AGENTS.md.
- **The Analyses section has two tables,** the first-look one in `ae.guide()`'s order and "Other analyses" grouped by module. A test checks that every registered analysis appears in exactly one of them. AGENTS.md's "move it to the Analyses table" says which table.
- **Every link to a repository file is absolute** (`https://github.com/chrislemke/agent-eda/blob/main/...`), and a test forbids relative links. Anchors within the README stay relative.
- The Installation section keeps its pre-release wording. The release spec rewrites it for PyPI.

### Generated Reference

- `scripts/generate_reference.py`, a development tool that isn't shipped, writes the Reference between `<!-- reference:start -->` and `<!-- reference:end -->`. It covers:
  - every analysis: its signature, with the keyword-only star and defaults; its summary; each argument's type, default and meaning; the fields to read; each finding code with its assessment; its statuses;
  - the discovery functions;
  - Project, DataPolicy, ColumnSpec, LogicalType and AnalysisResult;
  - the error codes;
  - the MCP server command.
- All of this comes from the registry, the signatures, the docstrings and `ae.errors`.
- A test regenerates the section and compares it with the README, failing with the command to run. Hand-written checks in `tests/test_readme.py` that the generation makes redundant go.
- AGENTS.md says the docstring is the only hand-written copy, and that the README's Reference is generated from it. The sentence about generated agent docs in a later slice goes.

### Glossary and wording

- New entries, each with an Avoid line in the existing format:
  - Limitation
  - First look
  - Level
  - Native missing value
  - Compatible, convertible and invalid value
  - Key, primary key and candidate key
  - Target, class and label
  - Sentinel candidate
- The words STD-23 lists go from field descriptions, names and tests. Unqualified "server" goes from 8 test names and private docstrings, and the MCP instructions stop saying every MCP tool runs an analysis (STD-24). README's "placeholder codes" becomes "sentinel values".

## Testing Decisions

### The seam

The public `agent_eda` API, plus the repository files that tests already read: README.md, AGENTS.md, the workflow files and `pyproject.toml`. This spec adds `.scratch/` and `docs/agents/` to that list. Guard tests inspect the package through `ae` and its submodules, reading `__init__.py` as text where needed, as `tests/test_suite_rules.py` does.

### What makes a good test here

A guard test fails with a message that names the fix, such as "export count_rows is not registered: add @analysis(...) or rename it with a leading underscore". Check each guard against a deliberately broken copy (a mutant) once, and record that it failed in the ticket.

### Test groups

1. **Tracker:** every ticket file has a `Status:` line with a label from `triage-labels.md` and a `Blocked by:` line, and every spec in ROADMAP.md exists.
2. **Commands:** the minimum-versions commands in ci.yml equal AGENTS.md's.
3. **Registration and exports:** every public analysis function is registered, and the three export lists agree.
4. **Error codes:** documented and in use agree, both ways.
5. **README:** every registered analysis is in exactly one table, there are no relative links, and the generated Reference is current.

## Out of Scope

- Generated `llms.txt` or `SKILL.md`: no spec asks for them yet.
- Rewriting finished tickets beyond their `Status:` line and one Comments line per superseded box.
- Moving the issue tracker to GitHub Issues.
- PyPI install wording, badges and the changelog, which the release spec covers.

## Further Notes

- The plan was settled in a grilling session on 2026-10-19. Every recommendation was accepted. The decisions behind this spec are round 1, Q5 and Q6.
- Finding IDs refer to `.scratch/audit/findings.md`.
- The minimal AGENTS.md change that points agents at ROADMAP.md was made together with this spec, so that agents reach ticket 01.
