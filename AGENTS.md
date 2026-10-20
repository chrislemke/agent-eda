# agent-eda

An open-source library of deterministic EDA analyses. Its callers are agents in AWS Bedrock AgentCore (via Code Interpreter), coding agents, and people. Agent-friendliness is the top priority: judge every public name, signature, docstring, result and error message by how an agent will find it, call it and read what comes back.

## Commands

pyproject.toml makes uv refuse to run below the version these commands need (`required-version`). Installing and running through uv gives the mcp extra and the dev tools, on the latest numpy 2.x and pandas 3.x.

```sh
uv sync                                               # install the package, the mcp extra and the dev tools
uv run pre-commit install                             # install the git hooks: ruff and ty on commit, pytest on push
uv run ruff check --fix .                             # lint, fixing what it can
uv run ruff format .                                  # format
uv run python scripts/generate_reference.py           # rewrite the README's Reference from the docstrings, signatures and ae.errors
uv run python scripts/generate_reference.py --check   # exit 1 when the Reference is stale, without writing
uv run python scripts/record_baseline.py --check      # exit 1 where a record, text, table or message differs from tests/baseline
uv run python scripts/record_baseline.py              # refresh the byte baseline, only where a change is meant to alter an output
uv run python scripts/record_pickles.py               # write a pickle fixture for each result class (see tests/fixtures/pickles)
uv run pytest --hypothesis-profile=thorough           # test with 1,000 generated examples per analysis instead of 50
uv run pytest --cov                                   # test with coverage, subprocesses included
uv run --extra mcp pytest tests/test_mcp_*.py         # test the MCP server alone
uv add --dev <package>                                # add a dev dependency and update uv.lock
uv build --wheel                                      # build the wheel
uv run pre-commit run --all-files                     # run every hook on every file
```

### Finish checks

Run these before finishing a change. CI runs the same ones, and the git hooks run ruff and ty on commit and pytest on push. Fix what they flag.

1. Lint, format and types: `uv run ruff check .`, `uv run ruff format --check .` and `uv run ty check`.
2. Tests, the MCP server tests included: `uv run pytest`.
3. Tests on the minimum versions (Python 3.12, numpy 1.26, pandas 2.2), two runs. Keep `--isolated` on both: it leaves `uv.lock` unchanged, and without it uv rewrites the lock.
   - Without the mcp extra, on pydantic 2.7, where the MCP tests skip: `uv run --isolated --resolution lowest-direct --no-group mcp --group core --python 3.12 pytest`
   - With the extra at its floors, which lift pydantic to 2.12: `uv run --isolated --resolution lowest-direct --extra mcp --python 3.12 pytest`
4. The MCP server over real stdio, which starts `agent-eda-mcp` and calls one MCP tool: `uv run --extra mcp python -I tests/smoke_mcp.py`.
5. The benchmark, `uv run pytest -m benchmark`, after changing an analysis whose cost grows with rows or columns, or code that such an analysis calls. The default test run skips it.

## Sources of truth

- `GLOSSARY.md`: use its terms in code, docstrings and messages. Check it before naming anything.
- `docs/adr/`: read the ADRs before touching dependencies, the call pattern, data policy behaviour, anything that would add LLM or network use, or (ADR 0006) the MCP server, data reading and anything an MCP caller reads.
- `docs/planned-analyses.md`: the backlog of planned analyses. Its names and modules are proposals. When an analysis ships, move it from there to the README's Analyses section: into the first-look table when `ae.guide()` makes it a step of the first look, otherwise into the "Other analyses" table under its module. From then on the code and docstring are authoritative.

## Working on a ticket

Specs and tickets live under `.scratch/<feature>/` (see `docs/agents/issue-tracker.md`). `.scratch/ROADMAP.md` lists the specs in the order they are worked on. The current spec is the first one there with a `ready-for-agent` ticket that isn't `Status: done` and whose `Blocked by:` tickets are all `Status: done`. When no spec has such a ticket, ask the maintainer which spec comes next.

1. In the current spec, take the lowest-numbered such ticket. Skip tickets marked `ready-for-human` or `needs-info`: they are for the maintainer. Read the ticket, the spec sections it names, and the ADRs behind them.
2. Work test-first with the `/tdd` skill: for each acceptance box, write a failing test through the public API, then the code that makes it pass.
3. Tick a box (`- [x]`) once a test or a command shows it holds.
4. Under `## Comments` at the end of the ticket, record what the ticket settled that the spec left open, any measurement it asks for, and any check you couldn't run locally.
5. A ticket is finished when every box is ticked and the Finish checks under Commands pass. Set its `Status:` line to `done`, and commit with the message `NN: <ticket title>`.

The ticket is the acceptance bar. Where it settles a point the spec left open, the ticket wins, and the spec overrides the proposed names in `docs/planned-analyses.md`. Later tickets extend the shared parts that earlier ones built (the result base class, the text template, the registry, the errors, the column and row ID rules, masking). If a ticket contradicts an ADR, flag it as `docs/agents/domain.md` describes instead of overriding the ADR.

## Layout

- `src/agent_eda/`: the package, in a src layout built by the uv build backend. An analysis lives in the submodule that `docs/planned-analyses.md` gives it, such as `agent_eda.data.missingness` or `agent_eda.statistics.descriptive`. Exceptions live in the public `agent_eda.errors` module. Result classes are given by the public `agent_eda.results`, a lazy facade with an export table, and defined in a private `_<module>_results.py` next to their analysis (`data/_structure_results.py`); the shared base is `_result_base.py` and the discovery records `_discovery_results.py`.
- A module's location says its scope. Shared helpers sit at the package root (`_arguments` reads and words the arguments of a call, `_registry` holds only the registry, `_status` and `_warnings` serve a run). The MCP adapter sits in the private `_mcp/` package, and `agent_eda.mcp` stays its public entry point. An engine that serves one analysis sits beside it as a private sibling (`data/_keys.py`, `statistics/_association.py`). A module imports no other module's underscore names, except those of `_result_base` into the `_*_results` modules and the export table of the package root into `_registry`; `tests/test_layout.py` checks it.
- `tests/`: pytest tests. They import only `agent_eda`.

## Rules for every analysis

Before adding or renaming an analysis, follow docs/agents/new-analysis.md.

- **Call pattern.** `def analysis(data: pd.DataFrame, *, ...) -> AnalysisSpecificResult`. Data first, everything else keyword-only. Columns are referenced by name, in parameters ending in `column` or `columns`; `columns=None` means all suitable columns. Analyses over several DataFrames name each by role (`before=`/`after=`, `sample=`/`population=`). Treat `data` as read-only.
- **Typed arguments.** Fixed choices such as `method` are `Literal[...]`. Options are typed keyword arguments; genuinely nested specs are small pydantic models. Free-form dicts (`dict[str, Any]`, JSON mappings) are not part of any public signature. Numeric arguments have documented valid ranges.
- **Defaults.** Only standard conventions (`seed=0`, `confidence_level=0.95`, quantile levels) and work or display limits (`max_*`, `top_n`) get defaults, and the result records the value used. Project-specific values such as thresholds, units, schemas, sentinel values and expected labels never get one: they are required, or optional with `None` meaning the check that needs them doesn't run and the result says so.
- **Project.** Analyses take an optional `project=` when dataset roles matter or when their results can show raw values; the second group also takes `mask_sensitive=`. Without a project, role checks report "not checked", never "passed". Masking is off by default, and a per-call `mask_sensitive` wins over the Project's setting. Masking is a guardrail against accidental exposure, not anonymization or access control (ADR 0001).
- **Results.** Each analysis has its own pydantic result class named after it (`summarize_numeric` returns `SummarizeNumericResult`) that subclasses `AnalysisResult`. Its fields sit next to the shared ones (`status`, `parameters`, `support`, `findings`, `limitations` and the rest), and every field has a description. `print(res)` gives plain text from a fixed template, at most 40 lines of 120 characters, and counts what it leaves out. `to_dict()` and `to_json()` give the full record as strict JSON, in which every list that grows with the data is capped and carries its total: 10 row IDs, 5 example values, 100 entries for anything else. `to_frame()` gives the complete tables. A value that doesn't exist for the data is null, and a limitation names the field path and the reason; NaN and infinity never reach the record. Fractions are named `…_share`. Results have no HTML display.
- **Row IDs.** Results point at rows by DataFrame index label, or by the value in `id_column=` when given. Matching rows across DataFrames always uses explicit key columns.
- **Statuses.** `completed`, `partial`, `insufficient_data`, `not_applicable`, `missing_prerequisites`, `budget_exceeded`, `blocked_by_policy`, `failed`. A status says how the run ended; a finding's assessment says what the data showed. Each analysis declares the statuses it can return and never returns another, so a status no analysis can return is declared by none and listed nowhere (`failed` is for an anticipated numerical failure with the reason code `numerical_failure`; none declares it yet). A `status_reason`'s code comes from `StatusReasonCode` in `_result_base.py`, with its meaning in `STATUS_REASON_MEANINGS` beside it (data, not a comment, so it survives an install with no source): nothing to work on is `no_rows`, `no_columns` or `no_values`, and the message says which. The `Statuses:` entry of each status but `completed` ends with `Reason code: ...`, and the registry raises when a result returns another code. Limitation codes (`LimitationCode`, with `LIMITATION_MEANINGS`) follow the same rule, and exclusion reasons (`ExclusionReason`) take one comment per reason. An unexpected exception is a bug and propagates.
- **Errors.** Caller mistakes raise a subclass of `AgentEDAError` with a stable `code`, before any computation starts. The message names what is wrong, what is allowed and the fix: list the available columns and suggest the closest name. pydantic validation errors are re-raised as `InvalidArgumentError` in this format. Data problems return a status with a reason, and data issues go into findings and limitations.
- **Warnings.** Data problems never become Python warnings. Inside an analysis, numerical RuntimeWarnings become limitations, and DeprecationWarning and FutureWarning are suppressed for callers but fail the test suite. `warnings.warn` is reserved for renamed names.
- **MCP surface.** `agent-eda-mcp` serves every analysis as an MCP tool built from the registry and the docstring (ADR 0006), so what you write is read by MCP callers too. The docstring's sections become the tool description (`Examples:` left out, `See Also:` as names only) and `Args:` the input schema. A message an MCP tool can reach gives a fix an MCP caller can make: state it with `agent_eda._wording.mistake` as texts and fixes (`Pass`, `LeaveOut`, `SeeArguments`, `PythonOnly`, `DeclareSensitive`), never by asking which surface the caller is on, since an MCP caller passes JSON arguments and file paths, not Python. Every parameter type needs a JSON Schema, and a DataFrame passed by role needs a `<role>_path` argument. Nothing an MCP caller reads holds Python: no `res.columns[...]`, `df.loc[...]`, `to_frame()` or `ae.` calls.
- **Cost limits.** Analyses whose cost grows faster than the row count take `max_*` arguments and return `budget_exceeded` when a limit is hit.
- **Names.** Verb first, unique across the package, exported lazily from `agent_eda` so that type checkers and `dir()` still see them. Analyses that run statistical hypothesis tests keep natural `test_*` names; the registration decorator sets `__test__ = False` so pytest skips them. A renamed analysis keeps its old name working for two minor releases, with a warning that names the replacement.

## Docstrings

Google style. Every public analysis has these sections, in this order, under these exact headers:

1. A summary line phrased as the question the analysis answers, ending in "?".
2. `When to use:` when to use it, and which other analysis to use instead.
3. `Args:` every parameter, with type, unit, allowed values and what `None` means.
4. `Returns:` the fields to read, how to interpret them, and every declared finding code, one bullet each that starts with its assessment and a semicolon, as in ``- `constant_columns`: suspected; a column holds one value``.
5. `Statuses:` exactly the declared statuses, and why each occurs, each but `completed` ending in its reason codes (``Reason codes: `no_rows` and `no_values`.``).
6. `Examples:` a runnable doctest that prints a result.
7. `See Also:` related analyses.

`When to use:` names the trigger and at most three alternatives, and `See Also:` lists at most three analyses, among them the analysis's natural next step in the first-look order. Both name only analyses that exist, never `list_analyses` or another discovery function, since search and the guide reach the rest. When a new analysis is the better choice for a neighbour's question, update the neighbour's sections too, dropping the weakest alternative if it would make a fourth. A contract test enforces all of this. `guide()`, `search_analyses()` and `describe_analysis()` read the docstring at runtime, so it is the only hand-written copy. The README's Reference is generated from the docstrings, the signatures and `ae.errors`: after you change one, run the Reference generator (`scripts/generate_reference.py`, under Commands) and commit the README. A test fails when the Reference is stale.

## Implementation

- Deterministic: no LLM calls, no network access, no global state, no threads or processes of our own. Concurrent calls from several threads are safe: warning capture, the one process-wide state an analysis touches, runs under a module-level lock (`_warnings.capture_warnings`), so captures run one at a time, and so does any other change to the warning filters (`_warnings.scoped_filters`; a bare `warnings.catch_warnings` outside `_warnings` fails `tests/test_layout.py`), and a call never changes the caller's warning filters or another call's limitations. Randomness comes only from the `seed` argument. Results hold no timestamps or durations, and output order never follows hash or set iteration, so identical calls give byte-identical JSON across processes.
- pandas at the boundary, NumPy inside. Convert to NumPy arrays wherever the computation allows; keep pandas for grouping, nullable types, categoricals and time zones (ADR 0002).
- When two analyses compute the same thing (numeric summaries, parse rules, declared sentinel values, row IDs, masking), they call one shared implementation, so they can't disagree.
- Support Python 3.12 to 3.14, numpy 1.26 through 2.x and pandas 2.2 through 3.x, the minimums in `pyproject.toml`. The default environment only has the latest versions, so the minimum-versions run is what catches these:
  - NumPy 2-only functions such as `np.unique_counts`, `np.trapezoid` and `np.strings` don't exist in 1.26.
  - pandas 3 stores text in the `str` dtype by default, pandas 2.2 in `object`. Code that finds or reads text columns handles `object`, `string`, `str` and pyarrow-backed strings.
  - `to_numpy()` can return a view of the caller's data: read-only under pandas 3's copy-on-write, writable under 2.2. Copy before sorting or assigning in place.
- Pure Python. Required dependencies stay within AgentCore Code Interpreter's preinstalled set (ADR 0005). Anything else is an optional extra, and a missing extra raises `MissingDependencyError` with the install command.
- Load bundled files with `importlib.resources`, so the wheel still works when imported straight from a zip on `sys.path`.
- Performance target: each first-look analysis runs in under 5 seconds on 1,000,000 rows × 100 columns, as `tests/test_benchmark.py` checks.

## Tests

pytest, with Hypothesis for property tests.

- One seam: the public `agent_eda` API, used the way callers use it. A test imports the package, calls an analysis or discovery function and checks what comes back: result fields, status, findings, printed text, `to_frame()`, `to_json()`, or the exception and its message. Private modules and helpers are covered through the analyses that use them.
- Use small hand-built DataFrames where the right answer is obvious. Check numbers against reference values computed on the same data with SciPy, statsmodels, pandas or a published worked example, within a stated tolerance.
- Cover edge cases with Hypothesis: empty data, all missing, constant, one row, ties, infinities, mixed types, nullable, pyarrow-backed and pandas 2 and 3 string dtypes. Hypothesis runs derandomized, and every test is independent of the network, unseeded randomness and test order.
- A test function's name never reuses an analysis name.
- `tests/recipes.py`: a `Recipe` for a new analysis builds its valid arguments, which puts it into the property, masking, benchmark and MCP contract tests; one without a recipe fails the suite. `tests/strategies.py`: `data_frames()` draws DataFrames full of edge cases and `EDGE_FRAMES` holds fixed ones; reuse them rather than writing new generators.
- `tests/baseline/pandas<major>.json` holds the SHA-256 of every record, printed text, table, MCP tool definition and argument message, which `tests/test_baseline.py` compares with a fresh run, so a restructuring changes no byte. A change that alters an output on purpose, or adds an analysis or a message row, refreshes it with `scripts/record_baseline.py`, once for pandas 3 and once for pandas 2 with the minimum-versions command in the script's docstring; commit only the entries the change meant to alter.
- The repository's MCP server (`.mcp.json`) serves the code as it was when the Claude Code session started. After you change an analysis, restart the session before you check the change through its MCP tools, or you read the old behaviour.
- Before finishing a change, run the Finish checks under Commands. A `# noqa` or `# ty: ignore` carries its reason on the same line.

## Agent skills

### Issue tracker

Local markdown. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) and `done` for a finished ticket, recorded on each issue file's `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
