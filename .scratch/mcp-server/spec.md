# Spec: local MCP server

Status: ready-for-agent

## Problem Statement

agent-eda's callers are agents, and today they reach the analyses only by writing and running Python: `import agent_eda as ae`, load a DataFrame, call an analysis, print the result. That suits AgentCore Code Interpreter and coding agents with a shell, but not agents that only call tools, such as Claude Desktop, chat clients with MCP support, or a coding agent whose shell access is restricted. Such an agent falls back to whatever generic tools it has, and with them to the ad-hoc pandas code the package exists to replace.

Even an agent that can write Python pays for it on every call: it writes the import, the file read and the print, and it reads parameter documentation through `help()` or `ae.describe_analysis()` instead of getting a typed schema its client validates before the call.

## Solution

An optional MCP server ships with the package. `pip install 'agent-eda[mcp]'` adds the dependencies, and `agent-eda-mcp` starts the server as a stdio subprocess of the client.

- Every registered analysis is an MCP tool with the analysis's name. Its description and JSON Schema come from the analysis's docstring and registry entry, so an MCP caller gets the same documentation a Python caller gets, and nothing is maintained twice.
- An MCP tool takes a data path to a CSV, TSV or Parquet file in place of the DataFrame. The server reads it, counting only empty CSV fields as missing so that sentinel values reach the analyses.
- A call returns the printed result as text, worded for an MCP caller, and the full strict-JSON record as structured content. An optional `table_path` writes the result's main table to a file.
- A caller mistake comes back as a tool error whose message says how to fix the call: the message a Python caller gets, with the fix worded for an MCP caller (user's decision, 2026-10-18).
- A project file passed per call, or once when the server starts, gives masking to MCP callers.
- The server's instructions tell an MCP caller where to start, and `search_analyses` and `list_analyses` are MCP tools too.

## User Stories

### Installing and starting

1. As a data scientist, I want `pip install 'agent-eda[mcp]'` or `uv add 'agent-eda[mcp]'` to add the MCP server, so that the core package stays free of MCP dependencies when I don't need them.
2. As an AgentCore agent installing the wheel with `--no-deps`, I want the core package to work without the MCP extra, so that the server's dependencies never block my install.
3. As a user, I want `agent-eda-mcp` to start the server over stdio, so that I can register it with one command in Claude Code, Claude Desktop, Cursor or any other MCP client.
4. As a user who starts `agent-eda-mcp` without the extra installed, I want an error that names the install command, so that I fix it in one step.
5. As a contributor working in this repo, I want Claude Code to pick up the server from the repo's `.mcp.json`, so that I can try the analyses through MCP while I develop them.

### Finding the right MCP tool

6. As an MCP caller, I want the server's instructions to say what the package does, how to pass data, how to read results and which first-look order to follow, so that I can start without reading the README.
7. As an MCP caller, I want one MCP tool per analysis, named like the analysis, so that my client's tool search finds it and the name matches the docs.
8. As an MCP caller, I want each MCP tool's description to say what question it answers, when to use it instead of its neighbours, what the result fields mean and which statuses can occur, so that I pick the right tool and read the result correctly.
9. As an MCP caller, I want each parameter in the JSON Schema to carry its type, allowed values, valid range, default and description, so that my client validates the call and my first call is right.
10. As an MCP caller, I want `search_analyses` and `list_analyses` as MCP tools that return MCP tool names, so that I can find an analysis from my own wording.

### Passing data

11. As an MCP caller, I want to pass a path to a CSV, TSV or Parquet file, compressed with gzip or not, so that I can analyse the files I already have.
12. As an MCP caller, I want relative paths to resolve against the server's working directory, so that paths in my project work as I write them.
13. As an MCP caller, I want strings such as "N/A", "NULL" or "-999" in a CSV to reach the analyses as values, so that detect_missing_sentinels can find them and results agree with the same analysis run in Python on the same values.
14. As an MCP caller who mistypes a path, I want the error to show the absolute path the server tried and the closest file names in that directory, so that I fix the path in one step.
15. As an MCP caller who passes an unsupported file type, I want the error to list the supported extensions, so that I know what to convert to.
16. As an MCP caller running several analyses on the same large file, I want the server to read it once, so that the later calls are fast.
17. As an MCP caller who edits a file between calls, I want the next call to read the new version, so that I never get a result for stale data.
18. As an MCP caller, I want row IDs to be 0-based data-row numbers unless I pass `id_column`, so that I know what a row ID in a result points at.

### Reading what comes back

19. As an MCP caller, I want the printed result as the text of the response, so that I see status, key numbers, findings and limitations in at most 40 lines.
20. As an MCP caller, I want the full strict-JSON record as structured content, validated by an output schema, so that I can read any field without parsing text.
21. As an MCP caller, I want `table_path` to write the complete table to CSV or Parquet and the response to say where, so that I can get past the caps of the JSON record.
22. As an MCP caller, I want `table_path` to refuse to overwrite an existing file, so that a call can never destroy data.
23. As an MCP caller, I want the same call on the same file to return byte-identical structured content, so that my work is reproducible.

### Mistakes and statuses

24. As an MCP caller who makes a mistake in a call, I want a tool error carrying the stable error code and the message a Python caller gets, with a fix I can make over MCP, so that I can correct the call and retry.
25. As an MCP caller, I want a run that doesn't complete, such as `insufficient_data`, to come back as a normal result with its status and reason, not as an error, so that I can tell "my call was wrong" from "the data can't answer this".

### Masking

26. As an MCP caller, I want to pass `project_path` to the analyses that support masking, so that sensitive values are masked as in Python.
27. As a user who registers the server, I want `agent-eda-mcp --project project.toml` to set the Project for every call, so that an agent can't forget masking on a single call.
28. As an MCP caller, I want a per-call `project_path` to replace the server's project and a per-call `mask_sensitive` to win over both, so that the rules match Python's.
29. As a user, I want the docs and the MCP tool descriptions to say plainly that masking is a guardrail, not access control, so that nobody mistakes the server for an enforcement layer.

### Quality

30. As a maintainer, I want a contract test to fail when a registered analysis has no MCP tool or its tool can't be called with valid arguments, so that the server never falls behind the registry.
31. As a maintainer, I want the MCP tests to run in CI on the latest and minimum versions, and a smoke test over real stdio, so that a broken server can't be merged.

## Implementation Decisions

### Package and dependencies

- The server lives in the public module `agent_eda.mcp`, which the package does not import. `import agent_eda` and the discovery functions still import no MCP code, which the packaging tests check.
- The optional extra `mcp` requires `mcp>=2.3` (the official MCP Python SDK) and `pyarrow>=14.0.1` (14.0.0 and earlier have CVE-2023-47248). `mcp` 2.3 needs pydantic 2.12 or later, which is fine for an optional extra; the core minimum stays at 2.7.
- The console script `agent-eda-mcp` points at `agent_eda.mcp:main`. Without the extra, `main` raises the new `MissingDependencyError` (code `missing_dependency`, in `agent_eda.errors`), whose message names `pip install 'agent-eda[mcp]'` and the uv equivalent. The console script prints that message to stderr and exits with status 1, without a traceback.
- `MissingDependencyError` is general: later optional extras (lifelines and others, ADR 0005) reuse it.

### Server

- The server is named `agent-eda` and reports the package version.
- stdio transport only. No HTTP, no SSE, no network access.
- Analyses run synchronously on the server's event loop, one call at a time. The server starts no threads or processes of its own.
- Command-line options: `--project PATH`, the default Project. Nothing else in v1. `--help` documents it.
- The MCP tools are built from the registry when the server starts, so a newly registered and exported analysis becomes an MCP tool with no change to the server.

### MCP tools for analyses

- **Name**: the analysis's name, such as `summarize_numeric`.
- **Description**: built from the parsed docstring, in this order: the summary line, `When to use:`, `Returns:` (including the finding codes), `Statuses:` and `See Also:`. `Examples:` is left out because it is Python. See Also names are analysis names, which are also MCP tool names.
- **Input schema**: the analysis's registry `arguments` model's JSON Schema, with each property's `Args:` text as its description, as `describe_analysis()` already builds it. The server then makes these changes:
  - `data` becomes the required `data_path` (string).
  - `<role>_path` is deferred. Later analyses that take DataFrames by role, such as `before=` and `after=`, will get one `<role>_path` per role, such as `before_path` and `after_path`. v1 doesn't build it, because no analysis takes a role yet: today a DataFrame parameter other than `data` has no JSON Schema, so the server raises at start. The first role-taking analysis must extend the server with the rule, its reader calls and its tests.
  - `project` becomes `project_path` (string or null), on the analyses that take `project`. `mask_sensitive` stays as it is.
  - Every analysis MCP tool gets an optional `table_path` (string or null).
  - The descriptions of these three properties are fixed server text, not the docstring's.
- **Annotations**: `readOnlyHint: false`, because `table_path` writes a file; `destructiveHint: false`, because it never overwrites; `idempotentHint: true`; `openWorldHint: false`.
- **Output schema**: the JSON Schema of the result class in the shape `to_dict()` produces, with capped lists and their totals. A test validates real `to_dict()` output against it for every analysis.
- A parameter whose type has no JSON Schema (an arbitrary type such as a DataFrame) raises at server start, naming the analysis and parameter. That includes a role DataFrame until the deferred `<role>_path` rule is built. None of today's 11 analyses has one.

### Reading data

- The format comes from the file extension, case-insensitively: `.csv`, `.tsv`, `.parquet`, `.pq`, and `.csv.gz` and `.tsv.gz`. Anything else raises `UnsupportedDataFormatError` (code `unsupported_data_format`, an `InvalidArgumentError`), listing the supported extensions.
- CSV and TSV are read with pandas' default C parser, `sep` from the extension, and `keep_default_na=False, na_values=[""]`, so only an empty field is missing. Column labels are strings. The index is a `RangeIndex`, so row IDs are 0-based data-row numbers unless the caller passes `id_column`.
- Parquet is read with pandas and pyarrow, keeping the file's dtypes. A pandas index stored in the file is restored as pandas restores it, and row IDs follow it.
- No reader options in v1: no separator, encoding, index column or dtype overrides. CSVs are read as UTF-8.
- A file that pandas can't parse raises `DataFileError` (code `unreadable_data_file`, an `InvalidArgumentError`), quoting the parser's message.
- Relative paths resolve against the server's working directory. There is no root restriction; the server makes no access-control claim (ADR 0001, ADR 0006).
- A missing file raises `DataFileNotFoundError` (code `data_file_not_found`, a `FileNotFoundError` and `AgentEDAError`), giving the absolute path tried, whether its directory exists, and up to 3 closest file names in that directory from `closest_names`.
- An LRU cache keeps at most the 2 most recently read DataFrames, keyed by resolved path, modification time in nanoseconds and size. A changed file is read again. Analyses treat `data` as read-only, so the cached DataFrame stays valid across calls.

### Responses

- Text content: the printed result from the same template as `print(res)`, worded for an MCP caller. The footer names the full table that `table_path` writes and the structured content where Python names `res.to_frame()` and `res.to_json()`; it leaves out the tables only Python can ask for by name; record fields are named without `res.`, and settings are spelled as JSON arguments and TOML (`mask_sensitive: true`). Everything else in the text is the same as in Python. When `table_path` is given, one more line follows: `Full table written to <absolute path> (<n> rows).` This overrides the earlier "text equal to `print(res)`", by the user's decision on 2026-10-18.
- Structured content: `res.to_dict()`. When `table_path` is given, the record gains `table_file`, an object with the written file's absolute `path` and its `row_count`, and the output schema allows it without requiring it. Without `table_path` the structured content equals `res.to_dict()` exactly.
- Some clients show the model only one part of the response. Claude Code shows it only the structured content. Records measured 26 to 78 KB on 1,000 rows x 100 columns, where the printed text is about 1.2 KB. Keeping both parts costs that size in Claude Code's context, but dropping the record would lose the fields an agent reads, and dropping the text would lose the summary that clients showing text rely on. So both stay, `table_file` repeats the text's last line in the record, and the instructions don't promise that the caller sees the text.
- `table_path` writes `res.to_frame()` without arguments: the result's main table, format by extension (`.csv`, `.parquet`, `.pq`). Results with more tables, such as `compute_association_matrix`'s `table="matrix"`, keep the others for Python callers. An existing file raises `TableFileExistsError` (code `table_file_exists`, a `FileExistsError` and `AgentEDAError`) before the analysis runs, as does an unsupported extension or a missing parent directory. The index is written as a column when it carries information (not a default `RangeIndex`).

### Errors

- An `AgentEDAError` raised by argument checks, file reading or the analysis becomes a tool result with `isError: true`. Its text is the exception's message as an MCP caller reads it, and its structured content is `{"code": ..., "message": ...}`. That message says what is wrong and what is allowed as the Python one does; where the Python message gives a fix only Python can make, the MCP one gives the MCP fix in its place: a JSON object such as `{"task": "regression"}` for `task='regression'`, leaving an option out for `columns=None`, the MCP tool's input schema for `help(ae.<name>)`, `project_path` and the TOML of a project file for `ae.Project(...)`, and no step at all for pandas code that converts a column. Each raise site an MCP tool can reach builds both messages from one function of the surface (`agent_eda._wording`), and Python messages are unchanged.
- Validation by the SDK of arguments against the input schema happens before our code runs. Where the SDK lets the server shape that error, it carries code `invalid_argument` and names the property; otherwise the SDK's message stands, and ticket 04 records which.
- Any other exception is a bug and propagates, which the SDK turns into a JSON-RPC internal error.
- A non-completed status is not an error: the response is a normal result.

### Project and masking

- `project_path` is loaded with `Project.load`, so its errors (`ProjectFileError`, `ProjectFileNotFoundError`) and messages are the Python ones, with the fix worded for an MCP caller as Errors says. `Project.load` opens only a regular file, reads at most 1 MiB, and turns an operating-system refusal such as a missing read permission into `ProjectFileError`, so a named pipe, `/dev/zero` or an unreadable file gives a tool error instead of hanging the server or becoming an internal error.
- `--project` is loaded once when the server starts. An invalid file stops the server with the same message on stderr and exit status 1.
- Precedence: per-call `project_path`, then `--project`, then no Project. A per-call `mask_sensitive` wins over the Project's setting, as in Python.
- An analysis that doesn't take `project` has no `project_path`. `--project` doesn't apply to it.
- The server's instructions and the descriptions of `project_path` and `mask_sensitive` repeat that masking is a guardrail against accidental exposure, not anonymization or access control.

### Discovery

- The server's `instructions` are an MCP version of `ae.guide()`, built from the same parts: what the package does, how to pass data (data path, supported formats, the empty-field rule, relative paths, row IDs), how to read results and statuses, masking with `project_path` and `--project`, and the first-look order from `discovery.FIRST_LOOK` limited to registered analyses. They contain no Python. The guide's 60-line, 120-character limits apply.
- The `search_analyses` MCP tool takes `query` and `limit` (default 5) and returns the `SearchListing`'s text and record. The `list_analyses` MCP tool takes nothing and returns the `AnalysisListing`'s text and record. Their names are MCP tool names. The records equal Python's. The text is worded for an MCP caller: where Python suggests `ae.search_analyses(...)`, `ae.describe_analysis(...)` or `ae.guide()`, it points to the `limit` argument, the MCP tool's description and input schema, and `list_analyses` (user's decision, 2026-10-18).
- Nothing an MCP caller reads holds `ae.` or Python call syntax: the instructions, the printed results, the listings, the MCP tool descriptions, the input schemas (a model's description in `$defs` leaves out its `Examples:`), the output schemas' field descriptions, which name the full table as "to_frame, or table_path over MCP", and the error messages.
- `describe_analysis` and `guide` are not MCP tools: each analysis MCP tool's description and schema carry the same docstring content, and the instructions carry the guide.

### Docs and records

- ADR 0006 records the server's shape. The glossary defines MCP server, MCP tool and data path.
- The README gets an "Use over MCP" section: the install, `claude mcp add agent-eda -- agent-eda-mcp` (and with `--project`), a generic `mcpServers` JSON snippet, the data path rules, and the masking caveat.
- The repo gets a `.mcp.json` that runs `uv run --extra mcp agent-eda-mcp`.

## Testing Decisions

### The seam

For this feature the seam is the MCP protocol. Tests start the server through the SDK's in-memory client session and act as a client does: list tools, read descriptions and schemas, call tools, and check the text, the structured content, `isError` and the error code. Tests import only `agent_eda` and `mcp` (plus `jsonschema` for output-schema validation), and each MCP test module skips when `mcp` isn't installed. A smoke test launches `agent-eda-mcp` as a subprocess over real stdio and calls one MCP tool. Data files are written to pytest's `tmp_path`.

### What makes a good test here

The rules of the main suite apply: small hand-built data with obvious answers, no network, no unseeded randomness, no dependence on test order, and no test function named like an analysis. An MCP test compares an MCP call against the same analysis called in Python on the same DataFrame, so the server can never disagree with the package.

### Test groups

1. **Packaging.** The console script exists. Without the extra (simulated by blocking the `mcp` import in a subprocess), `agent-eda-mcp` exits 1 with the install command on stderr. `import agent_eda` and the discovery functions import no `mcp` module.
2. **Contract, looping over `ae.list_analyses()`.** Every analysis has an MCP tool with its name, description sections in order, a schema with `data_path`, `table_path` and, where it applies, `project_path`. A call with valid arguments on a small mixed-dtype CSV succeeds with a declared status. Its `to_dict()` validates against the output schema, and its structured content equals the Python call's `to_dict()`.
3. **Reading data.** Each format and compression; the empty-field rule (`"N/A"` stays a string, an empty field is missing); Parquet dtypes; unknown extension; missing file with closest names; unparseable file; relative paths; the cache (a second call doesn't re-read, a changed file is re-read, a third file evicts the oldest).
4. **Errors and statuses.** Each new error code; an `AgentEDAError` from an analysis (a misspelled column) arrives with its code and message; schema validation errors; a non-completed status is a normal result.
5. **Project.** `project_path`, `--project`, precedence, `mask_sensitive` override, an invalid project file at start and per call; with masking on, no raw sensitive value appears in text, structured content or the written table.
6. **table_path.** CSV and Parquet; the rows equal `to_frame()`; existing file refused before the analysis runs; the response line.
7. **Discovery.** The instructions name every registered analysis, contain no Python call syntax and keep within the guide's limits; `search_analyses` and `list_analyses` return the Python records with the text in MCP wording.
8. **Determinism.** Two identical calls in two server processes with different `PYTHONHASHSEED` values give identical structured content.

### CI

- `uv run pytest` installs the extra and runs the MCP tests, so every test job with the latest versions covers them.
- The existing minimum-versions run stays without the extra, because `mcp` needs pydantic 2.12 and the core must keep being tested on pydantic 2.7. In that run the MCP test modules skip with a reason naming the missing extra.
- A second minimum-versions run installs the extra at its floors (`mcp` 2.3, pyarrow 14.0.1, and the pydantic 2.12 that `mcp` needs) and runs the whole suite. The README's Development table gets its command.
- The stdio smoke test runs on Python 3.12 and 3.14.
- The packaging job's offline install without extras stays as it is, which shows the core doesn't need them.

## Out of Scope

- HTTP, SSE or any network transport; authentication.
- Access control or enforcement of the data policy (ADR 0001 stays).
- Dataset handles or server-side state beyond the read cache; inline data in calls.
- Reader options (separator, encoding, index column, dtypes) and formats beyond CSV, TSV and Parquet.
- MCP resources, prompts, sampling and elicitation; progress notifications and cancellation of a running analysis.
- Exposing a subset of analyses or a preset; revisit when the tool list grows long.
- A `describe_analysis` or `guide` MCP tool.
- Publishing to an MCP registry or packaging as an MCPB bundle.
- Wrappers for other frameworks (Strands, LangChain).

## Further Notes

- The design was settled in a grilling session on 2026-10-18; every recommendation was accepted.
- `mcp` versions before 2.3 are untested, so the floor is the version the tests run against.
- The catalogue plans hundreds of analyses. One MCP tool each is fine while MCP clients search tools on demand; if the list becomes a problem, a server option that exposes only a preset is the planned fix (ADR 0006).
