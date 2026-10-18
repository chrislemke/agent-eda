# 06: Full tables with table_path

**What to build:** An MCP caller who needs more than the capped JSON record passes `table_path`, and the server writes the complete `to_frame()` table there as CSV or Parquet, refusing to overwrite anything, and says where it wrote it.

Read these sections of `.scratch/mcp-server/spec.md`: Responses, and test group 6. Covers user stories 21 and 22.

**Blocked by:** 05-project-and-masking

**Status:** ready-for-agent

- [x] Every analysis MCP tool has an optional `table_path` (string or null) with fixed server text as its description, and the contract test checks it.
- [x] `table_path` ending in `.csv`, `.parquet` or `.pq` writes `res.to_frame()`; reading the file back gives the same rows and columns, with the index written as a column unless it is a default `RangeIndex`.
- [x] The text gains the line `Full table written to <absolute path> (<n> rows).`, and nothing else changes in the response.
- [x] An existing file, an unsupported extension or a missing parent directory raises `TableFileExistsError` (`table_file_exists`) or an `InvalidArgumentError` before the analysis runs, and no file is written.
- [x] With masking on, the written table holds no raw value of a sensitive column.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

**Where it lives.** Private `agent_eda._table_files` (`table_file(name, table_path)` checks, `TableFile.write(frame)` writes and returns the response line). `_call` checks `table_path` right after the unknown/missing-argument checks, before the project and the data are read; `_response(result, table)` appends the line. `TableFileExistsError(AgentEDAError, FileExistsError)`, code `table_file_exists`, is in `agent_eda.errors` and its module docstring. `table_path` is the last property of every input schema, with fixed server text (`_TABLE_PATH`) that the contract test checks is identical on every MCP tool.

**What the ticket left open, settled here.**
- Errors: an existing file (or a symlink) gives `table_file_exists`, whose message suggests the first free `<stem>-2<ext>`, `-3`, ...; an unsupported extension, a missing parent directory, a directory at the path and a non-string `table_path` give `invalid_argument`. Extensions are case-insensitive, like data paths; `.csv.gz` and `.tsv` are refused. The server never creates directories.
- Never overwriting holds after the check too: the file is created with `open("xb")`, so a file that appears while the analysis runs gives `table_file_exists`, and the existing file stays. A missing directory or a permission error at that point gives `invalid_argument`. The content is built in memory first, so a conversion error never leaves a partial file.
- Cells: a list, tuple, set or dict (row IDs, evidence, sentinel_values, candidate key columns) is written as JSON text via `row_id_json`, in both formats. Parquet can't store an object column that mixes types (detect_missing_sentinels' `value`, summarize_categorical's `label`, both beside a `*_type` column) or an empty dict as a struct, so in Parquet a column pyarrow rejects is written as text, `str(value)`, as the CSV shows it; missing stays null. Every other column keeps its dtype in Parquet. CSV: UTF-8, `\n` line ends, missing as an empty field, which the server's reader reads back as missing.
- Index: written as the first column(s) via `reset_index()` unless it equals `RangeIndex(len)` with no name. Today every analysis's `to_frame()` has a default `RangeIndex`; a stand-in with a named index covers the rule.
- The line says `(1 row)` for one row and uses thousands separators (`_text.plural`), like the printed result.
- `to_frame()` is called with no arguments, so analyses with several tables (`compute_association_matrix` `table="matrix"`, `profile_missing_values`, `infer_column_semantics`, `validate_column_types`) write their default table. A `table` choice over MCP is not part of v1.

**For ticket 07/08.** Instructions and the README should say that `table_path` writes the full table (`to_frame`), never overwrites, and that lists are JSON text in the file. Tests reading Parquet back on pandas 2.2 + pyarrow 14 filter the BlockManager DeprecationWarning (`tests/test_mcp_tables.py:read_parquet`).

Checks: `ruff check`, `ruff format`, `ty check` and `pytest` pass on the latest versions (with `--extra mcp`, local uv 0.10.11; 2042 passed), and in both minimum-versions runs (core: 1835 passed, MCP modules skipped; with the extra: 2042 passed). CI wasn't run.
