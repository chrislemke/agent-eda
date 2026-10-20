# 03: Reading data

**What to build:** An MCP caller passes a CSV, TSV or Parquet file, gzipped or not, by absolute or relative path. A wrong path or an unsupported file gets an error that says how to fix it, and repeated calls on the same large file read it once.

Read these sections of `.scratch/mcp-server/spec.md`: Reading data, and test group 3. Covers user stories 11 to 18.

**Blocked by:** 02-every-analysis-as-an-mcp-tool

**Status:** done

- [x] `.csv`, `.tsv`, `.csv.gz`, `.tsv.gz`, `.parquet` and `.pq` are read, case-insensitively. Parquet keeps its dtypes, nullable and pyarrow-backed ones included.
- [x] CSV and TSV count only empty fields as missing; `"NA"`, `"N/A"`, `"NULL"` and `"-999"` reach the analyses as values, and `detect_missing_sentinels` over MCP finds them.
- [x] Row IDs of a CSV are 0-based data-row numbers; `id_column` replaces them as in Python.
- [x] Relative paths resolve against the server's working directory.
- [x] An unknown extension raises `UnsupportedDataFormatError` (`unsupported_data_format`) listing the supported extensions.
- [x] A missing file raises `DataFileNotFoundError` (`data_file_not_found`) with the absolute path tried, whether its directory exists, and up to 3 closest file names.
- [x] A file pandas can't parse raises `DataFileError` (`unreadable_data_file`) quoting the parser's message.
- [x] The cache keeps the 2 most recently read files: a second call on the same file doesn't read it again, a changed file (mtime or size) is read again, and a third file evicts the least recently used. The tests observe reads without importing private modules (for example by changing the file's content without changing its key, or by counting opens with an audit hook).
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

- The reader lives in the private module `agent_eda._data_files` (`DataReader`, `resolve`, `FORMATS`, `supported_extensions`), which imports no `mcp`. `create_server` makes one `DataReader` per server, so the cache is per server instance, not module state; `_mcp_server.read_data` is gone.
- Only local files are read: `resolve` turns the data path into an absolute `pathlib.Path` against `Path.cwd()` and the reader opens it itself, handing pandas a binary file handle. A path such as `http://…` or `s3://…` is a relative local path, and a missing one raises `data_file_not_found` (tested with `.invalid` hosts). No root restriction (ADR 0006).
- Error classes in `agent_eda.errors`: `UnsupportedDataFormatError(InvalidArgumentError)`, `DataFileError(InvalidArgumentError)`, and `DataFileNotFoundError(DataFileError, FileNotFoundError)`, following the `ProjectFileNotFoundError` pattern. The extension check runs before the existence check. A data path that names a directory is `data_file_not_found` too. The parser exceptions turned into `DataFileError` are `ValueError`, `OSError`, `EOFError` and `zlib.error`, which cover bad UTF-8, an empty file, ragged rows, a file that isn't gzip and a file that isn't Parquet.
- The pandas 2.2 + pyarrow 14 combination warns twice with a DeprecationWarning on `read_parquet`: "Passing a BlockManager to DataFrame is deprecated" on every read, and "DatetimeTZBlock is deprecated" on a tz-aware column. The reader filters out just these two messages, so the extra keeps `pyarrow>=14` and the lock stays as it is.
- The `data_path` description in every input schema now names the six extensions and says that Parquet keeps its dtypes and its stored index.
- Errors are observed over MCP with `Client(..., raise_exceptions=True)`: the client re-raises the handler's `AgentEDAError` inside nested exception groups, and the test helper `data_error` in `tests/test_mcp_data.py` unwraps it. Once ticket 04 returns an `AgentEDAError` as an `isError` result, that helper has to read the code and message from the result instead.
- The cache tests rewrite a file to the same size and then put back its modification time with `os.utime(ns=...)`, so they see whether a call re-read the file without touching private modules.

**Spec review follow-up (2026-10-18).** "A pandas index stored in the file is restored ... and row IDs follow it" had no test; `test_row_ids_of_a_parquet_file_follow_the_index_stored_in_it` now covers it.
