# 03: Reading data

**What to build:** An MCP caller passes a CSV, TSV or Parquet file, gzipped or not, by absolute or relative path. A wrong path or an unsupported file gets an error that says how to fix it, and repeated calls on the same large file read it once.

Read these sections of `.scratch/mcp-server/spec.md`: Reading data, and test group 3. Covers user stories 11 to 18.

**Blocked by:** 02-every-analysis-as-an-mcp-tool

**Status:** ready-for-agent

- [ ] `.csv`, `.tsv`, `.csv.gz`, `.tsv.gz`, `.parquet` and `.pq` are read, case-insensitively. Parquet keeps its dtypes, nullable and pyarrow-backed ones included.
- [ ] CSV and TSV count only empty fields as missing; `"NA"`, `"N/A"`, `"NULL"` and `"-999"` reach the analyses as values, and `detect_missing_sentinels` over MCP finds them.
- [ ] Row IDs of a CSV are 0-based data-row numbers; `id_column` replaces them as in Python.
- [ ] Relative paths resolve against the server's working directory.
- [ ] An unknown extension raises `UnsupportedDataFormatError` (`unsupported_data_format`) listing the supported extensions.
- [ ] A missing file raises `DataFileNotFoundError` (`data_file_not_found`) with the absolute path tried, whether its directory exists, and up to 3 closest file names.
- [ ] A file pandas can't parse raises `DataFileError` (`unreadable_data_file`) quoting the parser's message.
- [ ] The cache keeps the 2 most recently read files: a second call on the same file doesn't read it again, a changed file (mtime or size) is read again, and a third file evicts the least recently used. The tests observe reads without importing private modules (for example by changing the file's content without changing its key, or by counting opens with an audit hook).
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments
