# 06: Full tables with table_path

**What to build:** An MCP caller who needs more than the capped JSON record passes `table_path`, and the server writes the complete `to_frame()` table there as CSV or Parquet, refusing to overwrite anything, and says where it wrote it.

Read these sections of `.scratch/mcp-server/spec.md`: Responses, and test group 6. Covers user stories 21 and 22.

**Blocked by:** 05-project-and-masking

**Status:** ready-for-agent

- [ ] `table_path` ending in `.csv`, `.parquet` or `.pq` writes `res.to_frame()`; reading the file back gives the same rows and columns, with the index written as a column unless it is a default `RangeIndex`.
- [ ] The text gains the line `Full table written to <absolute path> (<n> rows).`, and nothing else changes in the response.
- [ ] An existing file, an unsupported extension or a missing parent directory raises `TableFileExistsError` (`table_file_exists`) or an `InvalidArgumentError` before the analysis runs, and no file is written.
- [ ] With masking on, the written table holds no raw value of a sensitive column.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments
