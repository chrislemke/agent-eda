# 08: MCP paths, files and tool names

What to build: every path problem becomes a coded tool error that names its cause. `~` works in every path argument, `table_path` never leaves partial files or follows dangling symlinks, an unknown MCP tool name gets suggestions, and the test hook on `create_server` becomes private.

Read these sections of `.scratch/correctness/spec.md`: Paths and files over MCP, and test group 7. This ticket covers user stories 21 and 22. Findings: MCP-4, MCP-5, MCP-8, SPEC-13, STD-12, MCP-13, MCP-16, MCP-17, STD-40, STD-41, STD-42, SPEC-14, STD-43, plus round 3, Q7c.

Blocked by: 07

Status: done

- [x] For `data_path` and `table_path`, each of these comes back as a coded tool error naming the cause, and none as JSON-RPC error -32603:
  - a file name over 255 characters;
  - a directory without search permission;
  - a symlink loop;
  - an embedded NUL;
  - a read-only volume.
- [x] A `table_path` on a read-only location is refused before the analysis runs wherever that can be detected. A write that fails partway removes its partial file; test it with a simulated failure.
- [x] `~` expands in `data_path`, `table_path` and `project_path` alike.
- [x] A dangling symlink given as `table_path` is refused, and nothing is written anywhere else.
- [x] A Parquet file with broken pandas metadata gives `unreadable_data_file`. A uint64 sentinel of 18446744073709551615 written to a `.parquet` `table_path` succeeds or gives a coded error.
- [x] Reader messages name the actual cause:
  - not a regular file;
  - permission denied;
  - duplicate Parquet field names;
  - the last suffix of "sales.2024.json";
  - a hint about the separator when the only column name contains `;`.
- [x] An unknown MCP tool name, such as "Describe_Dataset", gets the closest names and the list of MCP tools. "guide" and "describe_analysis" point to the instructions and the tool descriptions.
- [x] A `table_path` CSV writes the empty string quoted, and reading it back gives "" rather than a missing value.
- [x] The `table_path` line keeps to 120 characters, and the text keeps to 40 lines.
- [x] `create_server(analyses=...)` is private, and the README and docs no longer mention it.
- [x] The finish checks pass.

## Comments

- Codes. A NUL character in any path argument is `invalid_argument`, found in `path_argument`. For `data_path`, "no such file" and "not a directory" stay `data_file_not_found`, and every other refusal (name too long, symlink loop, no permission, not a regular file) is `unreadable_data_file`. For `table_path`, every refusal is `invalid_argument`, except an existing file or symbolic link, which stays `table_file_exists`. `project_path` problems stay `invalid_project_file`, with the cause named instead of "give read permission".
- Read-only volume: found before the analysis with `os.statvfs(...).f_flag & ST_RDONLY`, then `os.access(directory, W_OK | X_OK)` for permission. The test simulates the volume with a patched `os.statvfs`, since a real read-only volume needs privileges. The partial-file test patches `os.write` to fail with ENOSPC after 5 bytes. `TableFile.write` now creates the file with `os.open(O_EXCL)` and `os.write`, and removes the file on any failure.
- Dangling symlink: the path is checked as given (directory resolved, last name kept) with `lstat`, so a link at the end is refused as `table_file_exists` with its own message.
- Quoted empty strings: pandas reads `""` as missing, so the reader marks each field that is just `""` (found by the parity of quote characters, with a delimiter, line break or file edge on both sides) with two control bytes that the file doesn't hold in a row, reads the bytes from memory, and sets those cells to `""` afterwards. A column holding one is text. This only happens for files that contain `""` at all; other files are read as before. A quoted empty header name becomes `Unnamed: n`.
- Semicolon hint: a CSV or TSV whose only column name holds `;` is refused with `unreadable_data_file`, since a one-column file named `a;b` is much rarer than a semicolon-separated one; the message says to rename the column if the name is real.
- Extension text: the last suffix, or the last two when the last is `.gz` (so `sales.json.gz` still shows `.json.gz`).
- Text caps: `AnalysisResult._render` takes `reserve`, the lines the caller adds. The `table_path` line is one line when it fits in 120 characters, otherwise the path on a line of its own, otherwise a pointer to `table_file.path` in the structured content, and the result's text gives up one or two lines when it has 40. Existing tests that compared the one-line form handle the longer macOS temporary paths with a helper.
- `create_server(analyses=...)` is now `agent_eda.mcp._create_server`; `create_server(*, project=None)` is the public one. Tests reach the hook through `tests/serving.py`, because the suite rules forbid importing private names. README and docs never mentioned it.
- Parquet: duplicate column names are found from the schema before reading. `KeyError`, `TypeError`, `IndexError`, `AttributeError`, `OverflowError` and pyarrow's exceptions are caught by the reader; a Parquet `table_path` writes a column that holds a Python int past 64 bits as text, as for mixed types, and any other encoding failure is `invalid_argument` that suggests the other format.
- Not run locally: the test that pads a path to between 100 and 120 characters skips when the temporary directory's own path is too long (it is on this macOS machine); it passed with `--basetemp=/private/tmp/claude-501/bt`. The suite passed on Python 3.12 with numpy 1.26 and pandas 2.2 (both minimum runs), the MCP stdio smoke test and the benchmark. No GitHub-only check applies.
