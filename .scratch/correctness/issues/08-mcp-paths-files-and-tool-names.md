# 08: MCP paths, files and tool names

What to build: every path problem becomes a coded tool error that names its cause. `~` works in every path argument, `table_path` never leaves partial files or follows dangling symlinks, an unknown MCP tool name gets suggestions, and the test hook on `create_server` becomes private.

Read these sections of `.scratch/correctness/spec.md`: Paths and files over MCP, and test group 7. This ticket covers user stories 21 and 22. Findings: MCP-4, MCP-5, MCP-8, SPEC-13, STD-12, MCP-13, MCP-16, MCP-17, STD-40, STD-41, STD-42, SPEC-14, STD-43, plus round 3, Q7c.

Blocked by: 07

Status: ready-for-agent

- [ ] For `data_path` and `table_path`, each of these comes back as a coded tool error naming the cause, and none as JSON-RPC error -32603:
  - a file name over 255 characters;
  - a directory without search permission;
  - a symlink loop;
  - an embedded NUL;
  - a read-only volume.
- [ ] A `table_path` on a read-only location is refused before the analysis runs wherever that can be detected. A write that fails partway removes its partial file; test it with a simulated failure.
- [ ] `~` expands in `data_path`, `table_path` and `project_path` alike.
- [ ] A dangling symlink given as `table_path` is refused, and nothing is written anywhere else.
- [ ] A Parquet file with broken pandas metadata gives `unreadable_data_file`. A uint64 sentinel of 18446744073709551615 written to a `.parquet` `table_path` succeeds or gives a coded error.
- [ ] Reader messages name the actual cause:
  - not a regular file;
  - permission denied;
  - duplicate Parquet field names;
  - the last suffix of "sales.2024.json";
  - a hint about the separator when the only column name contains `;`.
- [ ] An unknown MCP tool name, such as "Describe_Dataset", gets the closest names and the list of MCP tools. "guide" and "describe_analysis" point to the instructions and the tool descriptions.
- [ ] A `table_path` CSV writes the empty string quoted, and reading it back gives "" rather than a missing value.
- [ ] The `table_path` line keeps to 120 characters, and the text keeps to 40 lines.
- [ ] `create_server(analyses=...)` is private, and the README and docs no longer mention it.
- [ ] The finish checks pass.

## Comments
