# 17: Result objects and project files

What to build:
- results compare with `==`, survive a pydantic round trip, serialize every row ID and print any text;
- `Project.save` writes only what `Project.load` accepts, atomically, never blocking on a special file, and records a format version.

Read these sections of `.scratch/correctness/spec.md`: Result objects, and test group 11. This ticket covers user stories 28 and 29. Findings: CORE-5, CORE-13, CORE-14, CORE-10, CORE-8, CORE-9, STD questionable decision (`save` truncates in place), and round 3, Q7b.

Blocked by: 11

Status: done

- [x] `r1 == r2`, `r1 != r2` and `r2 in [r1]` work for all 11 result classes and never raise. Two identical calls compare equal.
- [x] `type(res).model_validate(res.model_dump())` keeps every `to_frame()` table, or makes `to_frame()` raise a clear error.
- [x] An object `id_column` or index holding `np.timedelta64(5000, "Y")` gives a result whose `to_json()` succeeds.
- [x] A column label `"caf\udce9"` prints without error, and `to_json()` encodes as UTF-8.
- [x] A label containing U+202E (right-to-left override) is escaped in the printed text.
- [x] `project.save(path, overwrite=True)` meets these conditions:
  - it refuses a FIFO or any other non-regular file;
  - it writes through a temporary file and a rename;
  - it refuses a project bigger than `Project.load` accepts, with a message that names the limit.
- [x] Project files carry `format_version = 1`. A file without the key loads as version 1, and a higher version is refused with a message naming it.
- [x] The README's project-file section and the `ae.errors` entry for `invalid_project_file` list every case: not a regular file, over 1 MiB, unreadable, and newer format (SPEC-18).
- [x] The finish checks pass.

## Comments

- `==` on a result compares its full record as JSON (row IDs as `row_id_json` writes them, no list cut), and only between results of the same class. The tables that `to_frame()` builds from the call stay out of it, since they follow from the record.
- The round trip chose the second option of the box: `model_validate(model_dump())` keeps every table that the record holds, and `to_frame()` raises the new `ResultTableUnavailableError` (code `result_table_unavailable`, added to `ae.errors` and the README table) only where a level table is needed and was lost: `summarize_categorical`, `profile_target` (classification) and `validate_column_types` (`table="invalid_values"`). A rebuilt result whose record holds no levels still returns the empty table. `copy.deepcopy` and `model_copy` keep the tables, and the message says so.
- `row_id_json` falls back to `str(value)` on `ValueError` and `OverflowError`, so `np.timedelta64(5000, "Y")` becomes `"5000 years"`.
- The text template now escapes categories Cs and Cf as well (so U+200D joiners in emoji sequences print as `\u200d`). `to_json()` writes a lone surrogate as its `\uXXXX` escape, which is valid JSON, encodes as UTF-8 and reads back as the same label with `json.loads`. The discovery records share that writer.
- `Project.save` follows a symlink and replaces the file it points to. It refuses an existing target that isn't a regular file with `InvalidArgumentError`, and a project over 1 MiB with `ProjectFileError` (`invalid_project_file`), both before anything is written. With `overwrite=False` the temporary file is hard-linked into place, so a file that appears meanwhile is never replaced; a file system without hard links falls back to an exclusive create.
- Security review follow-up: with `overwrite=False` the check is `lexists` on the path as given and the file is hard-linked or created with `O_EXCL`, so a dangling link is refused and never written through. With `overwrite=True` a link is followed only when its target exists; a dangling link is replaced itself. A new file gets mode 0600 (it names sensitive columns), a replaced file keeps its permissions through `fchmod` on the temporary file.
- `format_version` is an optional top-level key, not a Project field. It must be a positive integer, and a value above 1 raises `invalid_project_file` naming the version. The MCP `project_path` schema text is unchanged, since the key is optional for hand-written files.
- Checks: ruff, ty, `uv run pytest` (2415 passed), both minimum-version runs, the MCP stdio smoke and the benchmark all pass locally. Nothing is GitHub-only.
