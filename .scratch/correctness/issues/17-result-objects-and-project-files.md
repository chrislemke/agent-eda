# 17: Result objects and project files

What to build:
- results compare with `==`, survive a pydantic round trip, serialize every row ID and print any text;
- `Project.save` writes only what `Project.load` accepts, atomically, never blocking on a special file, and records a format version.

Read these sections of `.scratch/correctness/spec.md`: Result objects, and test group 11. This ticket covers user stories 28 and 29. Findings: CORE-5, CORE-13, CORE-14, CORE-10, CORE-8, CORE-9, STD questionable decision (`save` truncates in place), and round 3, Q7b.

Blocked by: 11

Status: ready-for-agent

- [ ] `r1 == r2`, `r1 != r2` and `r2 in [r1]` work for all 11 result classes and never raise. Two identical calls compare equal.
- [ ] `type(res).model_validate(res.model_dump())` keeps every `to_frame()` table, or makes `to_frame()` raise a clear error.
- [ ] An object `id_column` or index holding `np.timedelta64(5000, "Y")` gives a result whose `to_json()` succeeds.
- [ ] A column label `"caf\udce9"` prints without error, and `to_json()` encodes as UTF-8.
- [ ] A label containing U+202E (right-to-left override) is escaped in the printed text.
- [ ] `project.save(path, overwrite=True)` meets these conditions:
  - it refuses a FIFO or any other non-regular file;
  - it writes through a temporary file and a rename;
  - it refuses a project bigger than `Project.load` accepts, with a message that names the limit.
- [ ] Project files carry `format_version = 1`. A file without the key loads as version 1, and a higher version is refused with a message naming it.
- [ ] The README's project-file section and the `ae.errors` entry for `invalid_project_file` list every case: not a regular file, over 1 MiB, unreadable, and newer format (SPEC-18).
- [ ] The finish checks pass.

## Comments
