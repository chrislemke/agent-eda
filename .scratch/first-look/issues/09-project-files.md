# 09: Project files: save and load as TOML

**What to build:** An agent that runs each script in a new process saves its Project to a TOML file once and loads it in later scripts, instead of redefining it every time. A typo in the file fails loudly with the valid keys and the closest match, so a misspelled key can't quietly switch masking off.

Read these sections of `.scratch/first-look/spec.md`: Project, data policy and masking, and test group 7 under Testing Decisions. Covers user stories 68 and 69.

**Blocked by:** 08-project-and-masking

**Status:** ready-for-agent

- [ ] `Project.load(path)` reads TOML with the standard library. The file has `name` at the top level and a `data_policy` table that mirrors the DataPolicy fields.
- [ ] `project.save(path, overwrite=False)` writes the same format deterministically, without a TOML-writing dependency, and refuses to overwrite an existing file unless `overwrite=True`.
- [ ] Saving and loading round-trips every field, including column names that need quoting or escaping in TOML.
- [ ] An unknown or misspelled key, at the top level or in `data_policy`, raises `ProjectFileError`, a `ValueError`, listing the valid keys and the closest match.
- [ ] A TOML syntax error raises `ProjectFileError` that reports the line.
- [ ] A value that fails validation raises in the package's message format, never as a raw pydantic error.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
