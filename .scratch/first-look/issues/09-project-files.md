# 09: Project files: save and load as TOML

**What to build:** An agent that runs each script in a new process saves its Project to a TOML file once and loads it in later scripts, instead of redefining it every time. A typo in the file fails loudly with the valid keys and the closest match, so a misspelled key can't quietly switch masking off.

Read these sections of `.scratch/first-look/spec.md`: Project, data policy and masking, and test group 7 under Testing Decisions. Covers user stories 68 and 69.

**Blocked by:** 08-project-and-masking

**Status:** done

- [x] `Project.load(path)` reads TOML with the standard library. The file has `name` at the top level and a `data_policy` table that mirrors the DataPolicy fields.
- [x] `project.save(path, overwrite=False)` writes the same format deterministically, without a TOML-writing dependency, and refuses to overwrite an existing file unless `overwrite=True`.
- [x] Saving and loading round-trips every field, including column names that need quoting or escaping in TOML.
- [x] An unknown or misspelled key, at the top level or in `data_policy`, raises `ProjectFileError`, a `ValueError`, listing the valid keys and the closest match.
- [x] A TOML syntax error raises `ProjectFileError` that reports the line.
- [x] A value that fails validation raises in the package's message format, never as a raw pydantic error.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments

What this ticket settled that the spec left open:

- **Where things live.** `Project.load(path)` is a classmethod and `project.save(path, *, overwrite=False)` a method of `ae.Project` in `agent_eda.project`. The file format, its checks and its messages are in the private `agent_eda/_project_file.py`, whose docstring describes the format. Tests are in `tests/test_project_files.py`.
- **Format.** A comment line, `# An agent-eda Project. Load it with ae.Project.load(path).`, then `name = "..."`, left out when the name is None since TOML has no null, a blank line and the `[data_policy]` table with both of its keys, even at their defaults. Arrays hold one item per line with a trailing comma, as tomli-w writes them; an empty one is `[]`. Strings are TOML basic strings: quotes, backslashes and control characters get TOML's escapes, and every other character that `str.isprintable()` rejects, such as a zero-width or a no-break space, is written as `\uXXXX`, so that whoever opens the file sees it. Files are UTF-8 with `\n` line endings, and equal projects give identical bytes on the same Python version. `isprintable()` follows each version's Unicode database, so a character assigned after Unicode 15.0 is escaped on 3.12 and written as it is on 3.14; the review counted 5,812 such code points between 3.12 and 3.14.
- **What load requires.** load accepts any TOML spelling of the keys: inline tables, dotted keys, literal strings and comments. Only `name` may be left out. The `[data_policy]` table and both of its keys are required, with `sensitive_columns = []` for none. The ticket says the file "has" that table, and the masking decision is the one value that must never fall back to a default without anyone noticing. A file cut short by a full disk or a killed process, or read while another script writes it, therefore raises instead of loading with masking off. A test cuts a saved file at every character and checks that each shorter file raises. The independent review below had found that 68 of 149 such prefixes loaded quietly with masking off when every key was optional. So save needs no write-then-rename: every partial file is caught when it is loaded.
- **Errors.** `ProjectFileError` (a ValueError) has two codes. `invalid_project_file` covers a file that isn't UTF-8 TOML, lacks the table or one of its keys, or holds a value a field can't take, and text with a lone surrogate given to save. `unknown_project_key` covers a key that isn't a field. The spec names only `ProjectFileError`. Two subclasses cover the file system. `ProjectFileNotFoundError` (`project_file_not_found`, from load) is also a FileNotFoundError, and `ProjectFileExistsError` (`project_file_exists`, from save without overwrite) is also a FileExistsError. The spec's rule that each error inherits the closest built-in asks for that, so `except FileNotFoundError` around a load works; `io.UnsupportedOperation` is the standard library's precedent for an OSError that is also a ValueError. A path through a file, such as `notes.txt/project.toml`, also counts as no file for load.
- **Argument mistakes** raise `InvalidArgumentError` (`invalid_argument`) before any file is touched, each with the fix:
  - a path that isn't a str or an os.PathLike of str;
  - an empty path;
  - a path with a null character;
  - a path that ends in a separator, which would otherwise save a file named after the directory;
  - a path to an existing directory, for which mode `"x"` would otherwise report an existing file and ask for `overwrite=True` wrongly;
  - a save path whose directory doesn't exist (the message shows the `os.makedirs` call; save doesn't create directories, as pandas' `to_csv` doesn't) or is a file;
  - an `overwrite` that isn't a bool (NumPy bools are fine).

  Other OSErrors, such as a missing permission, pass through, as the docstrings say. `overwrite` is keyword-only, so `save(path, True)` raises Python's TypeError, as `ae.guide(1)` does.
- **Values.** load checks the form of each value against how a file writes it, so the message speaks TOML: `name` is a string, `sensitive_columns` an array of strings, a single string being fine, and `mask_sensitive` true or false. `mask_sensitive = "yes"`, `sensitive_columns = true` and `sensitive_columns = [0]` therefore raise, though `DataPolicy` in code names a label that isn't a string, such as `0` or `True`, by its string form. A data file has no such labels, and `sensitive_columns = true` is a mistake that would have loaded as a column named 'True'. `data_policy` given as an array of tables (`[[data_policy]]`) or as a value names its TOML type. Rules across fields, such as `mask_sensitive = true` without sensitive columns, and a column listed twice, go to `Project.model_validate`. load catches the package's own error and raises `ProjectFileError` with "Project.load can't build a project from 'project.toml': " in front of that message, which keeps the field, the value and what the field allows, and with a pointer to `help(ae.Project.load)` after it.
- **Unknown keys.** load checks the keys before building the Project, so the message speaks of the file rather than of `ae.Project(...)`. It names every unknown key at once, each with its level and the closest field, case-insensitive matches first through `_columns.closest_names`, and lists the allowed keys at both levels. A key that belongs to the other level, such as `sensitive_columns` above the `[data_policy]` line or `name` below it, gets "It belongs in [data_policy]" and a sentence on how TOML assigns keys to tables. A misspelled `mask_sensitve` therefore raises instead of leaving masking off.
- **Syntax errors** give tomllib's reason with the line and column, and a hint for "Invalid value" (TOML's quotes, lower-case true and false, no null). An error at the end of the file names the last line that isn't blank and says the file stops in the middle of its last key, value or table line. Messages never quote the file. A data file loaded by mistake, such as a CSV or JSON lines, would otherwise put its first values in front of the agent, which the package avoids as `_registry._value_text` does for arguments. tomllib has no line attribute before Python 3.14, so the line comes from its message, which ends in `(at line L, column C)` or `(at end of document)` on 3.12, 3.13 and 3.14 alike; an unknown format would show the message as it is. Bytes that aren't UTF-8 get the line of the first bad byte. A UTF-8 byte order mark is dropped before parsing, since tomllib rejects it and an editor on Windows may write one.
- **Paths.** A leading `~` means the home directory, as in pandas. save returns the absolute `Path` it wrote, for a later script to load. The messages for a missing or an existing file show the path as given and, when it differs, where it leads, since an agent whose working directory changed between scripts can't otherwise tell why the file is missing. Without overwrite, save opens with mode `"xb"`, which checks and creates in one step.
- **Docs.** `load` and `save` have Google-style docstrings with Args, Returns, Raises and examples that a test runs; `load`'s shows the format. The Project docstring says to save once and load in each later script. `errors.py` lists the four codes, GLOSSARY.md gains "Project file", and the README's status line and the catalogue's DataPolicy entry mention save and load. `_project_file.py` reuses `_registry._value_text` to show a wrong argument, the one import of an underscore name across modules. Renaming it while tickets 11, 13 and 15 change `_registry.py` in parallel risked breaking their merges.
- **Left to ticket 17:** `ae.guide()` doesn't mention project files. It prints 53 lines now and 59 once tickets 11, 13 and 15 land: two first-look steps and four lines in the list of analyses. The guide test caps it at 60, so one more line is better decided together with the README usage that ticket 17 writes.

Review:

- An independent review probed the branch through the public API on Python 3.12 (latest and minimum versions), 3.13 and 3.14. It saved and loaded every one of the 1,112,064 code points that aren't surrogates, in the name and in column names, and every file passed tomllib. It found no failed round trip, no raw pydantic error and no chained traceback. It did find the problems fixed above, each now with a test that fails on the code before the fix:
  - an empty path gave wrong advice and then a bare IsADirectoryError;
  - a path through a file gave a bare NotADirectoryError;
  - `save("newdir/")` wrote a file named `newdir`;
  - a null character in the path gave a bare ValueError;
  - an error at the end of the file named a line past the end, with a hint about values for errors that weren't about values;
  - the line at fault was quoted from a data file;
  - cut-short files loaded with masking off;
  - `sensitive_columns = true` loaded as a column named 'True';
  - value errors described the fix in Python rather than TOML;
  - the docstrings missed that the name can hold a lone surrogate too.

Checks:

- `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, and so does `uv run --isolated --resolution lowest-direct --python 3.12 pytest`, with numpy 1.26.0, pandas 2.2.0 and pydantic 2.7.0. `tests/test_project_files.py` and `tests/test_project.py` also pass with `uv run --isolated --python 3.13` and `--python 3.14`. The round-trip property passed with the `thorough` profile (1,000 examples) on the latest and the minimum versions.
- Not run locally: Linux x86_64 and ARM64, which CI covers.
