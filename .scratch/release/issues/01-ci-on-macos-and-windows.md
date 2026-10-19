# 01: CI on macOS and Windows

What to build: the latest-versions test job also runs on macOS and Windows, and whatever fails there is fixed in the code or the tests.

Read these sections of `.scratch/release/spec.md`: ci.yml, Verifying that it all runs, and test group 1. This ticket covers user story 1. Findings: REL-6.

Blocked by: none

Status: ready-for-agent

- [x] `tests-latest` has `macos-26` and `windows-2025` cells for at least Python 3.12 and 3.14, with `defaults.run.shell: bash`.
- [x] Every test that writes a text file passes `encoding="utf-8"`. Comparisons that depend on line endings say which they expect.
- [ ] The Windows-only branches run at least once, and the Comments list them: `os.altsep` in `_project_file.py`, and the skips in `tests/test_mcp_project.py` and `tests/test_project_files.py`.
- [ ] Every failure on macOS or Windows is fixed in the code, or in the test when the test was wrong. The Comments say which, and why.
- [ ] The run on a pull request is green on every cell. The Comments record the run's URL.
- [x] `tests/test_ci_workflow.py` checks that the macOS and Windows cells exist.
- [x] The finish checks pass.

## Comments

- Not finished, so `Status:` stays `ready-for-agent`: three boxes need a GitHub run that the implementer's worktree could not make (no push, no pull request). The maintainer or the next session must push `ticket/release-01`, open a pull request, run `gh pr checks --watch`, fix what the macOS and Windows cells flag, then record the run's URL here and tick the three boxes.
- Matrix: `tests-latest` gets `macos-26` (ARM64) and `windows-2025` (x86_64) cells for Python 3.12 and 3.14, `defaults.run.shell: bash` on the job, and an `os` key that the job name shows. `test_contributing_names_the_architectures_the_workflow_runs` now counts Linux ARM64 cells only, and CONTRIBUTING.md names the macOS and Windows cells.
- Encoding: every `read_text`, `write_text` and text-mode `open` in `tests/` passes `encoding="utf-8"`, and `tests/test_suite_rules.py` fails when one doesn't. No test compares line endings: Windows `write_text` writes `\r\n`, and `read_text` and pandas read it back as `\n`, so nothing needed a `newline=` argument. The doctests in `Project.load` and `Project.save` use ASCII text and were left alone.
- Local run: the whole suite, ruff, ty, both minimum-versions runs and the MCP stdio smoke test pass on macOS (Darwin, ARM64). The benchmark was not run: no analysis changed.
- Fixed by reading, before any Windows run (tests were wrong, not the code): `test_a_path_starting_with_a_tilde_means_the_home_directory` also sets `USERPROFILE`, which `os.path.expanduser` reads on Windows, and `test_a_path_that_looks_like_a_url_is_a_local_path` expects `os.sep` after the working directory. More Windows failures are likely in tests that format paths into messages (`tests/test_project_files.py`, `tests/test_mcp_tables.py`, `tests/test_mcp_data.py`); the first Windows run decides.
- Windows-only branches still to run on GitHub and list here with their result: `os.altsep` in `_project_file.py` (the `separators` tuple, only non-empty on Windows), the `sys.platform == "win32"` skips in `tests/test_mcp_project.py` (line 195) and `tests/test_project_files.py` (line 1005), which skip on Windows by design.
