# 01: CI on macOS and Windows

What to build: the latest-versions test job also runs on macOS and Windows, and whatever fails there is fixed in the code or the tests.

Read these sections of `.scratch/release/spec.md`: ci.yml, Verifying that it all runs, and test group 1. This ticket covers user story 1. Findings: REL-6.

Blocked by: none

Status: ready-for-agent

- [ ] `tests-latest` has `macos-26` and `windows-2025` cells for at least Python 3.12 and 3.14, with `defaults.run.shell: bash`.
- [ ] Every test that writes a text file passes `encoding="utf-8"`. Comparisons that depend on line endings say which they expect.
- [ ] The Windows-only branches run at least once, and the Comments list them: `os.altsep` in `_project_file.py`, and the skips in `tests/test_mcp_project.py` and `tests/test_project_files.py`.
- [ ] Every failure on macOS or Windows is fixed in the code, or in the test when the test was wrong. The Comments say which, and why.
- [ ] The run on a pull request is green on every cell. The Comments record the run's URL.
- [ ] `tests/test_ci_workflow.py` checks that the macOS and Windows cells exist.
- [ ] The finish checks pass.

## Comments
