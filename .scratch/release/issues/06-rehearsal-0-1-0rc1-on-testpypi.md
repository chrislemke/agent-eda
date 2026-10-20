# 06: Rehearsal: 0.1.0rc1 on TestPyPI

What to build: a release candidate goes through the whole release workflow to TestPyPI and a GitHub pre-release, which shows every job runs before anything reaches PyPI.

Read these sections of `.scratch/release/spec.md`: Tag routing under release.yml, Verifying that it all runs, and test group 3. This ticket covers user stories 13 to 19.

Blocked by: 05

Status: ready-for-human

- [ ] The version was set with `uv version 0.1.0rc1` and committed, then `git tag -a v0.1.0rc1 -m "agent-eda 0.1.0rc1"` was pushed together with main.
- [ ] Every release.yml job is green, the full CI with its benchmark included. The Comments record the run's URL.
- [ ] https://test.pypi.org/p/agent-eda shows 0.1.0rc1. `gh release view v0.1.0rc1` shows a pre-release with the wheel and the source distribution.
- [ ] `gh attestation verify <wheel> --repo chrislemke/agent-eda` passes on the downloaded wheel.
- [ ] In an environment that already holds the dependencies, `pip install --no-deps --index-url https://test.pypi.org/simple/ agent-eda==0.1.0rc1` installs, and `python -c "import agent_eda; print(agent_eda.__version__)"` prints 0.1.0rc1.
- [ ] Anything that failed was fixed through a ready-for-agent ticket, then rehearsed again with the next `rcN`. The Comments list each attempt.

## Comments
