# 09: Release 0.1.0

What to build: the maintainer tags 0.1.0, approves the PyPI deployment, and checks the published release.

Read these sections of `.scratch/release/spec.md`: Verifying that it all runs, and test group 3. This ticket covers user stories 13 to 19.

Blocked by: 08

Status: ready-for-human

- [ ] `git tag -a v0.1.0 -m "agent-eda 0.1.0"` was pushed, and the `pypi` deployment was approved. Every release.yml job is green. The Comments record the run's URL.
- [ ] In a fresh environment, `pip install agent-eda==0.1.0` and `pip install 'agent-eda[mcp]'` both install, and `agent-eda-mcp --help` runs.
- [ ] The PyPI page shows the README correctly, with the GitHub URLs marked verified.
- [ ] The provenance endpoint answers: https://pypi.org/integrity/agent-eda/0.1.0/agent_eda-0.1.0-py3-none-any.whl/provenance
- [ ] `gh release verify v0.1.0` passes, and the GitHub release carries both files and the 0.1.0 notes.

## Comments
