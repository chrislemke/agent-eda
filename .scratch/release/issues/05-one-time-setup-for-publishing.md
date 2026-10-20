# 05: One-time setup for publishing

What to build: the maintainer registers the trusted publishers on PyPI and TestPyPI and sets up the GitHub environments, the tag ruleset and release immutability, by running the wizard script from ticket 04.

Read these sections of `.scratch/release/spec.md`: The maintainer's one-time setup. This ticket covers user stories 16 and 21. Findings: REL-12.

Blocked by: 04

Status: ready-for-human

- [ ] `bash scripts/release-setup-wizard.sh` was run to the end.
- [ ] PyPI has a pending publisher for project `agent-eda`, owner `chrislemke`, repository `agent-eda`, workflow `release.yml`, environment `pypi`.
- [ ] TestPyPI has the same pending publisher, with environment `testpypi`.
- [ ] The GitHub environment `pypi` requires the maintainer's approval and allows only `v*` tags. The environment `testpypi` allows only `v*` tags.
- [ ] A tag ruleset protects `v*`, and release immutability is on.
- [ ] The Comments record the date of the setup, and anything the wizard couldn't check.

## Comments
