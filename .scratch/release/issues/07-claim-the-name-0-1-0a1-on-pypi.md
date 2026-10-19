# 07: Claim the name: 0.1.0a1 on PyPI

What to build: an alpha on PyPI, so that nobody else can register `agent-eda` before 0.1.0. A pending trusted publisher doesn't reserve a name.

Read these sections of `.scratch/release/spec.md`: Tag routing under release.yml, and the last Further Notes item. This ticket covers user story 13.

Blocked by: 06

Status: needs-info

The maintainer must confirm this ticket before it becomes `ready-for-human`. While `0.1.0a1` is the only version on PyPI, `pip install agent-eda` and `uv add agent-eda` install it, because installers fall back to a pre-release when no final release exists. That alpha carries the bugs the correctness spec fixes. The alternatives are to skip the claim and accept the small risk, or to claim the name later, once correctness is done.

- [ ] The maintainer confirmed in the Comments that the claim should go ahead, given that installs pick up the alpha.
- [ ] The README's opening says that 0.1.0a1 is an early alpha, and points to the changelog.
- [ ] `v0.1.0a1` was tagged and pushed, the `pypi` deployment was approved, and https://pypi.org/p/agent-eda shows 0.1.0a1. The Comments record the run's URL.
- [ ] The PyPI page shows the GitHub URLs as verified.

## Comments
