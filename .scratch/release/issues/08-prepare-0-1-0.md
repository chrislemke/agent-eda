# 08: Prepare 0.1.0

What to build: everything 0.1.0 needs in the repository before the maintainer tags it:
- the README written for an installed package;
- the changelog section for 0.1.0;
- the version set to 0.1.0.

Read these sections of `.scratch/release/spec.md`: CHANGELOG.md, and Verifying that it all runs. This ticket covers user stories 19 and 20. Findings: REL-4.

Blocked by: every ticket of the correctness spec, every ticket of the agent-experience spec, and 06

Status: ready-for-agent

- [ ] The README's Installation section and its MCP install lines say `pip install agent-eda` and `pip install 'agent-eda[mcp]'`. The wheel-from-a-clone route stays only as the offline AgentCore path. No sentence says the package isn't on PyPI yet.
- [ ] `CHANGELOG.md` turns `## [Unreleased]` into `## [0.1.0] - <date>`. It opens with a short summary of what 0.1.0 offers, followed by the collected lines, and leaves an empty `## [Unreleased]` above it.
- [ ] `uv version 0.1.0` set the version, and `uv.lock` is current.
- [ ] The README, rendered with PyPI's renderer (`uvx --from 'readme-renderer[md]' python -m readme_renderer README.md`), shows no broken links or markup. The Comments record the check.
- [ ] The finish checks pass, and the dispatched full CI on main is green. The Comments record its URL.

## Comments
