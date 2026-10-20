# 08: Prepare 0.1.0

What to build: everything 0.1.0 needs in the repository before the maintainer tags it:
- the README written for an installed package;
- the changelog section for 0.1.0;
- the version set to 0.1.0.

Read these sections of `.scratch/release/spec.md`: CHANGELOG.md, and Verifying that it all runs. This ticket covers user stories 19 and 20. Findings: REL-4.

Blocked by: every ticket of the correctness spec, every ticket of the agent-experience spec, and 06

Status: ready-for-agent

- [x] The README's Installation section and its MCP install lines say `pip install agent-eda` and `pip install 'agent-eda[mcp]'`. The wheel-from-a-clone route stays only as the offline AgentCore path. No sentence says the package isn't on PyPI yet.
- [x] `CHANGELOG.md` turns `## [Unreleased]` into `## [0.1.0] - <date>`. It opens with a short summary of what 0.1.0 offers, followed by the collected lines, and leaves an empty `## [Unreleased]` above it.
- [x] `uv version 0.1.0` set the version, and `uv.lock` is current.
- [x] The README, rendered with PyPI's renderer (`uvx --from 'readme-renderer[md]' python -m readme_renderer README.md`), shows no broken links or markup. The Comments record the check.
- [ ] The finish checks pass, and the dispatched full CI on main is green. The Comments record its URL.

## Comments

Done ahead of its blocker, at the maintainer's request (2026-10-20): ticket 06, the rehearsal on TestPyPI, is not done, and neither are 05 (the one-time setup) and 07. Tagging `v0.1.0` stays with the maintainer (ticket 09). The ticket is not `done`: its last box asks for a dispatched full CI on `main`, which can't exist before this change is merged. After the merge, `gh workflow run ci.yml --ref main`, then record its URL here, tick the box and set `Status: done`. The same tree is green on the branch (below).

- The README, rendered with PyPI's renderer (`python -m readme_renderer README.md`, from `readme-renderer[md]`): exit 0, nothing on stderr, 14 code blocks, 21 tables and 56 links, no literal fence or unrendered `](` left over. Every internal anchor (`#reference`, `#results`, `#use-over-mcp`) has a heading, and all 8 external links answer 200 (`docs.astral.sh/uv` and seven files under `github.com/chrislemke/agent-eda/blob/main/`). `twine check --strict`, which validates the README as the long description, also passes in the packaging job.
- Runs on the commit with all of this ticket's code, `6ba8727`, all green: the dispatched full run https://github.com/chrislemke/agent-eda/actions/runs/39420113447 (20 jobs, the benchmark and `tests-newest` included), the pull-request run https://github.com/chrislemke/agent-eda/actions/runs/39420090616 and the push run https://github.com/chrislemke/agent-eda/actions/runs/39420083861. The benchmark on the ARM64 runner: slowest on the latest versions `infer_column_semantics` at 3.28 s, on the minimum versions `describe_dataset` at 3.82 s (4.45 s in ticket 02's run: runner to runner, the figures move by about 15%).
- Finish checks on that tree, locally: ruff, ruff format and ty clean; 3,986 passed on the latest versions; 3,430 (without the extra) and 3,986 (with it) on the minimum versions; the stdio MCP smoke test exits 0; `generate_reference.py --check` passes. The byte baseline changed in one entry per file, `guide [text]`, whose printed text names the version; I refreshed `tests/baseline/pandas3.json` and `pandas2.json` (pandas 2 on the minimum versions) and checked that no other digest moved.

Settled here:
- The CHANGELOG section is dated 2026-10-20, the day of this change. If the maintainer tags on another day, the date in the heading should follow.
- The summary at the top of the `[0.1.0]` section says what the release offers (the 11 first-look analyses, the typed results, discovery, masking, the MCP server, the supported versions, the four required libraries) and that the lines below record what changed during development, so a caller who starts with 0.1.0 can skip them.
- The changelog's own introduction and AGENTS.md's step 5 said "until 0.1.0, the lines collect under Unreleased"; both now say that a release moves the lines into its section.
- The README keeps `uv build --wheel` only in the AgentCore Code Interpreter paragraph, for a sandbox without PyPI, and says the GitHub release of each version carries the wheel too. A test checks that.
- The test that took the notes of a release candidate from the changelog read `0.1.0rc1`, which needs lines under Unreleased; it now takes the version in `pyproject.toml` and does what the release workflow does for its tag (a final version needs its own section). With Unreleased empty and the version still `0.1.0rc1`, it failed, which is the mistake it guards against.

For the maintainer, two things that this ticket's end state does to tickets 06 and 07:
- The version is 0.1.0 now, so a rehearsal tag `v0.1.0rc1` (ticket 06) fails the build job's check of the tag against the package version. Before the rehearsal, set the version back with `uv version 0.1.0rc1` on the commit that gets tagged (as #3 did), and `uv version 0.1.0` again for the release.
- With `## [Unreleased]` empty, as this ticket asks, the build job also stops a pre-release tag at the release notes. I ran the job's notes step against this CHANGELOG.md: `0.1.0` exits 0, and `0.1.0rc1` and `0.1.0a1` each exit 1 with "The CHANGELOG.md section for 0.1.0rc1 is empty" (the step looks for a heading of the tag's own version, then for Unreleased, and never for `[0.1.0]`). For the rehearsal, put a line under `## [Unreleased]` or add a `## [0.1.0rc1]` section first; the same holds for `0.1.0a1` in ticket 07. Letting a pre-release fall back to the notes of its final version would remove the trap, but that is a change to `release.yml` and to the spec's tag routing, so I left it.
