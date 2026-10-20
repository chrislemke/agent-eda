# 04: Release workflow, changelog and setup wizard

What to build:
- `release.yml`, which a version tag starts: full CI, then build, provenance, publishing to PyPI or TestPyPI, and the GitHub release;
- `CHANGELOG.md`, whose sections become the release notes;
- the wizard script for the maintainer's one-time setup;
- ADR 0007, recording the design.

Read these sections of `.scratch/release/spec.md`: release.yml, CHANGELOG.md, The maintainer's one-time setup, and test group 1. This ticket covers user stories 13 to 21.

Blocked by: 03

Status: done

- [x] `.github/workflows/release.yml` has the five jobs in the spec's `needs:` order, with exactly the permissions it lists. It routes tags as the spec's table says, and pins every action to the SHAs it gives, or newer ones with the same pinning.
- [x] The build job fails when the tag differs from `v` followed by `uv version --short`. The Comments show the check failing on a mismatched tag in a local dry run of the script step.
- [x] `ci.yml`'s benchmark condition reads `inputs.full`, and release.yml calls ci.yml with `full: true`.
- [x] `CHANGELOG.md` exists with an `## [Unreleased]` section. The github-release job takes its notes from the tag's section, or from Unreleased for a pre-release tag, and fails a final tag that has no section of its own.
- [x] AGENTS.md says that a ticket whose change a caller can see adds a line under `## [Unreleased]`.
- [x] `tests/test_ci_workflow.py` checks that:
  - the file is named `release.yml`;
  - the publish job's permissions are exactly `id-token: write`;
  - the environments are `pypi` and `testpypi`;
  - `ci` is called with `full: true`;
  - every `uses:` in every workflow is a 40-character SHA;
  - both workflows build with `--force-pep517`.
- [x] actionlint and zizmor (auditor persona) are clean on both workflows. Any finding accepted deliberately is recorded in the Comments with its reason.
- [x] `scripts/release-setup-wizard.sh`, generated with the `mattpocock-skills:wizard` skill, walks through every step of the spec's one-time setup, checking what it can with read-only `gh api` calls. `bash -n` passes on it.
- [x] ADR 0007 records the release design: one workflow, why not two (with the two primary-source facts), trusted publishing, the tag routing and the approval gate.
- [x] CONTRIBUTING.md has a "Releasing" section with the release steps.
- [x] The finish checks pass.

## Comments

Done ahead of its blockers, at the maintainer's request (2026-10-20): tickets 02 and 03 are not done. This ticket changed only what a release needs of `ci.yml`: it can be called (`workflow_call` with the boolean input `full`), a tag push no longer starts it (`push` on `branches: ["**"]`), its benchmark condition reads `inputs.full`, and it has the concurrency group that 02 specifies (zizmor's auditor persona flags a workflow without one, and the ticket asks for both workflows to be clean). Still open in 02 and 03: the schedule trigger, `tests-newest`, `tests-agentcore`, the pre-commit file checks, `dependabot.yml`, the packaging metadata, and the `packaging` and `packaging-mcp` jobs. The release's build job already runs `twine check --strict`, `check-wheel-contents` and the install smoke tests on both files itself, so 03 adds the same checks to every pull request, not to the release.

Settled here, where the spec left it open or the two texts differ:
- **Release notes come from the build job, not the github-release job.** The spec says a final tag without its own section "fails the build job", the ticket says the github-release job fails it. Failing in github-release would happen after the upload, which leaves a version on PyPI with no GitHub release and no way to retry the upload. So the build job takes the section (or Unreleased for a pre-release), fails before anything is built for upload when a final tag has no section or an empty one, and uploads it as the artifact `release-notes`; github-release only reads that artifact.
- **The `index` output is the environment name.** The build job outputs `version`, `index` (`pypi` or `testpypi`), `repository_url` and `prerelease`; the publish job's environment is `needs.build.outputs.index`. A `.post` version is a final release. Any other form (`.dev`) fails the build job, since the spec's table has no route for it.
- **The README no longer lists the ADRs.** The new ADR's link put the README 120 characters over the 60,000 it may have (`test_an_agent_reads_the_whole_readme_in_one_go`), and every later ADR would again. The maintainer said the README needs no ADR information, so the Documentation section lost its "Architecture decisions" bullet, and `test_the_documentation_section_links_to_every_guide_for_contributors` no longer requires a link per ADR. The README is 59,277 characters. The one inline pointer in the masking text ("ADR 0001 records why") stays.
- **No `uv.lock` change, and no new dependency.** The tools the build job runs are pinned: `twine@7.0.0` and `check-wheel-contents@0.6.3` through `uvx`.

Checks:
- Tests: `tests/test_ci_workflow.py` has 32 tests. Besides the structure checks the ticket lists (file name, publish permissions exactly `id-token: write`, environments, `ci` called with `full: true`, SHA pins, `--force-pep517`), it runs the build job's `route` and `notes` steps as bash against a stub `uv` and a fixture `CHANGELOG.md`: the five kinds of version go to their index, a mismatched tag and a `.dev` version stop the build, a final tag without a section and an empty section stop it, a pre-release takes Unreleased, and the repository's own changelog gives a release candidate its notes. Changing the release-candidate route or the tag check in a copy of the workflow turned three of them red. The step tests skip on Windows, where the steps never run.
- Dry run of the tag check with the real `uv version --short` in this repository (version 0.1.0): `TAG=v0.2.0` exits 1 with "The tag is v0.2.0, but pyproject.toml says version 0.1.0. Delete the tag, set the version with 'uv version <new>', commit, and tag that commit as v<new>." `TAG=v0.1.0` exits 0 with `index=pypi` and `prerelease=false`; the notes step then exits 1 for 0.1.0, since `CHANGELOG.md` has no `## [0.1.0]` section yet, which is the intended stop until the 0.1.0 ticket moves the lines.
- actionlint 1.7.12 and zizmor 1.31.1 (`--persona auditor`) are clean on both workflows, with one finding accepted on purpose: zizmor's `self-repository` suggests writing `uses: $/.github/workflows/ci.yml`, which actionlint rejects, so `release.yml` keeps `./.github/workflows/ci.yml` with an inline `# zizmor: ignore[self-repository]`. zizmor ran offline, without its online audits, because those need a GitHub token and zizmor is a tool fetched from PyPI; instead, each action pin was resolved with `gh api repos/<action>/commits/<tag>` and equals the SHA the spec lists. shellcheck is clean on every `run:` script of `release.yml` and on the wizard.
- Built locally with `uv build --force-pep517` (uv 0.13.1): `twine check --strict` and `check-wheel-contents` pass on both files, and the wheel and the source distribution each install into a fresh Python 3.12 environment with their dependencies and run `tests/smoke.py` (11 analyses, 90 examples, 0 failed).
- The wizard: `bash -n` and shellcheck pass, and the checks it makes ran against this repository with read-only calls: they report the real state (no environments, no tag ruleset, actions not restricted, release immutability already on). The jq expressions of the passing paths ran against sample JSON shaped like GitHub's API. Its template library is unchanged. It needs `bash` 3.2 or later, so it also runs on macOS's own bash.

Not run locally:
- Finish checks, run on 2026-10-20 with uv 0.13.1 (the machine's own uv is 0.11): ruff, ruff format and ty clean; `pytest` 3,966 passed on the latest versions; the stdio MCP smoke test; 3,410 passed on the minimum versions without the extra and 3,966 with it. The benchmark wasn't run: no analysis changed.
- `release.yml` itself. Only a tag push starts it, so its first run on GitHub is the `0.1.0rc1` rehearsal (ticket 06). The `provenance`, `publish` and `github-release` jobs, `gh release create` on a repository with release immutability, and the trusted-publisher exchange have therefore never run. The `ci.yml` edits run on the pull request of this change.
- The wizard end to end: it opens a browser and waits for input.
- zizmor's online audits.

For the maintainer (tickets 05 and 06):
- The ruleset `main` requires status checks that a pull request never produces: `benchmark, latest versions (Python 3.12, ARM64, 2 CPUs)` (the benchmark runs on dispatch and in releases only), `tests, AgentCore versions (Python 3.12, ARM64)` (ticket 02 has not built it), and three names an earlier `ci.yml` used (`tests, latest versions (Python 3.12, ARM64)`, `... (Python 3.12, x86_64)` and `... (Python 3.13, x86_64)`). Until the list matches the job names, no pull request can satisfy it; admins can bypass the ruleset.
- `Require deployments to succeed` in that ruleset can't name `pypi` or `testpypi`: they allow deployments only from `v*` tags, after the merge, so no pull request could ever have a successful deployment and every merge would be blocked. ADR 0007 says so.
