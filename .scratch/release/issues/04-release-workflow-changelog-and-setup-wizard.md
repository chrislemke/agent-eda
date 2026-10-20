# 04: Release workflow, changelog and setup wizard

What to build:
- `release.yml`, which a version tag starts: full CI, then build, provenance, publishing to PyPI or TestPyPI, and the GitHub release;
- `CHANGELOG.md`, whose sections become the release notes;
- the wizard script for the maintainer's one-time setup;
- ADR 0007, recording the design.

Read these sections of `.scratch/release/spec.md`: release.yml, CHANGELOG.md, The maintainer's one-time setup, and test group 1. This ticket covers user stories 13 to 21.

Blocked by: 03

Status: ready-for-agent

- [ ] `.github/workflows/release.yml` has the five jobs in the spec's `needs:` order, with exactly the permissions it lists. It routes tags as the spec's table says, and pins every action to the SHAs it gives, or newer ones with the same pinning.
- [ ] The build job fails when the tag differs from `v` followed by `uv version --short`. The Comments show the check failing on a mismatched tag in a local dry run of the script step.
- [ ] `ci.yml`'s benchmark condition reads `inputs.full`, and release.yml calls ci.yml with `full: true`.
- [ ] `CHANGELOG.md` exists with an `## [Unreleased]` section. The github-release job takes its notes from the tag's section, or from Unreleased for a pre-release tag, and fails a final tag that has no section of its own.
- [ ] AGENTS.md says that a ticket whose change a caller can see adds a line under `## [Unreleased]`.
- [ ] `tests/test_ci_workflow.py` checks that:
  - the file is named `release.yml`;
  - the publish job's permissions are exactly `id-token: write`;
  - the environments are `pypi` and `testpypi`;
  - `ci` is called with `full: true`;
  - every `uses:` in every workflow is a 40-character SHA;
  - both workflows build with `--force-pep517`.
- [ ] actionlint and zizmor (auditor persona) are clean on both workflows. Any finding accepted deliberately is recorded in the Comments with its reason.
- [ ] `scripts/release-setup-wizard.sh`, generated with the `mattpocock-skills:wizard` skill, walks through every step of the spec's one-time setup, checking what it can with read-only `gh api` calls. `bash -n` passes on it.
- [ ] ADR 0007 records the release design: one workflow, why not two (with the two primary-source facts), trusted publishing, the tag routing and the approval gate.
- [ ] CONTRIBUTING.md has a "Releasing" section with the release steps.
- [ ] The finish checks pass.

## Comments
