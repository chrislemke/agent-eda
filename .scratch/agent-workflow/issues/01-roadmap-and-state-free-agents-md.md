# 01: Roadmap and state-free AGENTS.md

What to build: AGENTS.md tells an agent how to find the next ticket without naming a spec, finished tickets say `Status: done`, and the tracker docs describe the rules this repository actually uses.

Read these sections of `.scratch/agent-workflow/spec.md`: Roadmap and tickets, and test group 1. This ticket covers user stories 1 to 5. Findings: DOC-1, DOC-9, DOC-11, DOC-16 (the template sentences), SPEC-18.

Blocked by: none

Status: done

- [x] AGENTS.md's "Working on a ticket" names no spec. It says that the current spec is the first spec in `.scratch/ROADMAP.md` with a `ready-for-agent` ticket that isn't done and whose `Blocked by:` tickets are all done. It takes the lowest-numbered such ticket, skips tickets marked `ready-for-human` or `needs-info`, and asks the maintainer which spec comes next when none is left.
- [x] Finishing a ticket sets `Status: done`. `docs/agents/triage-labels.md` lists `done`.
- [x] `docs/agents/issue-tracker.md` describes the plain `Status:` and `Blocked by:` lines in the forms `none`, `01, 03` and a spec named in words. It has a "Closing a ticket" section that matches AGENTS.md. Its Wayfinding section says it applies only to `/wayfinder`.
- [x] The two setup-template sentences that do nothing are gone: the one in `triage-labels.md` that says to edit the label column, and the one in `domain.md` that says to proceed silently.
- [x] The 25 tickets and 2 specs of first-look and mcp-server say `Status: done`. Each box that SPEC-18 lists as superseded gets one line under that ticket's Comments, naming the decision or commit that replaced it.
- [x] mcp-server ticket 08's open point says it was settled in ADR 0006. The mcp-server spec's mention of "the catalogue" points to the planned analyses.
- [x] These stale lines are gone from AGENTS.md: "(ticket 01 creates it)", "(ticket 03 adds it)", "taken in number order" and the second `.scratch` pointer under Agent skills (DOC-11).
- [x] A test fails when a file under `.scratch/*/issues/` lacks a `Status:` line with a label from `triage-labels.md`, or lacks a `Blocked by:` line. It also fails when ROADMAP.md lists a spec that doesn't exist. Check it once against a broken ticket file.
- [x] The finish checks pass: ruff, ty and pytest on the latest versions, both minimum-versions runs and the MCP smoke test.

## Comments

What this ticket settled that the spec left open:

- **The finished tickets were restored first.** Commit b7db459 ("add: tickets") had deleted `.scratch/first-look/` and `.scratch/mcp-server/` (27 files), although ROADMAP.md links their specs and this ticket edits them. They were restored from 473f77a, then set to `Status: done`. If the deletion was intended, delete the two directories again and drop `test_the_finished_specs_and_all_their_tickets_say_done` and the ROADMAP link.
- **Bold header lines stay.** The finished tickets carry `**Status:**` and `**Blocked by:**` (bold) and one `Blocked by: None (can start immediately)`. Rewriting them is out of scope, so the tracker test accepts an optional bold marker around the label. New tickets use the plain form that `docs/agents/issue-tracker.md` now describes.
- **`tests/test_tracker.py`** checks every `.scratch/*/issues/*.md` for a `Status:` label from `triage-labels.md` and a `Blocked by:` line, every `.scratch/*/spec.md` except audit for a `Status:` label, and that every spec ROADMAP.md links exists. It also checks that the 27 finished files say `done` and that AGENTS.md holds none of the stale phrases, names no spec directory in "Working on a ticket" and mentions `ready-for-human`, `needs-info`, `Status: done` and asking the maintainer.
- **Mutants.** A ticket with `Status: finished` and no `Blocked by:` line made `test_every_ticket_has_a_status_label_and_a_blocked_by_line` fail with both messages; a ROADMAP link to `nope/spec.md` made `test_every_spec_in_the_roadmap_exists` fail naming it. Both were reverted.
- **Superseded boxes** got one Comments line each: first-look 01, 02, 06, 07, 10 to 17 (the catalogue boxes, b430a0d; 17 also box 1, 2e5abfc) and mcp-server 01 box 5, 06 box 3 and 07 box 3 (ea01363). mcp-server ticket 08's open point names ADR 0006, and the mcp-server spec's "catalogue" now says planned analyses.
- **"Taken in number order"** was already gone from AGENTS.md. The duplicate `.scratch` pointer under Agent skills is reduced to "Local markdown. See `docs/agents/issue-tracker.md`", and that section's triage-labels paragraph now names `done`. ROADMAP.md's selection sentence matches AGENTS.md.
- `done` has no mattpocock/skills role, so its row in `triage-labels.md` says `(none)` in the first column.

Checks run: `ruff check`, `ruff format --check`, `ty check` and `pytest` (2171 passed, 2 skipped) on the latest versions, and `uv run --extra mcp python -I tests/smoke_mcp.py`.

Both minimum-versions runs were run by the maintainer outside the sandbox (which cannot reach pypi.org): without the mcp extra 1865 passed, 10 skipped; with it 2171 passed, 2 skipped.
