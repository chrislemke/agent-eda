# 01: Roadmap and state-free AGENTS.md

What to build: AGENTS.md tells an agent how to find the next ticket without naming a spec, finished tickets say `Status: done`, and the tracker docs describe the rules this repository actually uses.

Read these sections of `.scratch/agent-workflow/spec.md`: Roadmap and tickets, and test group 1. This ticket covers user stories 1 to 5. Findings: DOC-1, DOC-9, DOC-11, DOC-16 (the template sentences), SPEC-18.

Blocked by: none

Status: ready-for-agent

- [ ] AGENTS.md's "Working on a ticket" names no spec. It says that the current spec is the first spec in `.scratch/ROADMAP.md` with a `ready-for-agent` ticket that isn't done and whose `Blocked by:` tickets are all done. It takes the lowest-numbered such ticket, skips tickets marked `ready-for-human` or `needs-info`, and asks the maintainer which spec comes next when none is left.
- [ ] Finishing a ticket sets `Status: done`. `docs/agents/triage-labels.md` lists `done`.
- [ ] `docs/agents/issue-tracker.md` describes the plain `Status:` and `Blocked by:` lines in the forms `none`, `01, 03` and a spec named in words. It has a "Closing a ticket" section that matches AGENTS.md. Its Wayfinding section says it applies only to `/wayfinder`.
- [ ] The two setup-template sentences that do nothing are gone: the one in `triage-labels.md` that says to edit the label column, and the one in `domain.md` that says to proceed silently.
- [ ] The 25 tickets and 2 specs of first-look and mcp-server say `Status: done`. Each box that SPEC-18 lists as superseded gets one line under that ticket's Comments, naming the decision or commit that replaced it.
- [ ] mcp-server ticket 08's open point says it was settled in ADR 0006. The mcp-server spec's mention of "the catalogue" points to the planned analyses.
- [ ] These stale lines are gone from AGENTS.md: "(ticket 01 creates it)", "(ticket 03 adds it)", "taken in number order" and the second `.scratch` pointer under Agent skills (DOC-11).
- [ ] A test fails when a file under `.scratch/*/issues/` lacks a `Status:` line with a label from `triage-labels.md`, or lacks a `Blocked by:` line. It also fails when ROADMAP.md lists a spec that doesn't exist. Check it once against a broken ticket file.
- [ ] The finish checks pass: ruff, ty and pytest on the latest versions, both minimum-versions runs and the MCP smoke test.

## Comments
