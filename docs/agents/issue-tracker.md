# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file
- Triage state is recorded as a plain `Status:` line near the top of each issue file, with a label from `triage-labels.md`
- A plain `Blocked by:` line sits next to it, in one of these forms: `Blocked by: none`, `Blocked by: 01, 03` (tickets of the same spec, by number), or another spec named in words, such as `Blocked by: every ticket of correctness`. A ticket is unblocked when every ticket it names says `Status: done`
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## Closing a ticket

A ticket is finished when every box is ticked and the finish checks in `AGENTS.md` pass. Finishing sets its `Status:` line to `done`, and the commit message is `NN: <ticket title>`. What the ticket settled, any measurement it asked for and any check that couldn't run locally go under `## Comments`. A spec whose tickets are all `done` gets `Status: done` too. `AGENTS.md`, section Working on a ticket, says how an agent finds the next ticket.

## Wayfinding operations

This section applies only to `/wayfinder`. Tickets of the specs under `.scratch/` follow the conventions and Closing a ticket above, not this section. In it, the **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
