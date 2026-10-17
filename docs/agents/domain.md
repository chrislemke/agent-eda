# Domain docs

How the engineering skills should use this repo's domain documentation.
This repo is single-context: one `GLOSSARY.md` and one `docs/adr/` at the root.

## Before exploring, read these

- `GLOSSARY.md` at the repo root.
- `docs/adr/`: the ADRs that touch the area you're about to work in.

If either is missing, proceed silently. `/domain-modeling` creates them when terms or decisions get resolved.

## Use the glossary's vocabulary

When your output names a domain concept (an issue title, a proposal, a test name), use the term as `GLOSSARY.md` defines it, not a synonym it lists under _Avoid_. If the concept isn't in the glossary yet, either you're inventing language the project doesn't use, or there's a real gap to note for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an ADR, say so explicitly instead of silently overriding it:

> _Contradicts ADR-0004 (plain functions with explicit arguments), but worth reopening because…_
