# Roadmap

The specs in the order they are worked on. The current spec is the first one below with a `ready-for-agent` ticket that isn't finished and whose `Blocked by:` tickets are finished; within it, take the lowest-numbered such ticket. A blocked ticket, such as release's "Prepare 0.1.0", which waits for specs 3 and 4, doesn't hold agents in its spec. Tickets marked `ready-for-human` or `needs-info` are for the maintainer, and agents skip them.

| Order | Spec | What it delivers |
| --- | --- | --- |
| 1 | [agent-workflow](agent-workflow/spec.md) | AGENTS.md without stale state, exact commands and finish checks, a checklist and guard tests for adding an analysis, a README fit for people, agents and PyPI, and a complete glossary. |
| 2 | [release](release/spec.md) | CI that tests every supported platform and version, packaging fit for PyPI, and one tag-started workflow that publishes to PyPI and creates the GitHub release. Its last ticket, the 0.1.0 release, waits for specs 3 and 4. |
| 3 | [correctness](correctness/spec.md) | Every wrong result, crash, hang, leak and nondeterminism the audit found, each fixed once in the shared code that causes it, with the tests that would have caught it. |
| 4 | [agent-experience](agent-experience/spec.md) | What agents read: printed text, records, names, statuses, the result API, discovery, search and the MCP tool definitions. |
| 5 | [deepening](deepening/spec.md) | Internal restructuring with no change in behaviour, so the next 270 analyses are cheap to add. |

Finished: [first-look](first-look/spec.md) and [mcp-server](mcp-server/spec.md).

[audit/findings.md](audit/findings.md) holds the October 2026 audit that specs 1 to 5 cite by finding ID, such as AN2-1; [audit/standards-review.md](audit/standards-review.md) has the full standards review.
