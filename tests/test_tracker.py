"""The tracker under .scratch/ holds the same header lines in every ticket, and the roadmap lists real specs."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"
TRIAGE_LABELS = ROOT / "docs" / "agents" / "triage-labels.md"

# A header line is plain ("Status: done") or bold ("**Status:** done", the form
# of the finished tickets).
STATUS_LINE = re.compile(r"^\**Status:\**[ \t]*(\S.*?)\s*$", re.MULTILINE)
BLOCKED_BY_LINE = re.compile(r"^\**Blocked by:\**[ \t]*\S", re.MULTILINE)
ROADMAP_SPEC_LINK = re.compile(r"\]\(([^)\s]+/spec\.md)\)")


def test_every_ticket_has_a_status_label_and_a_blocked_by_line():
    problems = []
    for ticket in sorted(SCRATCH.glob("*/issues/*.md")):
        problems += ticket_problems(ticket.read_text(), ticket.relative_to(ROOT))

    assert problems == []


def test_every_spec_has_a_status_label():
    problems = []
    for spec in sorted(SCRATCH.glob("*/spec.md")):
        if spec.parent.name == "audit":
            continue
        problems += status_problems(spec.read_text(), spec.relative_to(ROOT))

    assert problems == []


def test_every_spec_in_the_roadmap_exists():
    roadmap = (SCRATCH / "ROADMAP.md").read_text()
    links = ROADMAP_SPEC_LINK.findall(roadmap)

    missing = [link for link in links if not (SCRATCH / link).is_file()]

    assert links, "ROADMAP.md links no spec"
    assert missing == [], (
        f"ROADMAP.md lists specs that do not exist under .scratch/: {missing}. "
        "Fix the link or remove the row."
    )


def test_a_ticket_without_a_status_line_is_reported():
    problems = ticket_problems("# 01: Broken\n\nBlocked by: none\n", Path("t.md"))

    assert len(problems) == 1
    assert "Status:" in problems[0]


def test_a_ticket_with_an_unknown_status_label_is_reported():
    text = "# 01: Broken\n\nBlocked by: none\n\nStatus: finished\n"

    problems = ticket_problems(text, Path("t.md"))

    assert len(problems) == 1
    assert "finished" in problems[0]
    assert "done" in problems[0]


def test_a_ticket_without_a_blocked_by_line_is_reported():
    problems = ticket_problems("# 01: Broken\n\nStatus: done\n", Path("t.md"))

    assert len(problems) == 1
    assert "Blocked by:" in problems[0]


def test_the_triage_labels_include_done():
    assert "done" in labels()


def labels():
    """The label strings in the second column of triage-labels.md's table."""
    return set(re.findall(r"^\|[^|]*\|\s*`([^`]+)`", TRIAGE_LABELS.read_text(), re.M))


def status_problems(text, name):
    statuses = STATUS_LINE.findall(text)
    if not statuses:
        return [
            f"{name}: no 'Status:' line. Add one with a label from {sorted(labels())}."
        ]
    if statuses[0] not in labels():
        return [
            f"{name}: status '{statuses[0]}' is not a label in docs/agents/"
            f"triage-labels.md. Use one of {sorted(labels())}."
        ]
    return []


def ticket_problems(text, name):
    problems = status_problems(text, name)
    if not BLOCKED_BY_LINE.search(text):
        problems.append(
            f"{name}: no 'Blocked by:' line. Add 'Blocked by: none' or the tickets it waits for."
        )
    return problems


@pytest.mark.parametrize("label", ["done", "ready-for-agent"])
def test_the_label_check_accepts_a_documented_label(label):
    assert status_problems(f"Status: {label}\n", Path("t.md")) == []


def test_the_finished_specs_and_all_their_tickets_say_done():
    files = [
        path
        for feature in ("first-look", "mcp-server")
        for path in [
            SCRATCH / feature / "spec.md",
            *sorted((SCRATCH / feature / "issues").glob("*.md")),
        ]
    ]

    not_done = [
        str(path.relative_to(ROOT))
        for path in files
        if STATUS_LINE.findall(path.read_text())[:1] != ["done"]
    ]

    assert len(files) == 27
    assert not_done == []


@pytest.mark.parametrize(
    "stale",
    [
        "ticket 01 creates it",
        "ticket 03 adds it",
        "taken in number order",
        "`first-look` and `mcp-server` are finished",
    ],
)
def test_agents_md_holds_no_stale_state(stale):
    assert stale not in (ROOT / "AGENTS.md").read_text()


def test_agents_md_names_no_spec_directory():
    section = (ROOT / "AGENTS.md").read_text().split("## Working on a ticket")[1]
    section = section.split("\n## ")[0]

    named = [
        directory.name
        for directory in SCRATCH.iterdir()
        if directory.is_dir() and f"`{directory.name}`" in section
    ]

    assert named == []


def test_agents_md_says_when_to_ask_the_maintainer_and_which_tickets_to_skip():
    text = (ROOT / "AGENTS.md").read_text()

    for phrase in (
        "ready-for-human",
        "needs-info",
        "Status: done",
        "ask the maintainer",
    ):
        assert phrase in text
