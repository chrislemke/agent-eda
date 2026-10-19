"""The checklist for adding an analysis, and what AGENTS.md says about it and the MCP surface."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = (ROOT / "AGENTS.md").read_text()
CHECKLIST_PATH = ROOT / "docs" / "agents" / "new-analysis.md"
# Files that a later ticket of the agent-workflow spec creates (ticket 05).
CREATED_LATER = {"scripts/generate_reference.py"}


def bullet(text, name):
    """The line of AGENTS.md that starts with the bold name."""
    [line] = [line for line in text.splitlines() if line.startswith(f"- **{name}.**")]
    return line


def test_agents_md_points_to_the_checklist_in_one_line():
    line = "Before adding or renaming an analysis, follow docs/agents/new-analysis.md"

    assert AGENTS.count(line) == 1
    assert CHECKLIST_PATH.is_file()


def test_the_checklist_names_files_that_exist_and_a_check_for_each_step():
    text = CHECKLIST_PATH.read_text()
    paths = re.findall(r"`((?:tests|src|docs|scripts)/[\w/.-]+\.(?:py|md))`", text)
    rows = [line for line in text.splitlines() if re.match(r"\| \d+ \|", line)]

    missing = [p for p in set(paths) if p not in CREATED_LATER and not exists(p)]

    assert missing == []
    assert len(rows) >= 12
    assert all(row.count("|") == 4 for row in rows)
    for step in ("@analysis(", "__all__", "_EXPORTS", "TYPE_CHECKING", "recipes.py"):
        assert step in text
    assert "docs/planned-analyses.md" in text
    assert "Other analyses" in text


def exists(path):
    """Whether the path exists, or a file of that name is under src/agent_eda."""
    return (ROOT / path).exists() or (ROOT / "src" / "agent_eda" / path).exists()


def test_agents_md_states_the_mcp_surface_rule():
    rule = bullet(AGENTS, "MCP surface")

    assert "docstring" in rule
    assert "tool description" in rule
    assert "a fix an MCP caller can make" in rule
    assert "JSON Schema" in rule
    assert "<role>_path" in rule
    assert "Nothing an MCP caller reads holds Python" in rule


def test_the_adr_trigger_list_names_adr_0006():
    [line] = [line for line in AGENTS.splitlines() if line.startswith("- `docs/adr/`")]

    assert "ADR 0006" in line
    assert "MCP server" in line
    assert "data reading" in line
    assert "anything an MCP caller reads" in line


def test_the_tests_section_names_the_helpers_and_the_stale_mcp_server():
    tests = AGENTS.split("\n## Tests\n")[1].split("\n## ")[0]

    assert "tests/recipes.py" in tests
    assert "tests/strategies.py" in tests
    assert "data_frames()" in tests
    assert "EDGE_FRAMES" in tests
    assert "when the Claude Code session started" in tests
