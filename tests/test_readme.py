"""The README shows what agent-eda does, in examples that run as written."""

import contextlib
import io
import json
import re
from pathlib import Path

import agent_eda as ae

README = Path(__file__).resolve().parents[1] / "README.md"


def test_the_usage_examples_run_as_written_and_print_what_the_readme_shows(
    tmp_path, monkeypatch
):
    # The examples run in order in one namespace, as when pasted into one
    # session, from an empty directory, since one of them saves a project file.
    monkeypatch.chdir(tmp_path)
    blocks = fenced_blocks()
    examples = [i for i, (language, _) in enumerate(blocks) if language == "python"]
    namespace = {"__name__": "__main__"}

    assert examples, "The README has no python example."
    for i in examples:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(blocks[i][1], f"README.md, block {i}", "exec"), namespace)
        # A text block right after an example shows exactly what it prints, and
        # a toml block the project file it saved.
        if i + 1 < len(blocks) and blocks[i + 1][0] == "text":
            assert printed.getvalue() == blocks[i + 1][1]
        if i + 1 < len(blocks) and blocks[i + 1][0] == "toml":
            assert Path("project.toml").read_text() == blocks[i + 1][1]


def test_the_first_look_table_follows_the_guide_with_each_analysis_question(capsys):
    ae.guide()
    lines = capsys.readouterr().out.splitlines()
    summaries = {info.name: info.summary for info in ae.list_analyses().analyses}
    start = lines.index("A first look at a new dataset, in this order:")
    expected = []
    for line in lines[start + 1 :]:
        step = re.fullmatch(r"  (\d+)\. ([a-z_ ]+): .+", line)
        if not step:
            break
        expected += [(step[1], n, summaries[n]) for n in step[2].split(" and ")]

    # Rows such as: | 1 | `describe_dataset` | What does this dataset contain? | none |
    row = re.compile(r"^\| (\d+) \| `(\w+)` \| ([^|]+?) \|", flags=re.MULTILINE)
    assert row.findall(README.read_text()) == expected


def test_lists_each_planned_function_once_and_no_shipped_analysis_as_planned():
    text = README.read_text()
    planned = text[text.index("## Planned analyses") : text.index("## Development")]
    # Rows such as: | `estimate_quantiles` | `statistics.descriptive` | Estimate ... |
    names = re.findall(r"^\| `(\w+)` \| `[\w.]+` \|", planned, flags=re.MULTILINE)
    shipped = {info.name for info in ae.list_analyses().analyses}

    assert names
    assert len(names) == len(set(names))
    assert [name for name in names if name in shipped] == []


def test_calls_masking_a_guardrail_not_anonymization_or_access_control():
    text = " ".join(README.read_text().split())

    assert (
        "Masking is a guardrail against accidental exposure, not anonymization or "
        "access control."
    ) in text


def test_the_mcp_client_configuration_is_valid_json_that_starts_agent_eda_mcp():
    configs = [
        json.loads(code) for language, code in fenced_blocks() if language == "json"
    ]

    assert [config["mcpServers"]["agent-eda"]["command"] for config in configs] == [
        "agent-eda-mcp"
    ]


def fenced_blocks():
    """The README's fenced code blocks in order, as (language, code) pairs."""
    fence = re.compile(r"^```(\w*)\n(.*?)^```$", flags=re.MULTILINE | re.DOTALL)
    return fence.findall(README.read_text())
