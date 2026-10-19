"""The README shows what agent-eda does, in examples that run as written."""

import contextlib
import importlib.util
import inspect
import io
import json
import re
import typing
from pathlib import Path

import pandas as pd
import yaml

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PLANNED = ROOT / "docs" / "planned-analyses.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
GENERATOR = ROOT / "scripts" / "generate_reference.py"
COMMAND = "uv run python scripts/generate_reference.py"
START = "<!-- reference:start -->"
END = "<!-- reference:end -->"
BLOB = "https://github.com/chrislemke/agent-eda/blob/main/"
# The audit measured the old README, 97,999 characters, at 35.9K tokens, so 2.7
# characters per token, and an agent's Read showed only its first 631 lines. Read's
# cap of about 25K tokens is about 68K characters; 60K keeps a margin.
READ_LIMIT = 60_000
SECTIONS = [
    "Features",
    "Installation",
    "Quick start",
    "Analyses",
    "Discovery",
    "Reading a result",
    "Projects and masking",
    "Use over MCP",
    "Reference",
    "Documentation",
    "License",
]


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


def test_the_readme_has_the_sections_of_a_user_guide_in_order():
    headings = re.findall(r"^## (.+)$", README.read_text(), flags=re.MULTILINE)

    assert headings == SECTIONS


def test_an_agent_reads_the_whole_readme_in_one_go():
    assert len(README.read_text()) <= READ_LIMIT


def test_each_analysis_has_one_row_in_the_first_look_table_or_the_other_table():
    text = README.read_text()
    analyses = text[text.index("## Analyses") : text.index("## Discovery")]
    first_look, other = analyses.split("### Other analyses")
    # First-look rows: | 1 | `describe_dataset` | What does ...? | none |
    # Other rows: | `data.missingness` | `analysis_name` | What does ...? | none |
    listed = re.findall(r"^\| \d+ \| `(\w+)` \|", first_look, flags=re.MULTILINE)
    in_first_look = len(listed)
    listed += re.findall(r"^\| `[\w.]+` \| `(\w+)` \|", other, flags=re.MULTILINE)
    registered = sorted(info.name for info in ae.list_analyses().analyses)

    assert sorted(set(listed)) == registered, "Every analysis needs a row."
    assert len(listed) == len(set(listed)), "No analysis goes in both tables."
    if in_first_look == len(listed):
        assert "No analysis outside the first look ships yet." in other


def test_lists_each_planned_function_once_and_no_shipped_analysis_as_planned():
    # Rows such as: | `estimate_quantiles` | `statistics.descriptive` | Estimate ... |
    names = re.findall(
        r"^\| `(\w+)` \| `[\w.]+` \|", PLANNED.read_text(), flags=re.MULTILINE
    )
    shipped = {info.name for info in ae.list_analyses().analyses}

    assert names
    assert len(names) == len(set(names))
    assert [name for name in names if name in shipped] == []


def test_the_readme_and_agents_md_point_to_the_planned_analyses_file():
    readme = README.read_text()
    agents = (ROOT / "AGENTS.md").read_text()

    assert f"({BLOB}docs/planned-analyses.md)" in readme
    assert "docs/planned-analyses.md" in agents
    assert "Planned analyses" not in agents.replace("docs/planned-analyses.md", "")
    assert "Other analyses" in agents


def test_planned_analyses_stay_within_the_adrs_and_the_glossary():
    text = PLANNED.read_text()
    rows = {
        match[1]: line
        for line in text.splitlines()
        if (match := re.match(r"^\| `(\w+)` \|", line))
    }
    words = re.compile(r"config|split|\bS3\b|remote|artifact", flags=re.IGNORECASE)

    assert words.findall(text) == []
    # Names such as test_normality stay: they run hypothesis tests. A check of
    # declared relationships runs none.
    assert "test_functional_dependencies" not in rows
    assert "check_functional_dependencies" in rows
    assert "local" in rows["load_dataset"]
    for name in ("run_analysis", "run_standard_eda"):
        assert not re.search(r"binding|context|reproducibility", rows[name]), name
    assert "ADR 0004" in text


def test_no_link_in_the_readme_is_relative_and_each_repository_link_leads_to_a_file():
    links = re.findall(r"\]\(([^)\s]+)\)", README.read_text())

    assert links
    for link in links:
        if link.startswith("#"):
            continue
        assert link.startswith("https://"), f"{link} is relative; use an absolute URL."
        if link.startswith("https://github.com/chrislemke/agent-eda/"):
            assert link.startswith(BLOB), link
            assert (ROOT / link.removeprefix(BLOB)).is_file(), link


def test_the_documentation_section_links_to_every_guide_for_contributors():
    text = README.read_text()
    documentation = text[text.index("## Documentation") : text.index("## License")]
    files = ["CONTRIBUTING.md", "AGENTS.md", "GLOSSARY.md", "docs/planned-analyses.md"]
    files += [f"docs/adr/{p.name}" for p in sorted((ROOT / "docs/adr").glob("*.md"))]

    for name in files:
        assert f"({BLOB}{name})" in documentation, name


def test_contributing_holds_the_development_notes_and_leaves_the_commands_to_agents_md():
    text = CONTRIBUTING.read_text()

    assert f"({BLOB}AGENTS.md)" in text or "(AGENTS.md)" in text
    assert "lowest-direct" not in text
    assert "| Task | Command |" not in text
    assert "## Development" not in README.read_text()
    assert f"({BLOB}CONTRIBUTING.md)" in README.read_text()
    for notes in ("Property tests", "Benchmark", "tests/recipes.py", "pre-commit"):
        assert notes in text, notes


def test_contributing_names_the_architectures_the_workflow_runs():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    include = workflow["jobs"]["tests-latest"]["strategy"]["matrix"]["include"]
    arm64 = sorted({cell["python"] for cell in include if cell["arch"] == "ARM64"})
    text = " ".join(CONTRIBUTING.read_text().split())

    assert arm64 == ["3.12"]
    assert "Python 3.12 on Linux ARM64" in text
    assert "ARM64" not in text.replace("Python 3.12 on Linux ARM64", "")


def test_the_readme_describes_leading_zeros_and_columns_as_the_analyses_treat_them():
    text = README.read_text()
    codes = pd.DataFrame({"zip": ["01234", "00123", "02345", "01111", "01010"]})
    res = ae.infer_column_semantics(codes)
    lacking = sorted(
        info.name
        for info in ae.list_analyses().analyses
        if "columns" not in info.parameters
    )
    except_for = re.search(
        r"takes `columns=` .*?except ((?:`\w+`(?:, | and )?)+);", text
    )

    # No flag or finding is about leading zeros: the numbers count as text.
    assert res.columns["zip"].inferred_logical_type == "text"
    assert "leading_zeros" not in str(res.columns["zip"].flags) + str(res.findings)
    assert not re.search(r"flags[^.]*leading zeros", text)
    assert except_for
    assert sorted(re.findall(r"`(\w+)`", except_for[1])) == lacking


def test_the_readme_gives_a_list_cell_example_that_a_table_has():
    keys = ae.find_candidate_keys(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}))
    cells = keys.to_frame()["columns"]
    text = " ".join(README.read_text().split())

    assert all(isinstance(cell, list) for cell in cells)
    assert "such as the columns of a candidate key from `find_candidate_keys`" in text
    assert "row IDs of a duplicate group" not in text


def test_the_readme_words_sentinels_tools_and_determinism_as_the_glossary_and_code_do():
    text = README.read_text()
    prose = re.sub(r"^```.*?^```$", "", text, flags=re.MULTILINE | re.DOTALL)

    assert "placeholder codes" not in text
    assert "down to the byte" not in text
    assert "in the same environment, BLAS thread count included" in text
    assert re.findall(r"(?<!MCP )\btools?\b", prose) == []


def test_the_reference_is_the_one_the_generator_writes_from_the_docstrings():
    text = README.read_text()

    assert START in text
    assert END in text
    assert generator().update(text) == text, (
        f"The README's Reference is stale. Run `{COMMAND}` and commit the result."
    )


def test_running_the_generator_twice_in_a_row_changes_nothing(tmp_path):
    copy = tmp_path / "README.md"
    stale = re.sub(
        f"{START}.*{END}", f"{START}\nold\n{END}", README.read_text(), flags=re.S
    )
    copy.write_text(stale)

    generator().main(["--readme", str(copy)])
    first = copy.read_text()
    generator().main(["--readme", str(copy)])

    assert first != stale
    assert first == README.read_text()
    assert copy.read_text() == first


def test_the_generator_check_names_the_command_to_run_when_the_reference_is_stale(
    tmp_path, capsys
):
    copy = tmp_path / "README.md"
    copy.write_text(README.read_text().replace("**Statuses:**", "**Status:**", 1))

    assert generator().main(["--check", "--readme", str(copy)]) == 1
    assert COMMAND in capsys.readouterr().err
    assert generator().main(["--check", "--readme", str(README)]) == 0


def test_the_reference_gives_each_analysis_signature_summary_and_statuses():
    entries = reference_entries()

    for info in ae.list_analyses().analyses:
        entry = entries[info.name]
        description = ae.describe_analysis(info.name)
        signature = inspect.signature(getattr(ae, info.name))
        # The signature, as `ae.name(data, *, columns=None, top_n=10) -> Result`.
        parts = ["data", "*"]
        for parameter in list(signature.parameters.values())[1:]:
            if parameter.default is parameter.empty:
                parts.append(parameter.name)
            else:
                parts.append(f"{parameter.name}={parameter.default!r}")
        call = f"ae.{info.name}({', '.join(parts)}) -> {description.result_class}"
        statuses = re.search(r"^\*\*Statuses:\*\* (.+)$", entry, flags=re.MULTILINE)

        assert f"\n`{call}`\n" in entry, info.name
        assert f"\n{description.summary}\n" in entry, info.name
        assert statuses, info.name
        assert re.findall(r"`(\w+)`", statuses[1]) == info.statuses, info.name


def test_the_reference_gives_each_argument_type_default_and_meaning():
    entries = reference_entries()

    for info in ae.list_analyses().analyses:
        description = ae.describe_analysis(info.name)
        signature = inspect.signature(getattr(ae, info.name))
        # Rows such as: | `top_n` | `int` | `10` | How many ... |
        rows = re.findall(
            r"^\| `(\w+)` \| `(.+?)` \| (.+?) \| (.+) \|$",
            entries[info.name],
            flags=re.MULTILINE,
        )

        assert [row[0] for row in rows] == info.parameters[1:], info.name
        for (name, kind, default, meaning), parameter in zip(
            rows, description.parameters[1:], strict=True
        ):
            declared = signature.parameters[name].default
            required = declared is inspect.Parameter.empty
            assert kind.replace("\\|", "|") == parameter.type, (info.name, name)
            assert default == ("required" if required else f"`{declared!r}`")
            assert parameter.description.startswith(meaning.rstrip("…")), name


def test_the_reference_gives_each_finding_code_with_its_assessment():
    entries = reference_entries()

    for info in ae.list_analyses().analyses:
        description = ae.describe_analysis(info.name)
        # Lines such as: - `infinite_values` (suspected): a column holds inf.
        listed = re.findall(
            r"^- `(\w+)` \((\w+)\): (.+)$", entries[info.name], flags=re.MULTILINE
        )
        # A description starts with its assessment: "suspected; a column holds ...".
        expected = [
            (code.code, *(code.description or "").split("; ", 1))
            for code in description.finding_codes
        ]

        assert [row[:2] for row in listed] == [row[:2] for row in expected], info.name
        for (_, _, claim), (_, _, meaning) in zip(listed, expected, strict=True):
            assert meaning.startswith(claim.rstrip("…")), info.name


def test_the_reference_gives_the_fields_to_read_of_each_analysis_result():
    entries = reference_entries()

    for info in ae.list_analyses().analyses:
        description = ae.describe_analysis(info.name)
        line = re.search(
            r"^\*\*Fields:\*\* (.+)$", entries[info.name], flags=re.MULTILINE
        )

        assert line, info.name
        for field in description.result_fields:
            assert f"`{field.name}`" in line[1], (info.name, field.name)


def test_the_reference_covers_the_discovery_functions_the_classes_and_the_result_base():
    entries = reference_entries()
    discovery = ["guide", "search_analyses", "describe_analysis", "list_analyses"]

    for name in [*discovery, "Project", "DataPolicy", "ColumnSpec", "LogicalType"]:
        assert name in entries, name
    assert "`ae.guide() -> None`" in entries["guide"]
    assert "`ae.search_analyses(query, limit=5)" in entries["search_analyses"]
    assert "`ae.Project(*, name=None, data_policy=None)`" in entries["Project"]
    assert (
        "`ae.DataPolicy(*, sensitive_columns=(), mask_sensitive=False)`"
        in (entries["DataPolicy"])
    )
    for kind in typing.get_args(ae.LogicalType):
        assert f"`'{kind}'`" in entries["LogicalType"]
    for field in ae.AnalysisResult.model_fields:
        assert f"| `{field}` |" in entries["AnalysisResult"], field
    for method in ("to_frame", "to_dict", "to_json", "record_schema"):
        assert method in entries["AnalysisResult"], method


def test_the_reference_lists_every_error_code_with_the_class_that_raises_it():
    text = README.read_text()
    errors = text[text.index("### Errors") : text.index("### MCP server command")]
    # Rows such as: | `column_not_found` | `ColumnNotFoundError` | ... |
    rows = re.findall(r"^\| `(\w+)` \| `(\w+)` \|", errors, flags=re.MULTILINE)
    docstring = ae.errors.__doc__ or ""
    documented = re.findall(r"^    (\w+) \((\w+)", docstring, flags=re.MULTILINE)

    assert documented
    assert rows == documented


def test_the_reference_gives_the_mcp_server_command_and_its_option():
    text = README.read_text()
    command = text[text.index("### MCP server command") : text.index(END)]

    assert "`agent-eda-mcp [--project PATH]`" in command
    assert "`--project PATH`" in command
    assert "(#use-over-mcp)" in command


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


def generator():
    """The generator script as a module, loaded from its path since it isn't shipped."""
    spec = importlib.util.spec_from_file_location("generate_reference", GENERATOR)
    assert spec, f"{GENERATOR} is missing."
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_entries():
    """The Reference section's entries by public name, from its #### headings."""
    text = README.read_text()
    reference = text[text.index(START) : text.index(END)]
    parts = re.split(r"^#### `(\w+)`\n", reference, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def fenced_blocks():
    """The README's fenced code blocks in order, as (language, code) pairs."""
    fence = re.compile(r"^```(\w*)\n(.*?)^```$", flags=re.MULTILINE | re.DOTALL)
    return fence.findall(README.read_text())
