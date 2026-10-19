import inspect
import re

import agent_eda as ae

ANALYSES = ae.list_analyses().analyses
# The statuses some analysis can return; the guide lists these and no others.
STATUSES = [
    "completed",
    "partial",
    "insufficient_data",
    "not_applicable",
    "budget_exceeded",
]
UNREACHABLE = ["missing_prerequisites", "blocked_by_policy", "failed"]
# The order for a first look at a new dataset, which names all 11 first-look
# analyses; a step can name two.
FIRST_LOOK = [
    ["describe_dataset"],
    ["infer_column_semantics", "validate_column_types"],
    ["profile_missing_values"],
    ["detect_missing_sentinels"],
    ["summarize_numeric", "summarize_categorical"],
    ["find_duplicate_rows", "find_candidate_keys"],
    ["profile_target"],
    ["compute_association_matrix"],
]


def guide_lines(capsys):
    returned = ae.guide()
    assert returned is None
    return capsys.readouterr().out.splitlines()


def test_prints_one_page_and_returns_none(capsys):
    lines = guide_lines(capsys)

    assert 20 <= len(lines) <= 60
    assert max(len(line) for line in lines) <= 120
    assert lines == [line.rstrip() for line in lines]


def test_names_every_registered_analysis_with_its_summary_under_its_module(capsys):
    lines = guide_lines(capsys)

    for info in ANALYSES:
        line = lines.index(analysis_line(info.name, info.summary, info))
        modules = [
            candidate
            for candidate in lines[:line]
            if candidate.startswith("agent_eda.")
        ]
        assert modules[-1] == info.module


def test_reads_each_summary_from_the_first_line_of_the_docstring(capsys):
    lines = guide_lines(capsys)

    for info in ANALYSES:
        doc = inspect.getdoc(getattr(ae, info.name))
        assert doc
        assert analysis_line(info.name, doc.splitlines()[0], info) in lines


def analysis_line(name, summary, info):
    """An analysis's line in the guide, marked when it supports masking."""
    masking = " [supports masking]" if info.supports_masking else ""
    return f"  {name}: {summary}{masking}"


def test_says_what_the_package_is_how_to_call_it_and_how_to_read_results(capsys):
    text = "\n".join(guide_lines(capsys))

    assert text.startswith(
        f"agent-eda {ae.__version__}: deterministic exploratory data"
    )
    assert 'res = ae.summarize_numeric(df, columns=["age", "income"])' in text
    assert "print(res)" in text
    assert "Always print it" in text
    for word in ["res.status", "res.status_reason", "res.findings", "res.limitations"]:
        assert word in text
    assert all(status in text for status in STATUSES)
    assert not any(status in text for status in UNREACHABLE)
    assert "res.to_frame()" in text
    assert "res.to_json()" in text
    for call in ["ae.search_analyses(", "ae.describe_analysis(", "ae.list_analyses()"]:
        assert call in text


def test_names_all_eleven_analyses_in_the_first_look_order(capsys):
    lines = guide_lines(capsys)

    start = lines.index("A first look at a new dataset, in this order:")
    steps = []
    for line in lines[start + 1 :]:
        step = re.fullmatch(r"  (\d+)\. ([a-z_ ]+): .+", line)
        if not step:
            break
        assert int(step[1]) == len(steps) + 1
        steps.append(step[2].split(" and "))
    assert steps == FIRST_LOOK
    assert sum(len(step) for step in steps) == 11


def test_explains_how_to_mask_the_values_of_sensitive_columns(capsys):
    text = " ".join(" ".join(guide_lines(capsys)).split())

    assert "[supports masking] can show raw values" in text
    assert (
        'project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["email"], '
        "mask_sensitive=True))"
    ) in text
    assert "pass project=project" in text
    assert "<masked 1>" in text
    assert (
        "guardrail against accidental exposure, not anonymization or access control"
    ) in text


def test_explains_how_to_keep_a_project_for_later_scripts(capsys):
    text = " ".join(" ".join(guide_lines(capsys)).split())

    assert 'project.save("project.toml")' in text
    assert 'ae.Project.load("project.toml")' in text
    assert "later script" in text
