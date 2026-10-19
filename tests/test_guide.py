import importlib
import inspect
import re

from serving import with_stand_ins

import agent_eda as ae

ANALYSES = ae.list_analyses().analyses
STATUSES = [
    "completed",
    "partial",
    "insufficient_data",
    "not_applicable",
    "missing_prerequisites",
    "budget_exceeded",
    "blocked_by_policy",
    "failed",
]
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


def unwrapped(lines):
    """The lines as one text with single spaces, so wrapping doesn't matter."""
    return " ".join(" ".join(lines).split())


def test_prints_one_page_and_returns_none(capsys):
    lines = guide_lines(capsys)

    assert 20 <= len(lines) <= 60
    assert max(len(line) for line in lines) <= 120
    assert lines == [line.rstrip() for line in lines]


def areas():
    """Each area (a subpackage of agent_eda) with how many analyses it holds."""
    counts = {}
    for info in ANALYSES:
        area = info.module.split(".")[1]
        counts[area] = counts.get(area, 0) + 1
    return counts


def test_gives_each_area_with_its_count_and_a_line_from_its_subpackage_docstring(
    capsys,
):
    text = unwrapped(guide_lines(capsys))

    assert areas() == {"data": 8, "statistics": 3}
    for area, count in areas().items():
        subpackage = importlib.import_module(f"agent_eda.{area}")
        covers = inspect.getdoc(subpackage)
        assert covers
        noun = "analysis" if count == 1 else "analyses"
        assert f" {area} ({count} {noun}): {covers.splitlines()[0]}" in text


def test_lists_no_analysis_beyond_the_first_look_and_no_module_headings(capsys):
    lines = guide_lines(capsys)

    assert [line for line in lines if line.startswith("agent_eda.")] == []
    # Each summary is listed by list_analyses, not here.
    assert [info.name for info in ANALYSES if f"  {info.name}: " in lines] == []


def test_points_to_search_and_the_listing_for_the_rest(capsys):
    text = unwrapped(guide_lines(capsys))

    assert 'ae.search_analyses("nulls per column")' in text
    assert "ae.list_analyses()" in text
    assert "ae.describe_analysis(" in text


def test_the_page_stays_the_same_length_with_sixty_analyses(capsys):
    before = guide_lines(capsys)

    with with_stand_ins(60 - len(ANALYSES)):
        assert len(ae.list_analyses().analyses) == 60
        after = guide_lines(capsys)

    assert len(ae.list_analyses().analyses) == len(ANALYSES)
    assert len(before) <= 60
    assert len(after) == len(before)
    assert max(len(line) for line in after) <= 120
    # Only the counts change, in the heading and in the line of each area.
    changed = [line for line in after if line not in before]
    assert len(changed) == 3
    assert changed[0] == "60 analyses in 2 areas:"
    assert all(" analyses): " in line for line in changed[1:])


def test_says_that_read_csv_turns_n_a_and_null_into_missing_values(capsys):
    text = unwrapped(guide_lines(capsys))

    assert (
        "pd.read_csv turns 'N/A', 'NULL' and other default NA strings into missing"
        in text
    )
    assert "pd.read_csv(path, keep_default_na=False)" in text


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
