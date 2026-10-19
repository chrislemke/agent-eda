"""What the printed text shows: wrapping, order and wording.

These tests assert structure, such as order and completeness, not whole texts;
tests/test_result_text.py holds the goldens, one per template.
"""

import re

import pandas as pd
import pytest
from recipes import recipe_for

import agent_eda as ae
from agent_eda.results import Finding, Limitation

MAX_WIDTH = 120


def _words(text: str) -> str:
    """The text with every run of whitespace as one space."""
    return " ".join(text.split())


def _with(res, findings=None, limitations=None):
    update = {}
    if findings is not None:
        update["findings"] = findings
    if limitations is not None:
        update["limitations"] = limitations
    return res.model_copy(update=update)


def _finding(code, assessment, claim="Claim."):
    return Finding(code=code, claim=claim, assessment=assessment, columns=[], fields=[])


def test_a_long_limitation_wraps_with_a_hanging_indent_and_loses_nothing():
    df = pd.DataFrame({"a": [1, -999, 3, 4], "b": ["x", "y", "Y ", "y"]})
    res = ae.profile_missing_values(df)
    reason = next(
        lim.reason
        for lim in res.limitations
        if lim.code == "sentinel_values_not_declared"
    )

    lines = str(res).splitlines()

    start = next(i for i, line in enumerate(lines) if line.startswith("- sentinel"))
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        block.append(line)
    assert len(block) > 1
    assert all(line.startswith("  ") for line in block[1:])
    assert (
        _words(" ".join(block)) == f"- sentinel_values_not_declared: {_words(reason)}"
    )
    assert not any(line.endswith(("...", "…")) for line in lines)


def test_long_findings_and_facts_wrap_and_every_line_fits_the_width():
    df = pd.DataFrame({f"column_number_{i:03d}": [1, 1, 2] for i in range(40)})
    res = ae.describe_dataset(df)
    claim = " ".join(["word"] * 60) + " end."

    text = str(_with(res, findings=[_finding("one_code", "suspected", claim)]))

    lines = text.splitlines()
    assert max(len(line) for line in lines) <= MAX_WIDTH
    assert _words(text).count("word") >= 60
    assert "end." in text


def test_the_40_line_cap_holds_and_counts_what_it_leaves_out():
    res = ae.describe_dataset(pd.DataFrame({"x": [1, 2]}))
    claim = " ".join(["word"] * 80)
    findings = [_finding(f"code_{i}", "suspected", claim) for i in range(5)]
    limitations = [
        Limitation(code="unhashable_values", reason=claim, fields=[]) for _ in range(5)
    ]

    text = str(_with(res, findings=findings, limitations=limitations))

    lines = text.splitlines()
    assert len(lines) == 40
    assert max(len(line) for line in lines) <= MAX_WIDTH
    assert re.fullmatch(r"\.\.\. [\d,]+ more lines; see res\.to_json\(\)\.", lines[-1])


def test_findings_print_in_order_of_assessment():
    res = ae.describe_dataset(pd.DataFrame({"x": [1, 2]}))
    order = ["descriptive", "unknown", "passed", "suspected", "violation"]
    findings = [_finding(f"{a}_code", a) for a in order]

    text = str(_with(res, findings=findings))

    printed = [
        line.split()[1]
        for line in text.splitlines()
        if line.startswith("- ") and line.endswith("Claim.")
    ]
    assert printed == ["violation", "suspected", "descriptive", "passed", "unknown"]


def test_assessment_order_keeps_the_analysis_order_within_an_assessment():
    res = ae.describe_dataset(pd.DataFrame({"x": [1, 2]}))
    findings = [
        _finding("b_code", "suspected"),
        _finding("a_code", "violation"),
        _finding("a_suspected", "suspected"),
    ]

    lines = str(_with(res, findings=findings)).splitlines()

    codes = [line.split()[2].rstrip(":") for line in lines if line.startswith("- ")]
    assert codes == ["a_code", "b_code", "a_suspected"]


def _table_columns(text: str, title: str = "Columns") -> list[str]:
    """The first cell of each row of the printed table with this title."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{title} ("))
    rows = []
    for line in lines[start + 2 :]:
        if not line or line.startswith(("Findings", "Limitations", "More")):
            break
        rows.append(line.split()[0])
    return rows


def _wide_frame() -> pd.DataFrame:
    frame = {f"c{i:03d}": [float(j + i) for j in range(6)] for i in range(120)}
    for name in ("c050", "c077", "c101", "c119"):
        frame[name] = [-9999.0] * 6
    return pd.DataFrame(frame)


def test_columns_with_a_finding_come_first_in_summarize_numeric_s_table():
    res = ae.summarize_numeric(_wide_frame())

    shown = _table_columns(str(res))

    assert shown[:4] == ["c050", "c077", "c101", "c119"]
    assert shown[4:] == [f"c{i:03d}" for i in range(6)]


def test_columns_with_a_finding_come_first_in_describe_dataset_s_table():
    res = ae.describe_dataset(_wide_frame())

    shown = _table_columns(str(res))

    assert {"c050", "c077", "c101", "c119"} <= set(shown)
    assert shown[:4] == ["c050", "c077", "c101", "c119"]


def test_columns_with_a_limitation_come_first_in_the_table():
    df = pd.DataFrame(
        {f"c{i:02d}": [float(j * j + i) for j in range(10)] for i in range(15)}
    )
    df["c13"] = [1.0, 2.0] + [None] * 8

    res = ae.summarize_numeric(df)

    limited = {
        path.split("'")[1]
        for lim in res.limitations
        for path in lim.fields
        if path.startswith("columns['")
    }
    assert limited == {"c13"}
    assert _table_columns(str(res))[0] == "c13"


def test_columns_with_a_finding_come_first_in_infer_column_semantics_table():
    df = pd.DataFrame({f"c{i:02d}": [1.5, 2.5, 3.5, 4.5] for i in range(14)})
    df["c12"] = [1, 1, 1, 1]
    df["c13"] = ["2024-01-01"] * 4

    shown = _table_columns(str(ae.infer_column_semantics(df)))

    assert shown[:2] == ["c12", "c13"]


def _variants_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "answer": ["Yes", "yes ", "YES", "no", "no", "Yes"],
            "plan": ["basic", "pro", "basic", "pro", "team", "basic"],
        }
    )


def test_a_case_variant_finding_names_its_variants():
    res = ae.summarize_categorical(_variants_frame())

    claim = next(f.claim for f in res.findings if f.code == "case_or_space_variants")

    assert "'answer'" in claim
    for label in ("'Yes'", "'yes '", "'YES'"):
        assert label in claim
    assert "'no'" not in claim
    assert _words(claim) in _words(str(res))


def test_a_case_variant_finding_shows_placeholders_for_masked_columns():
    project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["answer"]))

    res = ae.summarize_categorical(
        _variants_frame(), project=project, mask_sensitive=True
    )

    text = str(res)
    claim = next(f.claim for f in res.findings if f.code == "case_or_space_variants")
    assert "<masked 1>" in claim
    for value in ("Yes", "YES", "yes"):
        assert value not in text


def test_summarize_categorical_prints_each_columns_top_levels_on_their_own_lines():
    res = ae.summarize_categorical(_variants_frame(), top_n=2)

    lines = str(res).splitlines()

    start = lines.index("Top levels per column:")
    assert lines[start + 1].startswith("  answer: ")
    assert "'Yes'" in lines[start + 1]
    assert lines[start + 2].startswith("  plan: ")
    assert "'basic': 3 (share 0.5)" in lines[start + 2]
    assert "'team'" not in lines[start + 2]


def test_the_top_levels_never_push_findings_out_of_the_40_lines():
    df = pd.DataFrame(
        {f"c{i:02d}": [f"value {i} number {j}" for j in range(30)] for i in range(12)}
    )
    df.loc[0, "c11"] = "VALUE 11 NUMBER 1"

    text = str(ae.summarize_categorical(df, top_n=10))

    lines = text.splitlines()
    assert len(lines) <= 40
    assert any(line.startswith("- suspected case_or_space_variants") for line in lines)
    assert lines[-1].startswith("    ") or lines[-1].startswith("More:")
    assert any("more not shown" in line for line in lines)


def _messy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, None, 4.0, -999.0, 6.0, 7.0, 8.0],
            "cat": ["a", "A", "b", None, "b", "N/A", "a", "b"],
            "target": [0, 1, 0, 1, 0, None, 1, 0],
            "when": ["2024-01-01", "2024-02-01"] * 4,
        }
    )


@pytest.mark.parametrize("name", [a.name for a in ae.list_analyses().analyses])
def test_no_analysis_prints_a_percentage(name):
    arguments = recipe_for(name).required(_messy_frame())
    if arguments is None:
        pytest.skip("the frame allows no valid call")

    text = str(getattr(ae, name)(_messy_frame(), **arguments))

    assert "%" not in text


def test_shares_print_as_fractions_under_their_field_names():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": [1, 2, 3, None]})

    text = str(ae.profile_missing_values(df))

    assert "missing_share 0.25" in text
    assert "complete_row_share 0.5" in text


def test_the_trimmed_mean_fact_names_the_trim_fraction_argument():
    text = str(ae.summarize_numeric(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})))

    assert "trim_fraction 0.1" in text


def test_left_out_lines_say_why_each_column_was_left_out():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "flag": [True, False, True],
            "name": ["a", "b", "c"],
            "when": pd.to_datetime(["2024-01-01"] * 3),
        }
    )

    lines = str(ae.summarize_numeric(df)).splitlines()

    left_out = {
        name: line
        for line in lines
        if line.startswith("Left out")
        for name in ("flag", "name", "when")
        if f"'{name}'" in line
    }
    assert set(left_out) == {"flag", "name", "when"}
    assert "boolean" in left_out["flag"]
    assert "not numeric" not in left_out["flag"]
    # pandas 3 stores text as str, pandas 2.2 as object.
    assert "not numeric (text)" in left_out["name"] or "(object)" in left_out["name"]
    assert "datetime" in left_out["when"]
    assert "'flag'" not in left_out["name"]


def test_summarize_categorical_says_why_a_numeric_column_was_left_out():
    df = pd.DataFrame({"x": [1.0, 2.0], "name": ["a", "b"]})

    lines = str(ae.summarize_categorical(df)).splitlines()

    [line] = [line for line in lines if line.startswith("Left out")]
    assert "'x'" in line
    assert "numeric" in line


def test_find_duplicate_rows_gives_no_finding_when_nothing_repeats():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    res = ae.find_duplicate_rows(df)

    assert res.findings == []
    assert res.duplicate_group_count == 0
    assert "Findings: none" in str(res)
    assert "Compared 2 columns: 'a' and 'b'; no two rows are equal in them;" in str(res)
    assert "no_duplicate_rows" not in str(ae.describe_analysis("find_duplicate_rows"))


def test_the_schema_fact_accounts_for_every_declared_column():
    df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None], "b": ["x"] * 3})

    text = str(
        ae.infer_column_semantics(
            df,
            declared_schema={
                "a": "integer",
                "empty": "integer",
                "b": "integer",
                "ghost": "text",
            },
        )
    )

    [line] = _words(text).split("declared_schema:")[1:2]
    assert line.startswith(
        " 4 columns declared: 1 matching (ties with the top count), 1 not, "
        "1 not inferred, 1 not in the data."
    )


def test_an_empty_table_says_it_has_no_rows_in_the_footer():
    for res in (
        ae.find_duplicate_rows(pd.DataFrame({"a": [1, 2]})),
        ae.describe_dataset(pd.DataFrame()),
    ):
        text = _words(str(res))

        assert "res.to_frame() has no rows;" in text
        assert "all 0" not in text


def test_a_violation_is_never_cut_behind_suspected_findings():
    # Seven columns with two declarations: the flags fill the findings' 5 lines.
    df = pd.DataFrame(
        {
            "id": range(1, 21),
            "const": [1] * 20,
            "empty": [None] * 20,
            "num_text": [str(i) for i in range(20)],
            "date_text": ["2024-01-01"] * 20,
            "flag": ["yes", "no"] * 10,
            "mixed": [1, "a"] * 10,
        }
    )

    res = ae.infer_column_semantics(
        df, declared_schema={"customer": "integer", "id": "text"}
    )

    text = str(res)
    violation = next(f for f in res.findings if f.assessment == "violation")
    assert f"- violation {violation.code}:" in text
    lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert lines[0].startswith("- violation ")
