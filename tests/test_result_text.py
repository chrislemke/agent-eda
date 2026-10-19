import textwrap

import numpy as np
import pandas as pd
import pytest

import agent_eda as ae
from agent_eda.results import Finding, Limitation, StatusReason


def test_text_follows_the_fixed_template():
    df = pd.DataFrame(
        {"age": [34, 51, 29, 51], "score": [0.5, None, 0.75, 0.5], "flag": [1] * 4},
        index=pd.Index([10, 20, 30, 40]),
    )

    res = ae.describe_dataset(df, source="survey.csv")

    assert str(res) == "\n".join(
        [
            "describe_dataset: completed on 4 rows x 3 columns",
            "Memory: 128 bytes in total. Index: int64, unique, 0 missing labels.",
            "Provenance: source 'survey.csv'; source_version not supplied.",
            "Columns (3 of 3 shown):",
            "column  dtype    non_missing_count  missing_count  distinct_count"
            "  memory_bytes",
            "age     int64                    4              0               3"
            "            32",
            "score   float64                  3              1               2"
            "            32",
            "flag    int64                    4              0               1"
            "            32",
            "Findings (1 of 1 shown):",
            "- suspected constant_columns: 1 column holds a single distinct "
            "non-missing value: 'flag'.",
            "Limitations: none",
            "More: res.to_frame() has all 3 columns; res.findings, res.limitations "
            "and res.to_json() have the rest.",
        ]
    )


def test_a_long_status_reason_wraps_instead_of_losing_its_end():
    # How to fix the call comes last in a reason, so cutting it would hide it.
    res = ae.describe_dataset(pd.DataFrame({"x": [1, 2]}))
    message = "The run needs more work than its limit allows. " * 3 + "Raise the limit."
    failed = type(res).model_validate(
        {
            **res.model_dump(),
            "status": "failed",
            "status_reason": {"code": "too_much_work", "message": message},
        }
    )

    lines = str(failed).splitlines()

    assert lines[1:3] == [
        "Reason: too_much_work: The run needs more work than its limit allows. The run "
        "needs more work than its limit allows. The",
        "    run needs more work than its limit allows. Raise the limit.",
    ]
    assert max(len(line) for line in lines) <= 120


def test_footer_names_a_single_column_in_the_singular():
    text = str(ae.describe_dataset(pd.DataFrame({"x": [1, 2]})))

    assert text.splitlines()[-1].startswith("More: res.to_frame() has the 1 column;")


def test_repr_gives_the_same_text_as_str():
    res = ae.describe_dataset(pd.DataFrame({"x": [1, 2]}))

    assert repr(res) == str(res)


def test_ipython_and_jupyter_display_the_same_plain_text_and_no_html():
    from IPython.core.formatters import DisplayFormatter

    res = ae.describe_dataset(pd.DataFrame({"x": [1, 2]}))

    shown, _ = DisplayFormatter().format(res)

    assert shown == {"text/plain": str(res)}


def test_text_shows_10_columns_and_counts_the_rest():
    df = pd.DataFrame({f"col_{i}": range(3) for i in range(40)})

    text = str(ae.describe_dataset(df))

    lines = text.splitlines()
    assert "Columns (10 of 40 shown):" in lines
    assert "col_9" in text
    assert "col_10" not in text
    assert lines[-1].startswith("More: res.to_frame() has all 40 columns")


def test_text_shows_5_findings_and_5_limitations_and_counts_the_rest():
    res = ae.describe_dataset(pd.DataFrame({"x": [1, 2]}))
    findings = [
        Finding(
            code=f"code_{i}",
            claim=f"Claim {i}.",
            assessment="descriptive",
            columns=[],
            fields=[],
        )
        for i in range(8)
    ]
    limitations = [
        Limitation(code="unhashable_values", reason=f"Reason {i}.", fields=[])
        for i in range(7)
    ]

    text = str(
        res.model_copy(update={"findings": findings, "limitations": limitations})
    )

    lines = text.splitlines()
    assert "Findings (5 of 8 shown):" in lines
    assert "- descriptive code_4: Claim 4." in lines
    assert "code_5" not in text
    assert "Limitations (5 of 7 shown):" in lines
    assert "- unhashable_values: Reason 4." in lines
    assert "Reason 5" not in text


def test_counts_use_thousands_separators_and_sizes_4_significant_digits():
    df = pd.DataFrame({"x": range(1_234_567)}, index=np.arange(1_234_567))

    text = str(ae.describe_dataset(df))

    assert text.splitlines()[0] == (
        "describe_dataset: completed on 1,234,567 rows x 1 column"
    )
    assert "Memory: 18.84 MiB (19,753,072 bytes) in total." in text
    assert "1,234,567" in text.splitlines()[5]


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"name\nwith\nnewlines" + chr(0x2028) + "and separators": [1, 2]}),
        pd.DataFrame({"x" * 500: [1, 2], "y" * 300: [None, None]}),
        pd.DataFrame(
            {f"{'long_' * 30}{i}": [None] * 3 for i in range(60)},
            index=pd.Index(["a", "a", "a"]),
        ),
        pd.DataFrame(
            [[1, 2, 3]],
            columns=pd.Index(["dup\r" * 40, "dup\r" * 40, "z"], dtype=object),
        ),
    ],
    ids=["control-characters", "long-labels", "wide-and-many", "duplicated-long"],
)
def test_text_stays_within_40_lines_of_120_characters(df):
    text = str(ae.describe_dataset(df, source="s3://" + "b" * 300))

    lines = text.splitlines()
    assert len(lines) <= 40
    assert max(len(line) for line in lines) <= 120
    # No other line separator, such as \r or U+2028, hides in a line.
    assert "\n".join(lines) == text


def reason_of(line_count):
    """A status reason that prints on exactly `line_count` lines."""
    words = []
    while True:
        words.append("raise_the_limit")
        text = f"Reason: too_much_work: {' '.join(words)}"
        wrapped = textwrap.wrap(text, width=120, subsequent_indent="    ")
        if len(wrapped) == line_count:
            return " ".join(words)
        assert len(wrapped) < line_count, "a word added more than one line"


def full_sections(reason_lines):
    """A result with a full table, more findings and limitations than the text
    shows and a reason on `reason_lines` lines, and its text's length without
    the cut at 40 lines."""
    res = ae.describe_dataset(pd.DataFrame({f"col_{i}": range(3) for i in range(12)}))
    findings = [
        Finding(
            code=f"code_{i}",
            claim=f"Claim {i}.",
            assessment="descriptive",
            columns=[],
            fields=[],
        )
        for i in range(8)
    ]
    limitations = [
        Limitation(code="unhashable_values", reason=f"Reason {i}.", fields=[])
        for i in range(7)
    ]
    reason = StatusReason(code="too_much_work", message=reason_of(reason_lines))
    filled = res.model_copy(
        update={
            "status": "failed",
            "status_reason": reason,
            "findings": findings,
            "limitations": limitations,
        }
    )
    # Without a reason, findings and limitations the text has the heading, 2
    # facts, the table (title, header, 10 rows), "Findings: none", "Limitations:
    # none" and a footer. A reason adds its lines, 5 findings and 5 limitations
    # add a line each next to the "none" lines.
    uncut = len(str(res).splitlines()) + reason_lines + 5 + 5
    return filled, uncut


@pytest.mark.parametrize("reason_lines", [10, 11, 12])
def test_a_text_that_fits_in_40_lines_is_not_cut(reason_lines):
    res, uncut = full_sections(reason_lines)

    lines = str(res).splitlines()

    assert len(lines) == uncut
    assert len(lines) <= 40
    assert not any("more lines" in line for line in lines)


@pytest.mark.parametrize("reason_lines", [13, 14, 30])
def test_a_text_past_40_lines_ends_on_a_line_that_counts_what_it_leaves_out(
    reason_lines,
):
    res, uncut = full_sections(reason_lines)

    lines = str(res).splitlines()

    assert uncut > 40
    assert len(lines) == 40
    # 39 lines stay, and the last says how many it replaces.
    assert lines[-1] == f"... {uncut - 39} more lines; see res.to_json()."
    assert lines[0] == "describe_dataset: failed on 3 rows x 12 columns"
    assert "Columns (10 of 12 shown):" in lines
