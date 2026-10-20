"""The message of every argument mistake, as a Python caller meets it.

Each row of MISTAKES makes one mistake and says the code of the error it raises
and the fragments its message holds. A message says what is wrong, what is
allowed and a fix that works. None shows a type annotation or a pydantic phrase.
tests/test_mcp_argument_messages.py runs the mistakes an MCP caller can make.
"""

import datetime as dt
import pathlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import agent_eda as ae
from agent_eda.errors import AgentEDAError

DF = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 3],
        "age": [31.0, 42.0, 25.0, 25.0],
        "email": ["a@x.org", "b@x.org", "c@x.org", "d@x.org"],
        "target": [0, 1, 0, 1],
    }
)

# What no message may show: a typing annotation, or what pydantic says.
LEAKS = re.compile(
    r"Input should|valid dictionary|valid number|valid string|Sequence\[|Mapping\[|"
    r"Literal\[|\bUnion\[|pydantic|Field required|Extra inputs"
)


# The calls below are wrong on purpose, so they go through an alias that the type
# checker reads as Any.
wrong: Any = ae


def sentinel_length():
    frame = pd.DataFrame(
        {"d": pd.to_timedelta([1, 2], unit="s").astype("timedelta64[s]")}
    )
    return wrong.detect_missing_sentinels(
        frame, sentinel_values={"d": [dt.timedelta(milliseconds=1)]}
    )


# Each mistake: the call that makes it, the code of the error, and fragments the
# message holds. The call is a function, so that building it can't raise.
MISTAKES = {
    # Wrong kinds of data.
    "data-is-a-path-string": (
        lambda: wrong.describe_dataset("churn.csv"),
        "unsupported_data_type",
        ["got a str", "pd.read_csv(path)", "pd.read_parquet(path)"],
    ),
    "data-is-a-path-object": (
        lambda: wrong.describe_dataset(pathlib.Path("churn.csv")),
        "unsupported_data_type",
        ["pd.read_csv(path)", "pd.read_parquet(path)"],
    ),
    "data-is-a-record-batch": (
        lambda: wrong.describe_dataset(pa.RecordBatch.from_pandas(DF)),
        "unsupported_data_type",
        ["got a pyarrow RecordBatch", "batch.to_pandas()"],
    ),
    "data-is-a-table": (
        lambda: wrong.describe_dataset(pa.Table.from_pandas(DF)),
        "unsupported_data_type",
        ["got a pyarrow Table", "table.to_pandas()"],
    ),
    "data-is-none": (
        lambda: wrong.describe_dataset(None),
        "unsupported_data_type",
        ["got None", "holds no data"],
    ),
    "data-is-a-series": (
        lambda: wrong.describe_dataset(DF["age"]),
        "unsupported_data_type",
        ["got a pandas Series", "series.to_frame()"],
    ),
    "data-by-position-and-by-keyword": (
        lambda: wrong.summarize_numeric(DF, data=DF),
        "invalid_call",
        ["got data both by position and by keyword", "pass it once"],
    ),
    "data-without-columns": (
        lambda: wrong.summarize_numeric(DF[[]], columns=["age"]),
        "column_not_found",
        ["unknown column 'age'", "The data has no columns."],
    ),
    # Column arguments.
    "id-column-that-names-the-index": (
        lambda: wrong.find_duplicate_rows(
            DF.set_index("customer_id"), id_column="customer_id"
        ),
        "column_not_found",
        [
            "id_column 'customer_id', which names the index, not a column",
            "index labels are already the row IDs",
            "pass id_column=None",
            "reset_index()",
        ],
    ),
    "columns-holding-none": (
        lambda: wrong.summarize_numeric(DF, columns=[None]),
        "column_not_found",
        [
            "unknown column 'None' in columns",
            "pass columns=None rather than a list holding None",
        ],
    ),
    "sentinel-values-of-the-id-column": (
        lambda: wrong.detect_missing_sentinels(
            DF, sentinel_values={"email": ["n/a"]}, id_column="email"
        ),
        "invalid_argument",
        [
            "sentinel values for 'email'",
            "leaving the columns argument out leaves out the id_column",
            "Name it in columns",
        ],
    ),
    # Arguments an analysis doesn't take.
    "project-for-an-analysis-that-shows-no-raw-values": (
        lambda: wrong.find_candidate_keys(DF, project=wrong.Project()),
        "invalid_call",
        [
            "find_candidate_keys takes no project",
            "shows no raw values",
            "Leave project out",
        ],
    ),
    "mask-sensitive-for-an-analysis-that-shows-no-raw-values": (
        lambda: wrong.summarize_numeric(DF, mask_sensitive=True),
        "invalid_call",
        [
            "summarize_numeric takes no mask_sensitive",
            "shows no raw values",
            "Leave mask_sensitive out",
        ],
    ),
    "argument-misspelled-in-case-only": (
        lambda: wrong.find_duplicate_rows(DF, ID_COLUMN="email"),
        "invalid_call",
        ["has no argument 'ID_COLUMN'", "Did you mean 'id_column'?"],
    ),
    "argument-misspelled": (
        lambda: wrong.profile_target(DF, target="target", task="regression"),
        "invalid_call",
        ["has no argument 'target'", "Did you mean 'target_column'?"],
    ),
    "project-given-as-a-path-string": (
        lambda: wrong.summarize_categorical(DF, project="project.toml"),
        "invalid_argument",
        [
            "summarize_categorical got 'project.toml' for project",
            "a Project, not a path",
            "project=ae.Project.load('project.toml')",
        ],
    ),
    "project-given-as-a-path-object": (
        lambda: wrong.summarize_categorical(DF, project=pathlib.Path("project.toml")),
        "invalid_argument",
        [
            f"got a {type(pathlib.Path()).__name__} for project",
            "project=ae.Project.load('project.toml')",
        ],
    ),
    "project-field-misspelled-in-case-only": (
        lambda: wrong.Project(NAME="x"),
        "invalid_call",
        ["Project has no field 'NAME'", "Did you mean 'name'?"],
    ),
    "column-spec-logical-type-by-position-and-by-keyword": (
        lambda: wrong.ColumnSpec("integer", logical_type="float"),
        "invalid_call",
        ["ColumnSpec got logical_type both by position and by keyword"],
    ),
    "column-spec-field-misspelled": (
        lambda: wrong.ColumnSpec(logical_tpe="integer"),
        "invalid_call",
        ["ColumnSpec has no field 'logical_tpe'", "Did you mean 'logical_type'?"],
    ),
    "column-spec-field-misspelled-beside-another-mistake": (
        lambda: wrong.ColumnSpec(
            logical_type="integer", nullabel=True, datetime_format=3
        ),
        "invalid_call",
        ["ColumnSpec has no field 'nullabel'", "Did you mean 'nullable'?"],
    ),
    "column-spec-datetime-format-without-a-directive": (
        lambda: wrong.ColumnSpec("datetime", datetime_format="dd/mm/yyyy"),
        "invalid_argument",
        [
            "ColumnSpec got datetime_format 'dd/mm/yyyy'",
            "a format for datetime.strptime",
            "It has no directive.",
        ],
    ),
    # Values.
    "set-inside-a-mapping": (
        lambda: wrong.detect_missing_sentinels(DF, sentinel_values={"age": {-1, -999}}),
        "invalid_argument",
        [
            "a set for sentinel_values['age']",
            "no fixed order",
            "a list or tuple for that key instead, such as {'age': sorted(...)}",
        ],
    ),
    "set-for-a-list": (
        lambda: wrong.summarize_numeric(DF, columns={"age"}),
        "invalid_argument",
        ["a set for columns", "columns=sorted(...)"],
    ),
    "object-for-a-number": (
        lambda: wrong.summarize_numeric(DF, trim_fraction=object()),
        "invalid_argument",
        ["got an object for trim_fraction", "a number with 0 <= trim_fraction < 0.5"],
    ),
    "interval-for-a-number": (
        lambda: wrong.summarize_numeric(DF, trim_fraction=pd.Interval(0, 1)),
        "invalid_argument",
        ["got an Interval for trim_fraction"],
    ),
    "nan-twice-in-expected-labels": (
        lambda: wrong.profile_target(
            DF,
            target_column="target",
            task="binary_classification",
            expected_labels=[np.nan, np.nan],
        ),
        "invalid_argument",
        ["nan in expected_labels", "NaN, NaT and infinity can't be class labels"],
    ),
    "task-that-is-a-prefix": (
        lambda: wrong.profile_target(DF, target_column="target", task="binary"),
        "invalid_argument",
        [
            "got 'binary' for task",
            "Did you mean 'binary_classification'?",
        ],
    ),
    "method-misspelled": (
        lambda: wrong.compute_association_matrix(DF, numeric_method="spearmen"),
        "invalid_argument",
        ["got 'spearmen' for numeric_method", "Did you mean 'spearman'?"],
    ),
    "method-in-case-only": (
        lambda: wrong.compute_association_matrix(DF, numeric_method="Pearson"),
        "invalid_argument",
        ["got 'Pearson' for numeric_method", "Did you mean 'pearson'?"],
    ),
    "schema-value-that-is-a-type": (
        lambda: wrong.validate_column_types(DF, schema={"age": int}),
        "invalid_argument",
        ["the type int at schema['age']", "one of 'boolean', 'integer'", "ColumnSpec"],
    ),
    "schema-that-is-a-list": (
        lambda: wrong.validate_column_types(DF, schema=["age"]),
        "invalid_argument",
        [
            "a list of 1 item for schema",
            "a mapping from column names",
            "one of 'boolean', 'integer'",
        ],
    ),
    "quantiles-that-is-a-string": (
        lambda: wrong.summarize_numeric(DF, quantiles="0.5"),
        "invalid_argument",
        ["got '0.5' for quantiles", "quantiles must be", "0 <= value <= 1"],
    ),
    "name-that-is-a-number": (
        lambda: wrong.Project(name=3),
        "invalid_argument",
        ["Project got 3 for name", "name must be a string"],
    ),
    "data-policy-that-is-a-number": (
        lambda: wrong.Project(data_policy=5),
        "invalid_argument",
        ["Project got 5 for data_policy", "a DataPolicy, a dict or None"],
    ),
    "timedelta-finer-than-the-column-unit": (
        sentinel_length,
        "invalid_argument",
        ["'d' holds durations in whole seconds", "can never match"],
    ),
    # Discovery functions and project files.
    "describe-analysis-keyword-misspelled": (
        lambda: wrong.describe_analysis(nme="summarize_numeric"),
        "invalid_call",
        [
            "describe_analysis has no argument 'nme'",
            "Did you mean 'name'?",
            "Allowed: name",
        ],
    ),
    "search-analyses-keyword-misspelled": (
        lambda: wrong.search_analyses(question="nulls"),
        "invalid_call",
        ["search_analyses has no argument 'question'", "Allowed: query, limit."],
    ),
    "search-analyses-keyword-with-a-typo": (
        lambda: wrong.search_analyses(quey="nulls"),
        "invalid_call",
        ["search_analyses has no argument 'quey'", "Did you mean 'query'?"],
    ),
    "project-load-keyword-misspelled": (
        lambda: wrong.Project.load(paht="project.toml"),
        "invalid_call",
        ["Project.load has no argument 'paht'", "Did you mean 'path'?"],
    ),
    "project-save-keyword-misspelled": (
        lambda: wrong.Project().save(paht="project.toml"),
        "invalid_call",
        ["Project.save has no argument 'paht'", "Did you mean 'path'?"],
    ),
    "project-save-overwrite-misspelled": (
        lambda: wrong.Project().save("project.toml", overwite=True),
        "invalid_call",
        ["Project.save has no argument 'overwite'", "Did you mean 'overwrite'?"],
    ),
    "list-analyses-with-a-keyword": (
        lambda: wrong.list_analyses(limit=3),
        "invalid_call",
        ["list_analyses takes no arguments, but got 'limit'"],
    ),
    "guide-with-a-keyword": (
        lambda: wrong.guide(topic="x"),
        "invalid_call",
        ["guide takes no arguments, but got 'topic'"],
    ),
}


@pytest.mark.parametrize("mistake", list(MISTAKES))
def test_an_argument_mistake_gets_its_code_and_fix(mistake):
    make, code, fragments = MISTAKES[mistake]

    with pytest.raises(AgentEDAError) as raised:
        make()

    message = str(raised.value)
    assert raised.value.code == code
    for fragment in fragments:
        assert fragment in message
    assert LEAKS.findall(message) == []
    assert "\n" not in message


def test_a_name_of_the_package_misspelled_in_case_only_names_the_right_one():
    with pytest.raises(AttributeError) as raised:
        getattr(ae, "SUMMARIZE_NUMERIC")  # noqa: B009  # the name is the test

    assert "no attribute 'SUMMARIZE_NUMERIC'" in str(raised.value)
    assert "Did you mean 'summarize_numeric'?" in str(raised.value)


@pytest.mark.parametrize(
    ("toml", "kind"),
    [
        ("name = 1.5", "a float"),
        ("name = 2020-01-01", "a date or time"),
        ("name = [1, 2]", "an array"),
        ("name = { first = 1 }", "a table"),
    ],
)
def test_a_project_file_value_of_the_wrong_kind_names_what_toml_calls_it(
    tmp_path, toml, kind
):
    path = tmp_path / "project.toml"
    path.write_text(
        f"{toml}\n[data_policy]\nsensitive_columns = []\nmask_sensitive = false\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentEDAError) as raised:
        ae.Project.load(path)

    assert raised.value.code == "invalid_project_file"
    assert f"found {kind} for name" in str(raised.value)
    assert LEAKS.findall(str(raised.value)) == []


@pytest.mark.skipif(sys.platform == "win32", reason="Windows reads long names alike")
def test_a_project_file_whose_name_cannot_be_read_says_why():
    with pytest.raises(AgentEDAError) as raised:
        ae.Project.load("x" * 300 + ".toml")

    assert raised.value.code == "invalid_project_file"
    assert "Project.load can't read" in str(raised.value)
    assert "name too long" in str(raised.value)


def test_a_path_names_the_call_the_fix_belongs_to():
    with pytest.raises(AgentEDAError) as raised:
        wrong.find_candidate_keys(pathlib.Path("churn.csv"))

    assert "ae.find_candidate_keys(pd.read_csv(path))" in str(raised.value)


def test_a_sentinel_for_a_column_that_columns_leaves_out_says_to_add_it():
    with pytest.raises(AgentEDAError) as raised:
        ae.detect_missing_sentinels(
            DF, columns=["age"], sentinel_values={"email": ["n/a"]}, id_column="email"
        )

    assert "Add it to columns" in str(raised.value)
    assert "leaving the columns argument out" not in str(raised.value)


@pytest.mark.parametrize(
    "name",
    [
        "describe_analysis",
        "search_analyses",
        "list_analyses",
        "guide",
    ],
)
def test_a_discovery_function_keeps_its_call_after_the_keyword_check(name):
    function = getattr(ae, name)

    assert function.__name__ == name
    assert function.__doc__
