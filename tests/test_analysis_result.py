import json
import platform

import numpy as np
import pandas as pd
import pydantic
import pytest

import agent_eda as ae


@pytest.fixture
def result():
    df = pd.DataFrame({"x": [1.0, 2.0, None], "y": ["a", "b", "c"]})
    return ae.describe_dataset(df, source="a.csv")


def test_result_is_a_frozen_pydantic_model_named_after_the_analysis(result):
    assert isinstance(result, ae.AnalysisResult)
    assert isinstance(result, pydantic.BaseModel)
    assert type(result).__name__ == "DescribeDatasetResult"
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        result.status = "failed"


def test_result_names_its_analysis_and_how_the_run_ended(result):
    assert result.analysis == "describe_dataset"
    assert result.method is None
    assert result.status == "completed"
    assert result.status_reason is None
    assert result.limitations == []


def test_a_failed_result_carries_an_error_code_and_a_message(result):
    record = result.model_dump()
    reason = {"code": "overflow", "message": "The memory total overflowed."}

    failed = type(result).model_validate(
        {**record, "status": "failed", "status_reason": reason}
    )

    assert failed.status_reason is not None
    assert failed.status_reason.code == "overflow"
    assert str(failed).splitlines()[:2] == [
        "describe_dataset: failed on 3 rows x 2 columns",
        "Reason: overflow: The memory total overflowed.",
    ]


def test_an_unexpected_exception_propagates_instead_of_becoming_a_result():
    class Exploding:
        def __hash__(self):
            raise ZeroDivisionError("a value's hash is broken")

    df = pd.DataFrame({"x": [Exploding(), Exploding()]})

    with pytest.raises(ZeroDivisionError, match="hash is broken"):
        ae.describe_dataset(df)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("failed", None),
        ("completed", {"code": "overflow", "message": "Overflowed."}),
        ("cancelled", None),
    ],
    ids=["failed-without-reason", "completed-with-reason", "unknown-status"],
)
def test_status_and_status_reason_must_agree(result, status, reason):
    record = {**result.model_dump(), "status": status, "status_reason": reason}

    with pytest.raises(pydantic.ValidationError):
        type(result).model_validate(record)


def test_parameters_record_every_argument_and_only_the_shape_of_the_data(result):
    assert result.parameters == {
        "data": {"row_count": 3, "column_count": 2},
        "source": "a.csv",
        "source_version": None,
    }


def test_support_records_rows_columns_and_the_counts_of_columns_with_exclusions():
    support = ae.summarize_numeric(
        pd.DataFrame({"x": [1.0, 2.0, None], "y": [1.0, 2.0, 3.0]})
    ).support
    assert support.row_count == 3
    assert support.column_count == 2
    assert [
        (
            c.column,
            c.eligible_count,
            c.evaluated_count,
            c.excluded_count,
            c.excluded_by_reason,
        )
        for c in support.columns
    ] == [("x", 3, 2, 1, {"missing": 1})]


def test_execution_records_library_versions_and_no_timing(result):
    assert result.execution.versions == {
        "agent_eda": ae.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    record = result.to_json()
    for word in ("time", "duration", "elapsed", "timestamp"):
        assert word not in record


def test_to_dict_gives_json_compatible_python_objects(result):
    record = result.to_dict()

    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["analysis"] == "describe_dataset"
    assert record["columns"][0]["column"] == "x"


def test_to_json_is_strict_json_of_the_same_record(result):
    text = result.to_json()

    assert json.loads(text, parse_constant=reject_constant) == result.to_dict()


def test_identical_calls_give_byte_identical_json():
    df = pd.DataFrame(
        {
            "city": ["Oslo", "Lyon", None, "Oslo"],
            "kind": pd.Categorical(["b", "a", "b", "b"]),
            "n": [3, 1, 2, 3],
        }
    )

    first = ae.describe_dataset(df).to_json()
    second = ae.describe_dataset(df.copy()).to_json()

    assert first == second


def test_record_keeps_the_first_100_columns_and_their_total():
    df = pd.DataFrame({f"c{i}": [None, None] for i in range(250)})

    res = ae.describe_dataset(df)
    record = res.to_dict()

    assert [c["column"] for c in record["columns"]] == [f"c{i}" for i in range(100)]
    assert record["columns_total"] == 250
    assert record["support"]["columns"] == []
    assert record["support"]["columns_total"] == 0
    finding = record["findings"][0]
    assert finding["code"] == "all_missing_columns"
    assert (len(finding["columns"]), finding["columns_total"]) == (100, 250)
    assert (len(finding["fields"]), finding["fields_total"]) == (100, 250)
    assert record["findings_total"] == 1
    assert record["limitations_total"] == 0
    assert len(res.columns) == 250
    assert len(res.findings[0].columns) == 250


def test_record_caps_list_parameters_too(result):
    # Later analyses record their resolved column list among the parameters.
    record = result.model_dump()
    resolved = [f"c{i}" for i in range(150)]
    with_columns = type(result).model_validate(
        {**record, "parameters": {**record["parameters"], "columns": resolved}}
    )

    parameters = with_columns.to_dict()["parameters"]

    assert parameters["columns"] == resolved[:100]
    assert parameters["columns_total"] == 150
    assert parameters["source"] == "a.csv"
    assert with_columns.parameters["columns"] == resolved
    assert "source_total" not in result.to_dict()["parameters"]


def test_to_frame_returns_the_complete_per_column_table():
    df = pd.DataFrame({f"c{i}": [float(i), None] for i in range(150)})

    frame = ae.describe_dataset(df).to_frame()

    assert list(frame.columns) == [
        "position",
        "column",
        "label_type",
        "dtype",
        "non_missing_count",
        "missing_count",
        "distinct_count",
        "memory_bytes",
    ]
    assert len(frame) == 150
    assert frame["column"].tolist() == [f"c{i}" for i in range(150)]
    assert frame["missing_count"].tolist() == [1] * 150
    assert frame["memory_bytes"].tolist() == [16] * 150


def test_to_frame_of_data_without_columns_is_empty_with_the_same_columns():
    frame = ae.describe_dataset(pd.DataFrame(index=range(3))).to_frame()

    assert frame.empty
    assert "distinct_count" in frame.columns


def reject_constant(name):
    raise AssertionError(f"non-strict JSON constant {name}")
