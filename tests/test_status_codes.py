"""Statuses and codes an agent can branch on: only reachable statuses are declared,
and every reason, limitation and exclusion code comes from a documented list."""

import json
import re
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from recipes import recipe_for

import agent_eda as ae
import agent_eda.data.duplicates
from agent_eda.results import (
    ExclusionReason,
    LimitationCode,
    Status,
    StatusReason,
    StatusReasonCode,
)

README = Path(__file__).resolve().parents[1] / "README.md"
ANALYSES = [info.name for info in ae.list_analyses().analyses]

EMPTY = pd.DataFrame(
    {"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="object")}
)
NO_COLUMNS = pd.DataFrame(index=range(3))
LISTS = pd.DataFrame({"x": [[1], [2], [3]]})
NUMBER_AND_NOTHING = pd.DataFrame({"a": [1.0, 2.0, 3.0], "p": [np.nan] * 3})
TEXT_AND_NOTHING = pd.DataFrame({"a": ["x", "y", "z"], "p": [None] * 3})
NOTHING = pd.DataFrame({"p": [None, None, None]})
NUMBERS = pd.DataFrame({"a": [1, 2, 3], "b": [3, 1, 2]})
WORDS = pd.DataFrame({"a": ["x", "y", "z"]})
ONE_ROW = pd.DataFrame({"a": [1], "b": [2]})
THREE = pd.DataFrame({"a": [1, 2, 3], "b": [3, 1, 2], "c": [2, 3, 1]})
TWO_ROWS = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]})
INFINITE = pd.DataFrame({"a": [np.inf, -np.inf, np.inf]})
CLASSES = {"target_column": "p", "task": "binary_classification"}
VALUES = {"target_column": "a", "task": "regression"}

# Every status an analysis can return and why: the analysis, the data and
# arguments that make it return them, the status and its reason code.
REACHED = [
    ("describe_dataset", NUMBERS, {}, "completed", None),
    ("infer_column_semantics", NUMBERS, {}, "completed", None),
    ("infer_column_semantics", NUMBER_AND_NOTHING, {}, "partial", "incomplete_columns"),
    ("infer_column_semantics", EMPTY, {}, "insufficient_data", "no_rows"),
    ("infer_column_semantics", NOTHING, {}, "insufficient_data", "no_values"),
    ("infer_column_semantics", NO_COLUMNS, {}, "not_applicable", "no_columns"),
    ("infer_column_semantics", LISTS, {}, "not_applicable", "unsupported_values"),
    ("validate_column_types", NUMBERS, {"schema": {"a": "integer"}}, "completed", None),
    (
        "validate_column_types",
        EMPTY,
        {"schema": {"a": "integer"}},
        "insufficient_data",
        "no_rows",
    ),
    ("profile_missing_values", NUMBERS, {}, "completed", None),
    ("profile_missing_values", EMPTY, {}, "insufficient_data", "no_rows"),
    ("profile_missing_values", NO_COLUMNS, {}, "not_applicable", "no_columns"),
    ("detect_missing_sentinels", NUMBERS, {}, "completed", None),
    ("detect_missing_sentinels", EMPTY, {}, "insufficient_data", "no_rows"),
    ("detect_missing_sentinels", NO_COLUMNS, {}, "not_applicable", "no_columns"),
    ("summarize_numeric", NUMBERS, {}, "completed", None),
    ("summarize_numeric", NUMBER_AND_NOTHING, {}, "partial", "incomplete_columns"),
    ("summarize_numeric", EMPTY, {}, "insufficient_data", "no_rows"),
    ("summarize_numeric", INFINITE, {}, "insufficient_data", "no_values"),
    ("summarize_numeric", WORDS, {}, "not_applicable", "no_columns"),
    ("summarize_categorical", WORDS, {}, "completed", None),
    ("summarize_categorical", TEXT_AND_NOTHING, {}, "partial", "incomplete_columns"),
    ("summarize_categorical", EMPTY, {}, "insufficient_data", "no_rows"),
    ("summarize_categorical", NOTHING, {}, "insufficient_data", "no_values"),
    ("summarize_categorical", NUMBERS, {}, "not_applicable", "no_columns"),
    ("summarize_categorical", LISTS, {}, "not_applicable", "unhashable_values"),
    ("find_duplicate_rows", NUMBERS, {}, "completed", None),
    ("find_duplicate_rows", EMPTY, {}, "insufficient_data", "no_rows"),
    ("find_duplicate_rows", NO_COLUMNS, {}, "not_applicable", "no_columns"),
    ("find_candidate_keys", THREE, {}, "completed", None),
    (
        "find_candidate_keys",
        THREE,
        {"max_combinations": 1},
        "budget_exceeded",
        "too_many_combinations",
    ),
    ("find_candidate_keys", EMPTY, {}, "insufficient_data", "no_rows"),
    ("find_candidate_keys", ONE_ROW, {}, "insufficient_data", "one_row"),
    ("find_candidate_keys", NO_COLUMNS, {}, "not_applicable", "no_columns"),
    ("find_candidate_keys", LISTS, {}, "not_applicable", "unhashable_values"),
    (
        "profile_target",
        THREE,
        {"target_column": "a", "task": "regression"},
        "completed",
        None,
    ),
    ("profile_target", EMPTY, VALUES, "insufficient_data", "no_rows"),
    ("profile_target", NOTHING, CLASSES, "insufficient_data", "no_values"),
    ("profile_target", INFINITE, VALUES, "insufficient_data", "no_values"),
    ("compute_association_matrix", THREE, {}, "completed", None),
    (
        "compute_association_matrix",
        THREE,
        {"max_pairs": 1},
        "budget_exceeded",
        "too_many_pairs",
    ),
    ("compute_association_matrix", EMPTY, {}, "insufficient_data", "no_rows"),
    ("compute_association_matrix", TWO_ROWS, {}, "insufficient_data", "no_values"),
    ("compute_association_matrix", WORDS, {}, "not_applicable", "no_columns"),
]


def case_id(case):
    name, _, _, status, code = case
    return f"{name}-{status}-{code}"


def declared(name):
    """The (status, reason code) pairs an analysis declares, as describe_analysis
    tells an agent; completed has no reason."""
    pairs = set()
    for status in ae.describe_analysis(name).statuses:
        if status.status == "completed":
            pairs.add(("completed", None))
        pairs |= {(status.status, entry.code) for entry in status.reason_codes}
    return pairs


@pytest.mark.parametrize("case", REACHED, ids=case_id)
def test_each_declared_status_and_reason_code_is_returned_for_data_that_causes_it(case):
    name, data, options, status, code = case

    res = getattr(ae, name)(data, **options)

    assert res.status == status
    assert (res.status_reason.code if res.status_reason else None) == code


@pytest.mark.parametrize("name", ANALYSES)
def test_an_analysis_declares_exactly_the_statuses_and_codes_it_returns(name):
    reached = {(status, code) for n, _, _, status, code in REACHED if n == name}

    assert declared(name) == reached, (
        f"{name}: add a row to REACHED for each status and reason code its "
        "Statuses: section declares, with data that makes it return them."
    )


def test_no_analysis_declares_failed_and_the_status_type_still_defines_it():
    for info in ae.list_analyses().analyses:
        assert "failed" not in info.statuses
        assert not re.search(r"^\s+failed:", ae.describe_analysis(info.name).example)
        docstring = getattr(ae, info.name).__doc__ or ""
        assert "failed:" not in docstring
    assert "failed" in typing.get_args(Status)


def test_a_failed_result_still_carries_a_documented_reason_code():
    reason = StatusReason(code="numerical_failure", message="A statistic overflowed.")

    assert reason.code == "numerical_failure"
    assert "numerical_failure" in typing.get_args(StatusReasonCode)


def test_every_reason_code_but_the_reserved_one_is_declared_by_some_analysis():
    used = {code for name in ANALYSES for _, code in declared(name) if code}

    assert set(typing.get_args(StatusReasonCode)) - used == {"numerical_failure"}


def test_a_reason_code_outside_the_list_is_rejected():
    with pytest.raises(ValidationError, match="no_columns_to_compare"):
        StatusReason(code="no_columns_to_compare", message="x")  # ty: ignore[invalid-argument-type]  # a code outside the list is the point


def test_the_six_spellings_of_nothing_to_work_on_are_one_family():
    codes = set(typing.get_args(StatusReasonCode))

    assert {"no_rows", "no_columns", "no_values"} <= codes
    for old in (
        "no_columns_to_compare",
        "no_columns_to_check",
        "no_numeric_columns",
        "no_categorical_columns",
        "no_labels",
        "no_valid_labels",
        "no_finite_values",
        "no_defined_pairs",
        "fewer_than_two_columns",
        "no_hashable_columns",
    ):
        assert old not in codes


@pytest.mark.parametrize("name", ANALYSES)
def test_a_statuses_entry_names_the_codes_and_describe_analysis_shows_their_meaning(
    name,
):
    description = ae.describe_analysis(name)

    printed = str(description)
    for status in description.statuses:
        assert bool(status.reason_codes) == (status.status != "completed")
        assert "Reason code" not in status.reason
        for entry in status.reason_codes:
            assert entry.meaning
            assert f"{entry.code}: " in printed


def test_a_result_with_a_reason_code_its_analysis_didnt_declare_is_a_bug(monkeypatch):
    monkeypatch.setattr(
        agent_eda.data.duplicates,
        "NO_ROWS",
        StatusReason(code="no_columns", message="The data has no rows."),
    )

    with pytest.raises(RuntimeError, match=r"undeclared reason code 'no_columns'"):
        ae.find_duplicate_rows(EMPTY)


def test_describe_analysis_gives_each_limitation_code_its_meaning():
    description = ae.describe_analysis("summarize_numeric")

    assert [entry.code for entry in description.limitation_codes] == list(
        typing.get_args(LimitationCode)
    )
    assert all(entry.meaning for entry in description.limitation_codes)
    printed = str(description)
    assert "Limitation codes:" in printed
    for entry in description.limitation_codes:
        assert entry.code in printed
    record = json.loads(description.to_json())
    assert record["limitation_codes"][0]["code"] == "unhashable_values"
    assert record["statuses"][0]["reason_codes"] == []


def test_the_readme_gives_each_limitation_code_its_meaning():
    readme = README.read_text(encoding="utf-8")

    for entry in ae.describe_analysis("summarize_numeric").limitation_codes:
        assert f"`{entry.code}`" in readme


def test_the_readme_reference_names_each_statuss_reason_codes():
    readme = README.read_text(encoding="utf-8")

    line = next(
        line
        for line in readme.splitlines()
        if line.startswith("**Statuses:**") and "budget_exceeded" in line
    )
    assert "`budget_exceeded` (`too_many_pairs`)" in line or (
        "`budget_exceeded` (`too_many_combinations`)" in line
    )


# The statuses the guide, the README and the MCP instructions list


def reachable():
    declared_statuses = {
        status for info in ae.list_analyses().analyses for status in info.statuses
    }
    return [s for s in typing.get_args(Status) if s in declared_statuses]


def statuses_in(text):
    """The statuses a line names, in the order Status lists them."""
    return [s for s in typing.get_args(Status) if re.search(rf"\b{s}\b", text)]


def test_the_listed_statuses_are_the_ones_some_analysis_can_return():
    assert reachable() == [
        "completed",
        "partial",
        "insufficient_data",
        "not_applicable",
        "budget_exceeded",
    ]


def test_the_guide_lists_only_reachable_statuses(capsys):
    ae.guide()
    text = " ".join(capsys.readouterr().out.split())

    sentence = re.search(r"res\.status says how the run ended: ([^;]*);", text)
    assert sentence is not None
    assert statuses_in(sentence[1]) == reachable()


def test_the_readme_lists_only_reachable_statuses():
    readme = README.read_text(encoding="utf-8")

    line = next(
        line for line in readme.splitlines() if line.startswith("- `res.status` says")
    )
    assert statuses_in(line) == reachable()
    for entry in re.findall(r"^\| `status` \| (.*) \|$", readme, flags=re.M):
        assert not set(statuses_in(entry)) - set(reachable())


# Exclusion reasons


def test_every_exclusion_reason_is_one_of_the_documented_spellings():
    allowed = set(typing.get_args(ExclusionReason))
    seen = set()
    data = pd.DataFrame(
        {
            "x": [1.0, np.nan, np.inf, 4.0],
            "g": ["a", None, "b", "a"],
            "many": ["p", "q", "r", "s"],
            "lists": [[1], [2], [3], [4]],
            "empty": [np.nan] * 4,
            "inf": [np.inf, -np.inf, np.inf, np.inf],
        }
    )
    for res in (
        ae.compute_association_matrix(data),
        ae.compute_association_matrix(data, max_categories=2),
        ae.summarize_numeric(data),
        ae.summarize_categorical(data),
        ae.find_candidate_keys(data),
        ae.profile_missing_values(data),
        ae.detect_missing_sentinels(data),
        ae.find_duplicate_rows(data),
        ae.validate_column_types(data, schema={"x": "float"}),
    ):
        for column in res.support.columns:
            seen |= set(column.excluded_by_reason)
    target = ae.profile_target(
        pd.DataFrame({"y": ["a", "b", None, "c"]}),
        target_column="y",
        task="binary_classification",
        expected_labels=["a", "b"],
    )
    seen |= set(target.support.columns[0].excluded_by_reason)

    assert seen <= allowed
    assert seen == allowed


def test_a_column_left_out_of_the_pairs_is_excluded_for_the_reason_its_values_have():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "empty": [np.nan] * 4,
            "inf": [np.inf, -np.inf, np.nan, np.inf],
            "lists": [[1], [2], [3], [4]],
        }
    )

    res = ae.compute_association_matrix(data)

    excluded = {c.column: c.excluded_by_reason for c in res.support.columns}
    assert excluded["empty"] == {"missing": 4}
    assert excluded["inf"] == {"missing": 1, "infinite": 3}
    assert excluded["lists"] == {"unhashable_values": 4}


def test_summarize_categorical_and_find_candidate_keys_spell_unhashable_alike():
    data = pd.DataFrame({"a": [[1], [2], [3]], "b": ["x", "y", "z"]})

    categorical = ae.summarize_categorical(data)
    keys = ae.find_candidate_keys(data)

    assert categorical.support.columns[0].excluded_by_reason == {"unhashable_values": 3}
    assert keys.support.columns[0].excluded_by_reason == {"unhashable_values": 3}


def test_an_exclusion_reason_outside_the_list_is_rejected():
    with pytest.raises(ValidationError, match="unhashable"):
        ae.results.ColumnSupport(
            column="a",
            eligible_count=1,
            evaluated_count=0,
            excluded_count=1,
            excluded_by_reason={"unhashable": 1},  # ty: ignore[invalid-argument-type]  # a reason outside the list is the point
        )


# Nulls and checks that didn't run leave a limitation


def limitation(res, code):
    [found] = [lim for lim in res.limitations if lim.code == code]
    return found


def test_detect_missing_sentinels_without_declared_values_says_the_check_didnt_run():
    data = pd.DataFrame({"a": [1, 2, -999, 4]})

    without = ae.detect_missing_sentinels(data)
    declared_values = ae.detect_missing_sentinels(data, sentinel_values={"a": [-999]})

    assert limitation(without, "sentinel_values_not_declared").fields == ["candidates"]
    assert declared_values.limitations == []


def test_infer_column_semantics_without_a_schema_says_no_column_was_compared():
    data = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    without = ae.infer_column_semantics(data)
    declared_schema = ae.infer_column_semantics(data, schema={"a": "integer"})

    assert limitation(without, "schema_not_declared").fields == [
        "columns['a'].matches_declared",
        "columns['b'].matches_declared",
    ]
    assert "schema_not_declared" not in [
        lim.code for lim in declared_schema.limitations
    ]


def test_infer_column_semantics_names_each_null_inferred_type_in_a_limitation():
    data = pd.DataFrame({"a": [1, 2, 3], "p": [None] * 3, "lists": [[1], [2], [3]]})

    res = ae.infer_column_semantics(data)

    assert limitation(res, "type_not_inferred").fields == [
        "columns['p'].inferred_logical_type",
        "columns['lists'].inferred_logical_type",
    ]
    nulls = [
        name
        for name, column in res.columns.items()
        if column.inferred_logical_type is None
    ]
    assert nulls == ["p", "lists"]
    assert "type_not_inferred" not in [
        lim.code for lim in ae.infer_column_semantics(NUMBERS).limitations
    ]


def test_validate_column_types_says_when_nullable_wasnt_declared():
    data = pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})

    res = ae.validate_column_types(
        data, schema={"a": "integer", "b": ae.ColumnSpec("text", nullable=True)}
    )

    assert limitation(res, "nullable_not_declared").fields == ["columns['a'].nullable"]
    every = ae.validate_column_types(
        data,
        schema={
            "a": ae.ColumnSpec("integer", nullable=False),
            "b": ae.ColumnSpec("text", nullable=True),
        },
    )
    assert every.limitations == []


@pytest.mark.parametrize("task", ["regression", "binary_classification"])
def test_profile_target_names_the_fields_of_the_task_it_didnt_profile(task):
    data = pd.DataFrame({"y": [1, 0, 1, 0, 1]})

    res = ae.profile_target(data, target_column="y", task=task)

    found = limitation(res, "not_applicable_to_task")
    record = res.to_dict()
    assert found.fields
    for path in found.fields:
        assert record[path] is None
    if task == "regression":
        assert "class_count" in found.fields
        assert "statistics" not in found.fields
    else:
        assert found.fields == ["statistics", "non_finite_count"]


def test_duplicate_share_is_null_with_a_limitation_when_nothing_was_compared():
    nothing_to_compare = ae.find_duplicate_rows(NO_COLUMNS)
    no_rows = ae.find_duplicate_rows(EMPTY)

    for res in (nothing_to_compare, no_rows):
        assert res.duplicate_share is None
        assert limitation(res, "too_few_values").fields == ["duplicate_share"]
    assert nothing_to_compare.status == "not_applicable"
    assert ae.find_duplicate_rows(NUMBERS).duplicate_share == 0.0


# Data without rows


@pytest.mark.parametrize("name", [n for n in ANALYSES if n != "describe_dataset"])
def test_data_without_rows_gives_insufficient_data_in_every_analysis(name):
    options = recipe_for(name).required(EMPTY) or {}

    res = getattr(ae, name)(EMPTY, **options)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_rows"


def test_describe_dataset_describes_data_without_rows_and_flags_it():
    res = ae.describe_dataset(EMPTY)

    assert res.status == "completed"
    assert [f.code for f in res.findings] == ["empty_data"]


def test_validate_column_types_still_reports_a_declared_column_the_data_lacks_without_rows():
    res = ae.validate_column_types(EMPTY, schema={"a": "integer", "zip": "text"})

    assert res.status == "insufficient_data"
    assert "missing_declared_columns" in [f.code for f in res.findings]
    assert res.columns["a"].compatible_count == 0
