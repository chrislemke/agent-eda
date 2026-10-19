"""Every analysis agrees on which values are equal.

pandas' factorize reads '\\x00' in text as the end of the text, so '' and
'\\x00a' are one value to it; Python's == tells them apart, and so must the
analyses. Values pandas can't hash, such as lists, are handled by one policy
per analysis. The property at the end counts levels, groups, classes and keys
with Python's == and compares them with each analysis's own.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from strategies import data_frames

import agent_eda as ae

NUL_TEXT = ["", "\x00a", "b"]


def nul_frame() -> pd.DataFrame:
    """An object column of three distinct texts, two of which pandas merges."""
    return pd.DataFrame({"t": pd.Series(NUL_TEXT, dtype=object)})


def test_describe_dataset_counts_text_with_a_nul_character_as_distinct():
    res = ae.describe_dataset(nul_frame())

    assert res.columns[0].distinct_count == 3


def test_summarize_categorical_counts_text_with_a_nul_character_as_distinct():
    res = ae.summarize_categorical(nul_frame())

    assert res.columns["t"].cardinality == 3


def test_infer_column_semantics_counts_text_with_a_nul_character_as_distinct():
    res = ae.infer_column_semantics(nul_frame())

    assert res.columns["t"].evidence.distinct_count == 3


def test_profile_target_counts_text_with_a_nul_character_as_classes():
    res = ae.profile_target(
        nul_frame(), target_column="t", task="multiclass_classification"
    )

    assert res.class_count == 3


def test_compute_association_matrix_counts_text_with_a_nul_character_as_levels():
    data = pd.DataFrame(
        {
            "t": pd.Series(["", "\x00a", "b"] * 3, dtype=object),
            "u": ["x", "y", "z"] * 3,
        }
    )

    res = ae.compute_association_matrix(data, columns=["t", "u"])

    assert {n: c.level_count for n, c in res.columns.items()} == {"t": 3, "u": 3}


def test_find_duplicate_rows_tells_text_with_a_nul_character_apart():
    data = pd.DataFrame({"t": pd.Series(["a", "a\x00b"], dtype=object)})

    res = ae.find_duplicate_rows(data)

    assert res.duplicate_group_count == 0
    assert res.duplicate_groups == []


def test_find_duplicate_rows_still_groups_equal_text_with_a_nul_character():
    data = pd.DataFrame({"t": pd.Series(["a\x00b", "a", "a\x00b"], dtype=object)})

    res = ae.find_duplicate_rows(data)

    assert [g.row_ids for g in res.duplicate_groups] == [[0, 2]]


def test_find_candidate_keys_finds_text_with_a_nul_character_a_key():
    data = pd.DataFrame({"t": pd.Series(["a", "a\x00b"], dtype=object)})

    res = ae.find_candidate_keys(data)

    assert [k.columns for k in res.keys] == [["t"]]


def test_find_candidate_keys_tells_text_with_a_nul_character_apart_in_a_set():
    data = pd.DataFrame(
        {
            "a": pd.Series(["", "\x00a", "", "\x00a"], dtype=object),
            "b": [1, 1, 2, 2],
        }
    )

    res = ae.find_candidate_keys(data)

    assert [k.columns for k in res.keys] == [["a", "b"]]


def test_row_ids_from_text_with_a_nul_character_are_not_ambiguous():
    data = pd.DataFrame(
        {"id": pd.Series(["", "\x00a", "b"], dtype=object), "v": [1, 1, 1]}
    )

    res = ae.find_duplicate_rows(data, columns=["v"], id_column="id")

    assert "ambiguous_row_ids" not in {limit.code for limit in res.limitations}


@pytest.mark.parametrize("dtype", [object, "string"])
def test_validate_column_types_counts_values_that_hold_a_nul_character(dtype):
    data = pd.DataFrame({"qty": pd.Series(["1\x00x", "1", "2", "1\x00x"], dtype=dtype)})

    res = ae.validate_column_types(data, schema={"qty": "integer"})

    column = res.columns["qty"]
    assert (column.convertible_count, column.invalid_count) == (2, 2)
    assert column.invalid_row_ids == [0, 3]


def test_validate_column_types_reads_object_text_as_string_dtype_text():
    values = ["1\x00x", "1", "2", "1\x00x"]
    schema: Any = {"qty": "integer"}

    def read(dtype):
        res = ae.validate_column_types(
            pd.DataFrame({"qty": pd.Series(values, dtype=dtype)}), schema=schema
        )
        return res.columns["qty"].model_dump(exclude={"dtype"})

    assert read(object) == read("string")


UNHASHABLE_MIXES = {
    "int and list": [1, [2], 3],
    "int and dict": [1, {"a": 2}, 3],
    "int, float and list": [1, 2.5, [3]],
}


@pytest.mark.parametrize("values", UNHASHABLE_MIXES.values(), ids=UNHASHABLE_MIXES)
def test_find_candidate_keys_leaves_out_a_column_that_mixes_scalars_with_lists(values):
    data = pd.DataFrame({"payload": pd.Series(values, dtype=object), "id": [1, 2, 3]})

    res = ae.find_candidate_keys(data)

    assert res.left_out_columns == ["payload"]
    assert "unhashable_values" in {limit.code for limit in res.limitations}
    assert [k.columns for k in res.keys] == [["id"]]


def with_list(values: list) -> pd.DataFrame:
    """An object column p that holds the values, and the IDs 1, 2, 3."""
    return pd.DataFrame({"p": pd.Series(values, dtype=object), "id": [1, 2, 3]})


# What each analysis does with a column that holds a list: how to call it, what
# in its result shows the policy, and what that is.
POLICIES = {
    "find_duplicate_rows compares by content": (
        lambda: ae.find_duplicate_rows(with_list([[1], 5, [1]]), columns=["p"]),
        lambda res: [g.row_ids for g in res.duplicate_groups],
        [[0, 2]],
    ),
    "summarize_categorical leaves the column out": (
        lambda: ae.summarize_categorical(with_list([1, [2], 3]), columns=["p"]),
        lambda res: (res.columns["p"].status, res.columns["p"].cardinality),
        ("not_applicable", None),
    ),
    "find_candidate_keys leaves the column out": (
        lambda: ae.find_candidate_keys(with_list([1, [2], 3])),
        lambda res: [res.left_out_columns, [k.columns for k in res.keys]],
        [["p"], [["id"]]],
    ),
    "compute_association_matrix leaves the column out": (
        lambda: ae.compute_association_matrix(
            with_list([1, [2], 3]), columns=["p", "id"]
        ),
        lambda res: [(s.column, s.reason) for s in res.skipped_columns],
        [("p", "unhashable_values")],
    ),
    "describe_dataset leaves the distinct count null": (
        lambda: ae.describe_dataset(with_list([1, [2], 3])),
        lambda res: [c.distinct_count for c in res.columns],
        [None, 3],
    ),
    "infer_column_semantics leaves the distinct count null": (
        lambda: ae.infer_column_semantics(with_list([1, [2], 3])),
        lambda res: res.columns["p"].evidence.distinct_count,
        None,
    ),
}


@pytest.mark.parametrize("policy", POLICIES)
def test_each_analysis_keeps_its_policy_for_a_column_that_holds_a_list(policy):
    call, read, expected = POLICIES[policy]

    assert read(call()) == expected


def test_find_duplicate_rows_compares_sets_that_hold_missing_values_by_content():
    nan = float("nan")
    data = pd.DataFrame(
        {"s": pd.Series([{1, nan}, {1, None}, {1, float("nan")}, {1}], dtype=object)}
    )

    res = ae.find_duplicate_rows(data)

    assert [g.row_ids for g in res.duplicate_groups] == [[0, 1, 2]]


# Python's ==, the reference every analysis is compared with.
def is_missing(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    )


def is_unhashable(value: object) -> bool:
    return isinstance(value, list | dict | set | np.ndarray)


def python_groups(values: list) -> list[list[int]]:
    """Positions of equal values by ==, with every missing value equal."""
    groups: list[tuple[object, list[int]]] = []
    for position, value in enumerate(values):
        for representative, members in groups:
            if (is_missing(value) and is_missing(representative)) or (
                not is_missing(value)
                and not is_missing(representative)
                and bool(value == representative)
            ):
                members.append(position)
                break
        else:
            groups.append((value, [position]))
    return [members for _, members in groups]


def python_distinct(values: list) -> int:
    return len(python_groups([v for v in values if not is_missing(v)]))


def python_repeated(values: list) -> list[list[int]]:
    return [group for group in python_groups(values) if len(group) > 1]


Check = Any


def check_describe_dataset(df: pd.DataFrame) -> None:
    res = ae.describe_dataset(df)
    for position, column in enumerate(res.columns):
        values = list(df.iloc[:, position])
        expected = None if any(map(is_unhashable, values)) else python_distinct(values)
        assert column.distinct_count == expected, (column.column, values)


def check_summarize_categorical(df: pd.DataFrame) -> None:
    res = ae.summarize_categorical(df, columns=list(df.columns))
    for name in df.columns:
        values = list(df[name])
        if any(map(is_unhashable, values)):
            assert res.columns[name].cardinality is None, (name, values)
        else:
            assert res.columns[name].cardinality == python_distinct(values), values


def check_infer_column_semantics(df: pd.DataFrame) -> None:
    res = ae.infer_column_semantics(df)
    for name in df.columns:
        values = list(df[name])
        expected = None if any(map(is_unhashable, values)) else python_distinct(values)
        assert res.columns[name].evidence.distinct_count == expected, values


def check_find_duplicate_rows(df: pd.DataFrame) -> None:
    for name in df.columns:
        values = list(df[name])
        res = ae.find_duplicate_rows(df, columns=[name])
        if res.status != "completed":
            continue
        got = sorted(g.row_ids for g in res.duplicate_groups)
        assert got == python_repeated(values), values


def check_find_candidate_keys(df: pd.DataFrame) -> None:
    for name in df.columns:
        values = list(df[name])
        res = ae.find_candidate_keys(df, columns=[name], max_key_size=1)
        if any(map(is_unhashable, values)):
            assert res.left_out_columns == [name], values
        elif len(values) >= 2:
            found = [k.columns for k in res.keys + res.keys_with_missing_values]
            unique = not python_repeated(values)
            assert (found == [[name]]) == unique, values


def check_profile_target(df: pd.DataFrame) -> None:
    for name in df.columns:
        values = list(df[name])
        if any(map(is_unhashable, values)):
            continue
        res = ae.profile_target(
            df, target_column=name, task="multiclass_classification"
        )
        if res.status == "completed":
            assert res.class_count == python_distinct(values), values


def check_compute_association_matrix(df: pd.DataFrame) -> None:
    res = ae.compute_association_matrix(
        df, columns=list(df.columns), max_categories=1000
    )
    for name, column in res.columns.items():
        values = list(df[name])
        assert column.level_count == python_distinct(values), values


# The analyses that count levels, groups, classes or keys, each with a check
# against Python's ==. Every other analysis counts none of them.
CHECKS = {
    "describe_dataset": check_describe_dataset,
    "summarize_categorical": check_summarize_categorical,
    "infer_column_semantics": check_infer_column_semantics,
    "find_duplicate_rows": check_find_duplicate_rows,
    "find_candidate_keys": check_find_candidate_keys,
    "profile_target": check_profile_target,
    "compute_association_matrix": check_compute_association_matrix,
}
COUNTS_NOTHING = {
    "detect_missing_sentinels",
    "profile_missing_values",
    "validate_column_types",
    "summarize_numeric",
}


def test_every_analysis_is_checked_against_pythons_equality_or_counts_nothing():
    names = {info.name for info in ae.list_analyses().analyses}

    assert names == set(CHECKS) | COUNTS_NOTHING, (
        "Add a check to CHECKS in tests/test_value_identity.py for an analysis that "
        "counts levels, groups, classes or keys, or name it in COUNTS_NOTHING."
    )


def text_and_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The columns whose values pandas compares as Python objects or text."""
    keep = [
        position
        for position, dtype in enumerate(df.dtypes)
        if isinstance(dtype, pd.StringDtype) or dtype is np.dtype(object)
    ]
    return df.iloc[:, keep]


@pytest.mark.parametrize("name", sorted(CHECKS))
@settings(deadline=None)
@given(data=data_frames())
def test_counts_of_levels_groups_classes_and_keys_equal_a_count_by_python(name, data):
    df = text_and_object_columns(data).reset_index(drop=True)
    assume(df.shape[1] > 0)
    df.columns = pd.Index([f"c{position}" for position in range(df.shape[1])])

    CHECKS[name](df)
