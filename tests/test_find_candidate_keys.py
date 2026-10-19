import itertools
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import data_frames, nan_as_missing

import agent_eda as ae


def orders():
    """Order lines: (order, line) identifies rows, and so does sku with a gap."""
    return pd.DataFrame(
        {
            "order": [1, 1, 2, 2, 3, 3],
            "line": [1, 2, 1, 2, 1, 2],
            "store": ["a", "a", "b", "b", "a", "a"],
            "sku": ["s1", "s2", "s3", "s4", "s5", None],
        }
    )


def test_records_the_default_key_size_and_combination_limit():
    res = ae.find_candidate_keys(orders())

    assert res.parameters["max_key_size"] == 2
    assert res.parameters["max_combinations"] == 10_000


def test_describe_analysis_lists_max_combinations_as_its_work_limit():
    description = ae.describe_analysis("find_candidate_keys")

    assert description.work_limit_parameters == ["max_combinations"]
    described = {p.name: p for p in description.parameters}
    assert described["max_combinations"].default == 10_000
    assert described["max_combinations"].valid_range == "1 <= max_combinations"
    assert described["max_key_size"].default == 2
    assert described["max_key_size"].valid_range == "1 <= max_key_size"


@pytest.mark.parametrize("argument", ["max_key_size", "max_combinations"])
def test_a_limit_below_1_raises_with_the_valid_range(argument):
    options: dict[str, Any] = {argument: 0}

    with pytest.raises(ae.errors.InvalidArgumentError) as caught:
        ae.find_candidate_keys(orders(), **options)

    assert str(caught.value) == (
        f"find_candidate_keys got 0 for {argument}, but {argument} must be an integer with "
        f"1 <= {argument}. See help(ae.find_candidate_keys) for every argument."
    )


def keyed():
    """id identifies rows alone, (a, b) together; a, b and c don't."""
    return pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 1],
            "id": [10, 11, 12, 13, 14],
            "b": ["x", "y", "x", "y", "z"],
            "c": [0, 0, 0, 0, 1],
        }
    )


def test_finds_single_columns_then_combinations_in_increasing_size():
    res = ae.find_candidate_keys(keyed())

    assert [key.columns for key in res.keys] == [["id"], ["a", "b"]]


def test_skips_supersets_of_the_keys_it_has_found():
    res = ae.find_candidate_keys(keyed())

    # Singles a, id, b, c, then the pairs without id: (a, b), (a, c), (b, c).
    assert res.search.examined_count == 4 + 3
    assert ["a", "id"] not in [key.columns for key in res.keys]


def test_a_search_that_examines_every_set_is_exhaustive():
    res = ae.find_candidate_keys(keyed())

    assert res.status == "completed"
    assert res.exhaustive is True
    # 4 single columns and 6 pairs, of which the 3 holding id are skipped.
    assert res.search.combination_count == 10
    assert res.search.examined_count == 7


def test_a_search_that_uses_exactly_max_combinations_still_completes():
    res = ae.find_candidate_keys(keyed(), max_combinations=7)

    assert res.status == "completed"
    assert res.exhaustive is True


def test_hitting_max_combinations_stops_the_search_but_keeps_what_it_found():
    res = ae.find_candidate_keys(keyed(), max_combinations=5)

    assert res.status == "budget_exceeded"
    assert res.exhaustive is False
    assert res.search.examined_count == 5
    # The pair (a, b) was the 5th set examined; (a, c) and (b, c) weren't.
    assert [key.columns for key in res.keys] == [["id"], ["a", "b"]]
    assert res.status_reason is not None
    assert res.status_reason.code == "too_many_combinations"
    assert res.status_reason.message == (
        "The complete search covers up to 10 column sets, more than the "
        "max_combinations limit of 5 allows; it stopped after examining 5 and kept "
        "what it found. Lower max_key_size to 1, raise max_combinations to 10, or "
        "pass fewer columns."
    )


def test_a_search_stopped_among_single_columns_ranks_only_those_it_examined():
    res = ae.find_candidate_keys(unkeyed(), max_combinations=2)

    assert res.status == "budget_exceeded"
    assert [s.columns for s in res.most_unique_column_sets] == [
        ["order_id"],
        ["customer"],
    ]


def test_a_set_unique_on_the_subset_is_still_checked_on_all_rows():
    # 32,768 rows: the subset of 16,384 evenly spaced rows holds every even
    # row, so the repeat between rows 1 and 3 hides from it.
    rows = 32_768
    ids = np.arange(rows)
    ids[1] = ids[3]
    df = pd.DataFrame({"id": ids})

    res = ae.find_candidate_keys(df)

    assert res.search.subset_row_count == 16_384
    assert res.search.rejected_on_subset_count == 0
    assert res.search.checked_on_all_rows_count == 1
    assert res.keys == res.keys_with_missing_values == []
    [found] = res.most_unique_column_sets
    assert (found.distinct_count, found.duplicate_row_count) == (rows - 1, 2)


def test_a_repeat_on_the_subset_rejects_a_set_without_checking_all_rows():
    rows = 32_768
    ids = np.arange(rows)
    ids[2] = ids[4]  # both even rows, so both in the subset
    df = pd.DataFrame({"id": ids})

    res = ae.find_candidate_keys(df)

    assert res.search.rejected_on_subset_count == 1
    assert res.search.checked_on_all_rows_count == 0
    assert res.keys == []


def test_small_data_is_its_own_subset():
    res = ae.find_candidate_keys(keyed())

    assert res.search.subset_row_count == 5
    assert res.search.rejected_on_subset_count == 5
    assert res.search.checked_on_all_rows_count == 2


def test_each_key_reports_its_counts():
    res = ae.find_candidate_keys(keyed())

    single, pair = res.keys
    assert single.columns == ["id"]
    assert single.distinct_count == 5
    assert single.duplicate_row_count == 0
    assert single.missing_row_count == 0
    assert single.uniqueness_share == 1.0
    assert single.is_minimal is True
    assert (pair.distinct_count, pair.duplicate_row_count) == (5, 0)


def test_lists_keys_with_missing_values_apart_from_the_keys():
    res = ae.find_candidate_keys(orders())

    assert [key.columns for key in res.keys] == [["order", "line"]]
    [sku] = res.keys_with_missing_values
    assert sku.columns == ["sku"]
    assert (sku.distinct_count, sku.duplicate_row_count) == (6, 0)
    assert sku.missing_row_count == 1


def test_missing_values_compare_equal_so_two_of_them_break_a_key():
    df = pd.DataFrame({"code": ["a", None, "b", None], "n": [1.0, 2.0, None, 3.0]})

    res = ae.find_candidate_keys(df, columns=["code"])

    assert res.keys == []
    assert res.keys_with_missing_values == []


def test_a_combination_with_a_missing_value_is_a_key_with_missing_values():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", None, "x", None]})

    res = ae.find_candidate_keys(df)

    assert res.keys == []
    [key] = res.keys_with_missing_values
    assert key.columns == ["a", "b"]
    assert key.missing_row_count == 2


def unkeyed():
    """Rows 5 and 6 repeat each other in every column, so no set is a key."""
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6, 6, 8],
            "customer": ["a", "b", "a", "c", "b", "d", "d", "a"],
            "day": [1, 1, 2, 2, 3, 3, 3, 4],
            "flag": [True, False, True, False, True, False, False, True],
        }
    )


def test_without_a_key_lists_the_most_unique_column_sets_first():
    res = ae.find_candidate_keys(unkeyed())

    assert res.keys == res.keys_with_missing_values == []
    # Most distinct combinations first, then fewer columns, then column order.
    # A pair that adds nothing to order_id, such as (order_id, day), isn't
    # minimal and is left out.
    assert [s.columns for s in res.most_unique_column_sets] == [
        ["order_id"],
        ["customer", "day"],
        ["day", "flag"],
        ["customer", "flag"],
        ["customer"],
        ["day"],
        ["flag"],
    ]
    assert [s.distinct_count for s in res.most_unique_column_sets] == [
        *(7, 7, 7, 5, 4, 4, 2)
    ]
    assert all(s.is_minimal for s in res.most_unique_column_sets)


def test_the_counts_of_every_listed_set_match_pandas():
    df = unkeyed()

    res = ae.find_candidate_keys(df)

    assert len(res.most_unique_column_sets) == 7
    for found in res.most_unique_column_sets:
        reference = pandas_counts(df, found.columns)
        assert (
            found.distinct_count,
            found.duplicate_row_count,
            found.missing_row_count,
        ) == reference
        assert found.uniqueness_share == reference[0] / len(df)


def test_prints_the_keys_with_their_counts_and_how_far_the_search_went():
    res = ae.find_candidate_keys(orders())

    assert str(res) == "\n".join(
        [
            "find_candidate_keys: completed on 6 rows x 4 columns",
            "Exhaustive search for keys of up to 2 columns among 4 columns: examined "
            "7 sets, skipped 3 supersets of keys.",
            "Missing values compare equal; a key with missing values can't serve "
            "as a primary key.",
            "Keys (2 of 2 shown):",
            "columns          kind                     distinct_count  "
            "duplicate_row_count  missing_row_count  uniqueness_share",
            "'order', 'line'  key                                   6  "
            "                  0                  0                 1",
            "'sku'            key_with_missing_values               6  "
            "                  0                  1                 1",
            "Findings (2 of 2 shown):",
            "- descriptive keys_found: 1 column set identifies every row and has no "
            "missing values: 'order', 'line'.",
            "- descriptive keys_with_missing_values: 1 column set identifies every "
            "row but has missing values: 'sku'.",
            "Limitations: none",
            "More: res.to_frame() has all 2 column sets; res.findings, "
            "res.limitations and res.to_json() have the rest.",
        ]
    )


def test_a_search_stopped_early_says_so_and_how_to_finish_it():
    res = ae.find_candidate_keys(keyed(), max_combinations=5)

    lines = str(res).splitlines()

    assert lines[1:6] == [
        "Reason: too_many_combinations: The complete search covers up to 10 column "
        "sets, more than the max_combinations limit of",
        "    5 allows; it stopped after examining 5 and kept what it found. Lower "
        "max_key_size to 1, raise max_combinations to",
        "    10, or pass fewer columns.",
        "Not exhaustive: max_combinations stopped the search for keys of up to 2 "
        "columns after 5 of up to 10 sets.",
        "Missing values compare equal; a key with missing values can't serve as a "
        "primary key.",
    ]


def test_without_a_key_prints_the_most_unique_sets_and_says_no_set_is_a_key():
    res = ae.find_candidate_keys(unkeyed(), max_key_size=1)

    lines = str(res).splitlines()

    assert lines[3] == "Most unique column sets (4 of 4 shown):"
    assert lines[5].split() == ["'order_id'", "not_a_key", "7", "2", "0", "0.875"]
    [finding] = res.findings
    assert finding.code == "no_key_found"
    assert finding.assessment == "descriptive"
    assert finding.claim == (
        "No single column is a key; the most unique, 'order_id', has 2 duplicate rows."
    )
    assert finding.fields == [
        "keys",
        "keys_with_missing_values",
        "most_unique_column_sets[0].duplicate_row_count",
    ]


def test_a_search_stopped_early_says_no_examined_set_is_a_key():
    res = ae.find_candidate_keys(unkeyed(), max_combinations=2)

    [finding] = res.findings
    assert finding.claim == (
        "No key in the 2 sets examined; the most unique, 'order_id', has 2 "
        "duplicate rows."
    )


def test_a_complete_search_without_keys_names_its_largest_set_size():
    res = ae.find_candidate_keys(unkeyed())

    [finding] = res.findings
    assert finding.claim == (
        "No key of up to 2 columns; the most unique, 'order_id', has 2 duplicate rows."
    )


def test_findings_name_the_keys_and_the_fields_behind_them():
    res = ae.find_candidate_keys(orders())

    found, with_missing = res.findings
    assert found.code == "keys_found"
    assert found.columns == ["order", "line"]
    assert found.fields == ["keys[0]"]
    assert with_missing.code == "keys_with_missing_values"
    assert with_missing.columns == ["sku"]
    assert with_missing.fields == ["keys_with_missing_values[0].missing_row_count"]


def test_to_frame_lists_every_candidate_with_its_kind():
    frame = ae.find_candidate_keys(orders()).to_frame()

    assert list(frame.columns) == [
        "columns",
        "kind",
        "distinct_count",
        "duplicate_row_count",
        "missing_row_count",
        "uniqueness_share",
        "is_minimal",
    ]
    assert frame["columns"].tolist() == [["order", "line"], ["sku"]]
    assert frame["kind"].tolist() == ["key", "key_with_missing_values"]
    assert frame["distinct_count"].dtype == "int64"


@pytest.mark.parametrize("frame", [orders, unkeyed, keyed])
def test_shows_no_raw_data_value(frame):
    # Every value becomes text that no count, share or column name could spell,
    # so finding one in a rendering means a raw value leaked.
    df = frame().astype(str).map(lambda value: f"secret-{value}-zq")

    res = ae.find_candidate_keys(df)

    renderings = [str(res), res.to_json(), res.to_frame().to_string()]
    for value in pd.unique(df.to_numpy().ravel()):
        assert all(value not in rendering for rendering in renderings), value


def test_declares_the_statuses_it_can_return():
    [info] = [i for i in ae.list_analyses().analyses if i.name == "find_candidate_keys"]

    assert info.statuses == [
        "completed",
        "budget_exceeded",
        "insufficient_data",
        "not_applicable",
        "failed",
    ]
    assert info.module == "agent_eda.data.structure"


@given(data=st.data())
def test_every_reported_set_counts_as_pandas_does(data):
    df = data.draw(data_frames(), "data")
    max_key_size = data.draw(st.integers(1, 3), "max_key_size")
    assume(not holds_nul(df))

    res = ae.find_candidate_keys(df, max_key_size=max_key_size)

    labels = {str(label): label for label in df.columns}
    reported = [
        *res.keys,
        *res.keys_with_missing_values,
        *res.most_unique_column_sets,
    ]
    for found in reported:
        columns = [labels[name] for name in found.columns]
        assert (
            found.distinct_count,
            found.duplicate_row_count,
            found.missing_row_count,
        ) == pandas_counts(df, columns)


def holds_nul(df):
    """Whether any text in the frame has a NUL character. pandas' factorize reads
    a NUL as the end of a text, so its counts of such text are no reference."""
    return any(
        isinstance(value, str) and "\x00" in value
        for column in df.columns
        for value in df[column]
    )


def pandas_counts(df, columns):
    """Distinct combinations, rows sharing one and rows with a missing value.

    A constant column joins the subset, so that pandas compares through
    factorize even for one column; Series.duplicated tells None and NaN apart.
    """
    df = nan_as_missing(df)
    keyed_frame = df[columns].assign(_constant=0)
    subset = [*columns, "_constant"]
    return (
        len(keyed_frame.drop_duplicates(subset=subset)),
        int(keyed_frame.duplicated(subset=subset, keep=False).sum()),
        int(df[columns].isna().any(axis=1).sum()),
    )


@given(data=st.data())
def test_the_keys_are_exactly_the_minimal_sets_whose_rows_are_all_different(data):
    # Every column set of up to max_key_size columns, tried one by one with pandas.
    df = data.draw(data_frames(), "data").reset_index(drop=True)
    max_key_size = data.draw(st.integers(1, 3), "max_key_size")

    assume(not holds_nul(df))
    res = ae.find_candidate_keys(df, max_key_size=max_key_size)

    assume(res.status == "completed")  # a search at its limit lists what it reached
    labels = {str(label): label for label in df.columns}
    # columns=None leaves out columns of other dtypes and floats with fractions.
    left_out = {*res.left_out_columns, *res.non_integral_float_columns}
    searched = [name for name in labels if name not in left_out]
    unique_sets: list[tuple[str, ...]] = []
    expected_keys, expected_with_missing = [], []
    for size in range(1, max_key_size + 1):
        for combination in itertools.combinations(searched, size):
            if any(set(found) <= set(combination) for found in unique_sets):
                continue  # not minimal: a smaller set is unique already
            _, duplicates, missing = pandas_counts(
                df, [labels[name] for name in combination]
            )
            if duplicates == 0:
                unique_sets.append(combination)
                (expected_with_missing if missing else expected_keys).append(
                    list(combination)
                )
    assert sorted(key.columns for key in res.keys) == sorted(expected_keys)
    assert sorted(key.columns for key in res.keys_with_missing_values) == sorted(
        expected_with_missing
    )


def test_lists_at_most_10_column_sets():
    once = pd.DataFrame(
        {name: [i % (2 + j) for i in range(12)] for j, name in enumerate("abcde")}
    )
    df = pd.concat([once, once], ignore_index=True)  # every row twice: no key

    singles = ae.find_candidate_keys(df, max_key_size=1)
    pairs = ae.find_candidate_keys(df)

    assert len(singles.most_unique_column_sets) == 5
    # 5 single columns and 7 minimal pairs, such as (b, c) with 12 combinations;
    # (a, c) isn't minimal, since c alone tells a.
    assert pairs.keys == []
    assert len(pairs.most_unique_column_sets) == 10


def test_lists_no_most_unique_sets_when_a_key_exists():
    assert ae.find_candidate_keys(keyed()).most_unique_column_sets == []


def test_a_ranking_that_bounds_prove_has_no_limitation():
    res = ae.find_candidate_keys(unkeyed())

    assert res.most_unique_column_sets
    assert res.limitations == []


def test_says_when_the_work_limit_stops_the_ranking_before_bounds_prove_it():
    # Uniform columns without missing values: nothing but a count on all rows
    # tells how many combinations two of them form.
    rng = np.random.default_rng(0)
    rows = 1_000_000
    df = pd.DataFrame(
        {f"c{i}": rng.integers(0, 1_000 + 50 * i, rows) for i in range(16)}
    )

    res = ae.find_candidate_keys(df)

    assert res.keys == []
    # Pairs take hundreds of thousands of combinations, single columns about a
    # thousand, so every set counted before the limit outranks every column.
    assert [len(s.columns) for s in res.most_unique_column_sets] == [2] * 10
    [limitation] = res.limitations
    assert limitation.code == "ranking_not_proven"
    assert limitation.fields == ["most_unique_column_sets"]
    counted = re.fullmatch(
        r"Counted (\d+) of 120 sets on all rows before the work limit; an "
        r"uncounted set could rank higher\.",
        limitation.reason,
    )
    assert counted
    assert 0 < int(counted[1]) < 120
    best = res.most_unique_column_sets[0]
    assert (
        best.distinct_count,
        best.duplicate_row_count,
        best.missing_row_count,
    ) == pandas_counts(df, best.columns)


def test_keeps_a_key_of_three_columns_only_when_no_pair_inside_it_is_a_key():
    df = pd.DataFrame(
        {
            "x": [0, 0, 0, 0, 1, 1, 1, 1],
            "y": [0, 0, 1, 1, 0, 0, 1, 1],
            "z": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    res = ae.find_candidate_keys(df, max_key_size=3)

    assert [key.columns for key in res.keys] == [["x", "y", "z"]]
    assert all(key.is_minimal for key in res.keys)


@pytest.mark.parametrize(
    ("rows", "code", "message"),
    [
        (0, "no_rows", "The data has no rows."),
        (
            1,
            "one_row",
            "The data has 1 row, which every column identifies; finding keys "
            "needs at least 2 rows.",
        ),
    ],
)
def test_fewer_than_two_rows_give_insufficient_data(rows, code, message):
    res = ae.find_candidate_keys(keyed().head(rows))

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (code, message)
    assert res.keys == []
    assert res.search.examined_count == 0


def test_data_without_columns_is_not_applicable():
    res = ae.find_candidate_keys(pd.DataFrame(index=range(3)))

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_columns"
    assert res.status_reason.message == "The data has no columns to search."


def test_data_whose_values_cannot_be_hashed_is_not_applicable():
    df = pd.DataFrame({"tags": pd.Series([["a"], ["b"], ["a"]], dtype=object)})

    res = ae.find_candidate_keys(df)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_hashable_columns"
    assert res.status_reason.message == (
        "No selected column can be searched: their values can't be hashed, such "
        "as lists or dicts."
    )
    assert res.left_out_columns == ["tags"]


def test_leaves_out_columns_whose_values_cannot_be_hashed():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            # Text, apart from one list that a subset of rows could miss.
            "tags": pd.Series(["a", "b", ["c"], "d"], dtype=object),
            "nested": pd.Series(
                [[1], [2], [3], [4]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))
            ),
        }
    )

    res = ae.find_candidate_keys(df)

    assert res.left_out_columns == ["tags", "nested"]
    assert [key.columns for key in res.keys] == [["id"]]
    assert res.parameters["columns"] == ["id", "tags", "nested"]
    [limitation] = res.limitations
    assert limitation.code == "unhashable_values"
    assert limitation.fields == ["left_out_columns"]
    assert limitation.reason == (
        "The search left out 2 columns whose values can't be hashed, such as "
        "lists: 'tags' and 'nested'."
    )


def test_a_key_size_beyond_the_column_count_finishes_at_once():
    data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]})

    started = time.perf_counter()
    res = ae.find_candidate_keys(data, max_key_size=sys.maxsize)

    assert time.perf_counter() - started < 1
    assert res.status == "completed"
    assert res.exhaustive is True
    assert [key.columns for key in res.keys] == [["a"]]
    # 2 single columns and 1 pair, whatever max_key_size says.
    assert res.search.combination_count == 3


def test_a_budget_stop_on_many_columns_suggests_the_key_size_that_fits():
    data = pd.DataFrame(np.arange(40 * 3).reshape(3, 40) % 2).add_prefix("c")

    res = ae.find_candidate_keys(data, max_key_size=40)

    assert res.status == "budget_exceeded"
    assert res.status_reason is not None
    # 40 + 780 sets of up to 2 columns fit the 10,000; 9,880 more of 3 don't.
    assert res.status_reason.message == (
        "The complete search covers up to 1,099,511,627,775 column sets, more than "
        "the max_combinations limit of 10,000 allows; it stopped after examining "
        "10,000 and kept what it found. Lower max_key_size to 2, or pass fewer "
        "columns."
    )


def test_a_budget_stop_suggests_raising_max_combinations_when_that_is_feasible():
    res = ae.find_candidate_keys(keyed(), max_combinations=5)

    assert res.status_reason is not None
    assert res.status_reason.message.endswith(
        "Lower max_key_size to 1, raise max_combinations to 10, or pass fewer columns."
    )


def test_a_budget_stop_among_single_columns_never_suggests_a_key_size():
    data = pd.DataFrame(np.arange(6).reshape(2, 3) % 2).add_prefix("c")

    res = ae.find_candidate_keys(data, max_key_size=1, max_combinations=2)

    assert res.status_reason is not None
    assert "max_key_size" not in res.status_reason.message
    assert res.status_reason.message.endswith(
        "Raise max_combinations to 3, or pass fewer columns."
    )


def floats():
    return pd.DataFrame(
        {
            "reading": [0.5, 1.25, 2.75, 3.5],
            "count": [1.0, 2.0, 3.0, 4.0],
            "group": [1, 1, 2, 2],
        }
    )


def test_leaves_out_non_integral_float_columns_by_default_and_names_them():
    res = ae.find_candidate_keys(floats())

    assert [key.columns for key in res.keys] == [["count"]]
    assert res.parameters["columns"] == ["count", "group"]
    assert res.non_integral_float_columns == ["reading"]
    (limitation,) = res.limitations
    assert limitation.code == "non_integral_float_columns"
    assert limitation.fields == ["non_integral_float_columns"]
    assert limitation.reason == (
        "The search left out 1 column holding non-integral floats, which are "
        "measurements rather than IDs: 'reading'. Name it in columns to search it."
    )
    assert "reading" in str(res)


def test_naming_a_float_column_in_columns_searches_it():
    res = ae.find_candidate_keys(floats(), columns=["reading", "group"])

    assert [key.columns for key in res.keys] == [["reading"]]
    assert res.non_integral_float_columns == []
    assert res.limitations == []


def test_float_columns_of_whole_numbers_stay_candidates():
    data = pd.DataFrame(
        {
            "plain": [1.0, 2.0, 3.0],
            "gaps": [1.0, np.nan, 3.0],
            "wide": pd.array([1.0, 2.0, None], dtype="Float64"),
            "arrow": pd.array([1.0, 2.0, 3.0], dtype=pd.ArrowDtype(pa.float32())),
            "all_missing": [np.nan, np.nan, np.nan],
        }
    )

    res = ae.find_candidate_keys(data)

    assert res.non_integral_float_columns == []
    assert res.parameters["columns"] == list(data.columns)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([0.5, 1.5, 2.5], dtype="float32"),
        pd.Series([0.5, 1.5, None], dtype="Float64"),
        pd.Series([0.5, 1.5, 2.5], dtype=pd.ArrowDtype(pa.float64())),
        pd.Series([1.0, np.inf, 2.5]),
    ],
    ids=["float32", "nullable", "arrow", "with_infinity"],
)
def test_every_float_dtype_with_a_fraction_is_left_out(values):
    data = pd.DataFrame({"x": values, "n": [1, 2, 3]})

    res = ae.find_candidate_keys(data)

    assert res.non_integral_float_columns == ["x"]


def test_data_of_only_fraction_floats_is_not_applicable_and_says_why():
    data = pd.DataFrame({"x": [0.5, 1.5, 2.5], "y": [0.25, 0.5, 0.75]})

    res = ae.find_candidate_keys(data)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert "non-integral floats" in res.status_reason.message
    assert res.non_integral_float_columns == ["x", "y"]
    named = ae.find_candidate_keys(data, columns=["x"])
    assert [key.columns for key in named.keys] == [["x"]]


def test_the_support_lists_only_the_columns_searched():
    res = ae.find_candidate_keys(floats())

    assert [c.column for c in res.support.columns] == ["count", "group"]
    assert res.support.column_count == 2


def test_the_docstring_describes_the_float_rule_but_not_a_pandas_equivalence():
    doc = str(ae.find_candidate_keys.__doc__)

    assert "non-integral float" in doc
    assert "DataFrame.duplicated" not in doc


def test_a_key_of_five_columns_with_thousands_of_levels_each_counts_as_pandas_does():
    # Five columns of 7,200 levels give 7,200**5 combinations, more than the
    # 2**62 an int64 holds, so the combination codes must be renumbered on the way.
    # Rows 2p and 2p + 1 agree everywhere except in column p % 5, so every set of
    # four columns repeats a row and only all five identify the rows.
    pairs = np.arange(6_000)
    columns = {}
    for position in range(5):
        first = pairs
        second = np.where(pairs % 5 == position, pairs + 6_000, pairs)
        columns[f"c{position}"] = np.column_stack([first, second]).ravel()
    df = pd.DataFrame(columns)

    res = ae.find_candidate_keys(df, max_key_size=5, max_combinations=100)

    assert [key.columns for key in res.keys] == [list(df.columns)]
    [key] = res.keys
    assert (
        key.distinct_count,
        key.duplicate_row_count,
        key.missing_row_count,
    ) == pandas_counts(df, list(df.columns))
    assert key.duplicate_row_count == 0
    for found in res.most_unique_column_sets:
        assert (found.distinct_count, found.duplicate_row_count) == pandas_counts(
            df, found.columns
        )[:2]
