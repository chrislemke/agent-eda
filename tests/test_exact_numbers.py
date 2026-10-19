"""Numbers are decided on the stored values, not on float64 copies of them.

One rule says when a column is constant, the standard deviation of very small
values is exact, decimals beyond 2^53 get a precision limitation like integers,
and NaN that a column stores as a value is missing in every analysis.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import agent_eda as ae

NAN = float("nan")
BIG = 2**54


def stored_nan(values: list[float | None]) -> pd.Series:
    """A double[pyarrow] column that keeps NaN apart from null, as Parquet does."""
    array = pa.array(values, type=pa.float64(), from_pandas=False)
    return pd.Series(pd.arrays.ArrowExtensionArray(array))


def exact_std(values: list[float]) -> float:
    """The sample standard deviation of the stored floats, computed exactly."""
    exact = [Fraction(v) for v in values]
    mean = sum(exact, Fraction(0)) / len(exact)
    variance = sum(((v - mean) ** 2 for v in exact), Fraction(0)) / (len(exact) - 1)
    with localcontext() as context:
        context.prec = 60
        root = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
    return float(root)


def codes(res) -> list[str]:
    return [finding.code for finding in res.findings]


# Distinct integers beyond 2^53 are distinct.


@pytest.fixture
def big_ids() -> pd.DataFrame:
    return pd.DataFrame({"id": [BIG, BIG + 1, BIG + 2]})


def test_summarize_numeric_does_not_call_distinct_big_integers_constant(big_ids):
    assert "constant_columns" not in codes(ae.summarize_numeric(big_ids))


def test_profile_target_does_not_call_distinct_big_integers_a_constant_target(big_ids):
    res = ae.profile_target(big_ids, target_column="id", task="regression")

    assert "constant_target" not in codes(res)


def test_describe_dataset_and_infer_column_semantics_agree_on_big_integers(big_ids):
    assert "constant_columns" not in codes(ae.describe_dataset(big_ids))
    assert "constant" not in ae.infer_column_semantics(big_ids).columns["id"].flags


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([BIG, BIG, BIG]),
        pd.Series([BIG, BIG, None], dtype="Int64"),
        pd.Series([2**64 - 1, 2**64 - 1], dtype="uint64"),
    ],
    ids=["int64", "Int64", "uint64"],
)
def test_equal_big_integers_are_still_constant(values):
    df = pd.DataFrame({"id": values})

    assert "constant_columns" in codes(ae.summarize_numeric(df))
    assert "constant_columns" in codes(ae.describe_dataset(df))
    assert "constant" in ae.infer_column_semantics(df).columns["id"].flags
    target = ae.profile_target(df, target_column="id", task="regression")
    assert "constant_target" in codes(target)


def test_distinct_big_decimals_are_not_constant():
    values = pd.Series(
        [Decimal(2**60 + 1), Decimal(2**60 + 3)],
        dtype=pd.ArrowDtype(pa.decimal128(38, 1)),
    )
    df = pd.DataFrame({"d": values})

    assert "constant_columns" not in codes(ae.summarize_numeric(df))
    assert "constant_columns" not in codes(ae.describe_dataset(df))


def test_infinite_values_of_one_sign_are_constant_and_mixed_ones_are_not():
    same = pd.DataFrame({"x": [np.inf, np.inf, np.nan]})
    mixed = pd.DataFrame({"x": [np.inf, -np.inf, 1.0]})

    assert "constant_columns" in codes(ae.summarize_numeric(same))
    assert "constant_columns" not in codes(ae.summarize_numeric(mixed))


# The standard deviation of very small values.


def normal_values(scale: float) -> list[float]:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(1000) * scale).tolist()


@pytest.mark.parametrize(
    "values",
    [
        [1e-200, 2e-200, 4e-200, 8e-200],
        normal_values(1e-170),
        normal_values(1e-160),
        [1e-160, 2e-160, 3e-160, 5e-160],
        [-1e-160, 1e-160],
    ],
    ids=["1e-200", "normal-1e-170", "normal-1e-160", "1e-160", "pair"],
)
def test_the_std_of_tiny_values_is_exact(values):
    expected = exact_std(values)
    df = pd.DataFrame({"x": values})

    assert expected > 0
    assert ae.summarize_numeric(df).columns["x"].std == pytest.approx(
        expected, rel=1e-12, abs=0
    )
    target = ae.profile_target(df, target_column="x", task="regression")
    assert target.statistics is not None
    assert target.statistics.std == pytest.approx(expected, rel=1e-12, abs=0)


def test_the_std_of_the_audit_example_is_the_exact_value():
    entry = ae.summarize_numeric(
        pd.DataFrame({"x": [1e-200, 2e-200, 4e-200, 8e-200]})
    ).columns["x"]

    assert entry.std == pytest.approx(3.0956959368344516e-200, rel=1e-12, abs=0)


def test_tiny_deviations_around_a_larger_centre_keep_their_shape():
    # Deviations of about 2^-370 have fourth powers that underflow float64, while
    # the values themselves, near 2^-330, are far from the underflow limit.
    unit = [1 + k * 2.0**-40 for k in (0, 1, 2, 3, 7, 8)]
    values = [v * 2.0**-330 for v in unit]

    tiny = ae.summarize_numeric(pd.DataFrame({"x": values})).columns["x"]
    reference = ae.summarize_numeric(pd.DataFrame({"x": unit})).columns["x"]

    assert tiny.std is not None
    assert reference.std is not None
    assert tiny.std == pytest.approx(reference.std * 2.0**-330, rel=1e-9, abs=0)
    assert tiny.skewness == pytest.approx(reference.skewness, rel=1e-9, abs=0)
    assert tiny.excess_kurtosis == pytest.approx(
        reference.excess_kurtosis, rel=1e-9, abs=0
    )


def test_tiny_values_keep_their_shape_statistics():
    values = normal_values(1e-170)
    scaled = [v * 1e170 for v in values]

    tiny = ae.summarize_numeric(pd.DataFrame({"x": values})).columns["x"]
    unit = ae.summarize_numeric(pd.DataFrame({"x": scaled})).columns["x"]

    assert tiny.skewness == pytest.approx(unit.skewness, rel=1e-9, abs=0)
    assert tiny.excess_kurtosis == pytest.approx(unit.excess_kurtosis, rel=1e-9, abs=0)


# Decimals beyond 2^53.


def big_decimals() -> pd.Series:
    values = [
        Decimal(2**60 + 1),
        Decimal(2**60 + 3),
        Decimal(2**60 + 5),
        Decimal("0.1"),
    ]
    return pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(38, 1)))


def precision(res) -> list:
    return [lim for lim in res.limitations if lim.code == "float_precision"]


def test_summarize_numeric_flags_decimals_beyond_2_to_the_53():
    res = ae.summarize_numeric(pd.DataFrame({"d": big_decimals()}))

    [limitation] = precision(res)
    assert limitation.fields == ["columns['d']"]
    assert "'d'" in limitation.reason


def test_profile_target_flags_decimals_beyond_2_to_the_53():
    df = pd.DataFrame({"d": big_decimals()})

    res = ae.profile_target(df, target_column="d", task="regression")

    [limitation] = precision(res)
    assert limitation.fields == ["statistics"]


def test_compute_association_matrix_flags_decimals_beyond_2_to_the_53():
    df = pd.DataFrame({"d": big_decimals(), "y": [1.0, 2.0, 3.0, 5.0]})

    res = ae.compute_association_matrix(df)

    [limitation] = precision(res)
    assert limitation.fields == ["columns['d']"]


def test_decimals_within_2_to_the_53_add_no_precision_limitation():
    values = pd.Series(
        [Decimal("1.5"), Decimal("2.5")], dtype=pd.ArrowDtype(pa.decimal128(10, 1))
    )

    res = ae.summarize_numeric(pd.DataFrame({"d": values}))

    assert precision(res) == []


# NaN stored as a value is missing everywhere.


def lone_nan_frame() -> pd.DataFrame:
    # NaN twice and null once; pandas' isna() counts only the null.
    return pd.DataFrame(
        {"x": stored_nan([1.0, 2.0, NAN, None, 2.0, NAN]), "y": [1, 2, 3, 4, 5, 6]}
    )


def stored_nan_limitations(res) -> list:
    return [lim for lim in res.limitations if lim.code == "nan_counted_as_missing"]


def test_describe_dataset_counts_stored_nan_as_missing():
    res = ae.describe_dataset(lone_nan_frame())

    column = res.columns[0]
    assert (column.missing_count, column.non_missing_count, column.distinct_count) == (
        3,
        3,
        2,
    )
    [limitation] = stored_nan_limitations(res)
    assert limitation.fields == ["columns[0].missing_count"]
    assert "'x'" in limitation.reason


def test_summarize_categorical_counts_stored_nan_as_missing():
    res = ae.summarize_categorical(lone_nan_frame(), columns=["x"])

    entry = res.columns["x"]
    assert (entry.missing_count, entry.non_missing_count, entry.cardinality) == (
        3,
        3,
        2,
    )
    [limitation] = stored_nan_limitations(res)
    assert limitation.fields == ["columns['x'].missing_count"]


def test_profile_target_classes_leave_out_stored_nan():
    res = ae.profile_target(
        lone_nan_frame(), target_column="x", task="multiclass_classification"
    )

    assert res.classes is not None
    assert [c.label for c in res.classes] == [2.0, 1.0]
    assert res.missing_count == 3
    [limitation] = stored_nan_limitations(res)
    assert "'x'" in limitation.reason


def test_find_duplicate_rows_treats_stored_nan_as_missing():
    res = ae.find_duplicate_rows(lone_nan_frame(), columns=["x"])

    # Rows 1 and 4 hold 2.0; rows 2, 3 and 5 hold a missing value, NaN or null.
    assert sorted(group.size for group in res.duplicate_groups) == [2, 3]
    [limitation] = stored_nan_limitations(res)
    assert "'x'" in limitation.reason


def test_find_candidate_keys_reports_a_stored_nan_as_a_missing_value():
    df = pd.DataFrame({"x": stored_nan([1.0, 2.0, NAN, 3.0])})

    res = ae.find_candidate_keys(df)

    assert res.keys == []
    [key] = res.keys_with_missing_values
    assert key.missing_row_count == 1
    [limitation] = stored_nan_limitations(res)
    assert "'x'" in limitation.reason


def test_profile_missing_values_counts_stored_nan_as_missing():
    res = ae.profile_missing_values(lone_nan_frame())

    assert res.columns["x"].native_missing_count == 3
    assert res.columns["x"].missing_count == 3
    [limitation] = stored_nan_limitations(res)
    assert limitation.fields == ["columns['x'].native_missing_count"]


def test_detect_missing_sentinels_counts_stored_nan_as_missing():
    df = pd.DataFrame({"x": stored_nan([-999.0, -999.0, NAN, 1.0, NAN, 2.0])})

    res = ae.detect_missing_sentinels(df)

    [column] = res.support.columns
    assert column.excluded_by_reason == {"missing": 2}
    [limitation] = stored_nan_limitations(res)
    assert "'x'" in limitation.reason


def test_a_column_that_stores_no_nan_has_no_limitation():
    df = pd.DataFrame({"x": stored_nan([1.0, None, 2.0, 2.0])})

    for res in (
        ae.describe_dataset(df),
        ae.summarize_categorical(df, columns=["x"]),
        ae.find_duplicate_rows(df),
        ae.profile_missing_values(df),
    ):
        assert stored_nan_limitations(res) == []


@pytest.mark.parametrize(
    "values",
    [[1.0, 1.0, NAN], [NAN, 5.0, 5.0, None]],
    ids=["one-value-and-nan", "one-value-nan-and-null"],
)
def test_stored_nan_gives_every_analysis_the_same_constancy_answer(values):
    df = pd.DataFrame({"x": stored_nan(values)})
    as_float64 = pd.DataFrame({"x": pd.Series(values, dtype="float64")})

    for frame in (df, as_float64):
        answers = {
            "describe_dataset": "constant_columns" in codes(ae.describe_dataset(frame)),
            "summarize_numeric": "constant_columns"
            in codes(ae.summarize_numeric(frame)),
            "infer_column_semantics": "constant"
            in ae.infer_column_semantics(frame).columns["x"].flags,
            "profile_target": "constant_target"
            in codes(ae.profile_target(frame, target_column="x", task="regression")),
        }
        assert set(answers.values()) == {True}, answers


def test_a_column_of_only_stored_nan_is_empty_in_every_analysis():
    df = pd.DataFrame({"x": stored_nan([NAN, NAN, NAN])})

    describe = ae.describe_dataset(df)

    assert "all_missing_columns" in codes(describe)
    assert "constant_columns" not in codes(describe)
    assert ae.summarize_numeric(df).columns["x"].status == "insufficient_data"
    assert "all_missing" in ae.infer_column_semantics(df).columns["x"].flags
