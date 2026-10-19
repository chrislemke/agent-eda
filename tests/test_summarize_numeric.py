import json
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import stats

import agent_eda as ae

INF = np.inf

# The same values in every numeric dtype family; None becomes NaN, NA or null.
COUNT_CASES = {
    "float64": pd.Series([1.5, -2.0, 0.0, None, INF, -INF, 0.0, 3.0]),
    "float32": pd.Series([1.5, -2.0, 0.0, None, INF, -INF, 0.0, 3.0], dtype="float32"),
    "Float64": pd.Series([1.5, -2.0, 0.0, None, INF, -INF, 0.0, 3.0], dtype="Float64"),
    "double[pyarrow]": pd.Series(
        [1.5, -2.0, 0.0, None, INF, -INF, 0.0, 3.0], dtype="double[pyarrow]"
    ),
    "int64": pd.Series([1, -2, 0, 7, 0, 3, -5, 4]),
    "Int64": pd.Series([1, -2, 0, None, 0, 3, None, 4], dtype="Int64"),
    "int64[pyarrow]": pd.Series(
        [1, -2, 0, None, 0, 3, None, 4], dtype="int64[pyarrow]"
    ),
    "UInt8": pd.Series([1, 2, 0, None, 0, 3, None, 4], dtype="UInt8"),
}

RNG = np.random.default_rng(20261017)
# Samples for the reference checks: none is constant or nearly constant, where
# SciPy's moments warn about precision loss.
REFERENCE_SAMPLES = {
    # The example in SciPy's skew docstring.
    "scipy_example": np.array([2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]),
    "normal": RNG.normal(10.0, 3.0, size=1_000),
    "lognormal": RNG.lognormal(0.0, 1.0, size=1_001),
    "integers_with_ties": RNG.integers(0, 5, size=200).astype(float),
    "large_offset": 1e6 + RNG.normal(0.0, 1.0, size=500),
    "four_values": np.array([3.0, -1.0, 4.0, 1.5]),
}
# Relative tolerance against SciPy, NumPy and pandas: the same formulas, summed
# in a different order.
RTOL = 1e-12
DEFAULT_QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "trimmed_mean",
    "mad",
    "mad_scaled",
    "skewness",
    "excess_kurtosis",
]
# Every method numpy.quantile offers, on numpy 1.26 and 2.x alike.
QUANTILE_METHODS = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


def reference_frame():
    """Every sample as a column, padded with missing and infinite values."""
    longest = max(len(x) for x in REFERENCE_SAMPLES.values())
    padded = {
        name: np.concatenate(
            [x, [np.nan, INF, -INF], np.full(longest - len(x), np.nan)]
        )
        for name, x in REFERENCE_SAMPLES.items()
    }
    return pd.DataFrame(padded)


def mixed_frame():
    return pd.DataFrame(
        {
            "flag": [True, False, True],
            "b": [1.5, 2.5, 3.5],
            "city": ["Lyon", "Oslo", "Rome"],
            "a": [3, 1, 2],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "n": pd.array([1, None, 3], dtype="Int64"),
            "f32": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        }
    )


def test_default_selects_numeric_columns_except_booleans_in_dataframe_order():
    res = ae.summarize_numeric(mixed_frame())

    assert list(res.columns) == ["b", "a", "n", "f32"]
    assert res.parameters["columns"] == ["b", "a", "n", "f32"]


def test_explicit_columns_keep_the_callers_order():
    res = ae.summarize_numeric(mixed_frame(), columns=["n", "b"])

    assert list(res.columns) == ["n", "b"]
    assert [entry.column for entry in res.columns.values()] == ["n", "b"]
    assert res.parameters["columns"] == ["n", "b"]


def test_a_boolean_column_named_in_columns_is_read_as_0_and_1():
    df = pd.DataFrame(
        {
            "member": [True, False, True, True],
            "flag": pd.array([True, None, False, False], dtype="boolean"),
            "arrow": pd.array([None, True, True, True], dtype="bool[pyarrow]"),
        }
    )

    res = ae.summarize_numeric(df, columns=["member", "flag", "arrow"])

    member, flag, arrow = res.columns.values()
    assert (member.mean, member.zero_count, member.max) == (0.75, 1, 1.0)
    assert (flag.non_missing_count, flag.missing_count, flag.mean) == (3, 1, 1 / 3)
    assert (arrow.min, arrow.max) == (1.0, 1.0)
    [limitation] = [lim for lim in res.limitations if lim.code == "boolean_as_numeric"]
    assert limitation.fields == [
        "columns['member']",
        "columns['flag']",
        "columns['arrow']",
    ]
    assert limitation.reason == (
        "3 boolean columns were read as 0 for False and 1 for True: 'member', "
        "'flag' and 'arrow'."
    )


def test_counts_missing_infinite_zero_and_negative_values():
    df = pd.DataFrame({"x": [1.5, -2.0, 0.0, None, INF, -INF, 0.0, 3.0]})

    entry = ae.summarize_numeric(df).columns["x"]

    # -inf is negative and infinite; count includes the infinite values.
    assert (
        entry.non_missing_count,
        entry.missing_count,
        entry.infinite_count,
        entry.zero_count,
        entry.negative_count,
    ) == (7, 1, 2, 2, 2)


@pytest.mark.parametrize("dtype", list(COUNT_CASES))
def test_counts_match_pandas_for_every_numeric_dtype(dtype):
    values = COUNT_CASES[dtype]
    df = pd.DataFrame({"x": values})

    entry = ae.summarize_numeric(df).columns["x"]

    assert entry.non_missing_count == values.count()
    assert entry.missing_count == values.isna().sum()
    assert entry.infinite_count == values.isin([INF, -INF]).sum()
    assert entry.zero_count == (values == 0).sum()
    assert entry.negative_count == (values < 0).sum()


INTEGERS = [3, None, -1, 7, 2, 2, 0]
FLOATS = [0.5, None, INF, -2.25, 3.0, 3.0, 0.0]
DECIMALS = [0.5, None, 12.75, -2.25, 3.0, 3.0, 0.0]


@pytest.mark.parametrize(
    ("values", "numpy_values"),
    [
        (pd.Series(INTEGERS, dtype="Int64"), pd.Series(INTEGERS, dtype="float64")),
        (pd.Series(INTEGERS, dtype="Int8"), pd.Series(INTEGERS, dtype="float64")),
        (
            pd.Series(INTEGERS, dtype="int64[pyarrow]"),
            pd.Series(INTEGERS, dtype="float64"),
        ),
        (
            pd.Series(
                [abs(v) if v is not None else v for v in INTEGERS], dtype="UInt16"
            ),
            pd.Series(
                [abs(v) if v is not None else v for v in INTEGERS], dtype="float64"
            ),
        ),
        (pd.Series(FLOATS, dtype="Float64"), pd.Series(FLOATS, dtype="float64")),
        (pd.Series(FLOATS, dtype="Float32"), pd.Series(FLOATS, dtype="float32")),
        (
            pd.Series(FLOATS, dtype="double[pyarrow]"),
            pd.Series(FLOATS, dtype="float64"),
        ),
        (
            pd.Series(FLOATS, dtype=pd.ArrowDtype(pa.float32())),
            pd.Series(FLOATS, dtype="float32"),
        ),
        (
            pd.Series([2, 7, 1, 8, 2, 8, 1], dtype="int64[pyarrow]"),
            pd.Series([2, 7, 1, 8, 2, 8, 1], dtype="int64"),
        ),
        (
            pd.Series(
                [Decimal(v) if v is not None else None for v in DECIMALS],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            ),
            pd.Series(DECIMALS, dtype="float64"),
        ),
    ],
    ids=[
        "Int64",
        "Int8",
        "int64[pyarrow]",
        "UInt16",
        "Float64",
        "Float32",
        "double[pyarrow]",
        "float[pyarrow]",
        "int64[pyarrow]-without-nulls",
        "decimal128[pyarrow]",
    ],
)
def test_nullable_and_pyarrow_columns_give_the_numbers_of_their_numpy_equivalent(
    values, numpy_values
):
    res = ae.summarize_numeric(pd.DataFrame({"x": values}))
    numpy_res = ae.summarize_numeric(pd.DataFrame({"x": numpy_values}))

    assert res.to_json() == numpy_res.to_json()


def test_nan_stored_as_a_value_counts_as_missing_with_a_limitation():
    # pyarrow keeps NaN apart from null, and pandas isna() counts only the null.
    stored = pa.array([1.5, None, float("nan"), 4.0, 2.0], from_pandas=False)
    values = pd.Series(pd.arrays.ArrowExtensionArray(stored))

    res = ae.summarize_numeric(pd.DataFrame({"x": values}))

    entry = res.columns["x"]
    assert values.isna().sum() == 1
    assert (entry.non_missing_count, entry.missing_count, entry.mean) == (3, 2, 2.5)
    [limitation] = [
        lim for lim in res.limitations if lim.code == "nan_counted_as_missing"
    ]
    assert limitation.fields == ["columns['x'].missing_count"]
    assert limitation.reason == (
        "1 column stores NaN as a value, which pandas isna() doesn't count as "
        "missing; missing_count counts it: 'x'."
    )


def test_mean_std_and_extremes_of_a_small_column():
    df = pd.DataFrame({"x": [2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]})

    entry = ae.summarize_numeric(df).columns["x"]

    assert entry.mean == 4.125
    # The squared deviations from 4.125 sum to 110.875; ddof 1 divides by 7.
    assert entry.std == pytest.approx((110.875 / 7) ** 0.5, rel=RTOL)
    assert (entry.min, entry.max) == (0.0, 9.0)


@pytest.mark.parametrize("name", list(REFERENCE_SAMPLES))
def test_mean_std_and_extremes_match_numpy_and_pandas_on_finite_values(name):
    x = REFERENCE_SAMPLES[name]

    entry = ae.summarize_numeric(reference_frame(), columns=[name]).columns[name]

    assert entry.mean == pytest.approx(np.mean(x), rel=RTOL)
    assert entry.mean == pytest.approx(pd.Series(x).mean(), rel=RTOL)
    assert entry.std == pytest.approx(np.std(x, ddof=1), rel=RTOL)
    assert entry.std == pytest.approx(pd.Series(x).std(), rel=RTOL)
    assert (entry.min, entry.max) == (x.min(), x.max())


def test_quantiles_default_to_seven_levels_with_the_linear_method():
    df = pd.DataFrame({"x": np.arange(101.0)})

    res = ae.summarize_numeric(df)

    assert res.parameters["quantiles"] == DEFAULT_QUANTILES
    assert res.parameters["quantile_method"] == "linear"
    quantiles = res.columns["x"].quantiles
    assert list(quantiles) == DEFAULT_QUANTILES
    assert list(quantiles.values()) == pytest.approx([1, 5, 25, 50, 75, 95, 99])


def test_quantiles_keep_the_callers_levels_and_order():
    df = pd.DataFrame({"x": [4.0, 1.0, 3.0, 2.0]})

    res = ae.summarize_numeric(df, quantiles=[0.9, 0, 1, 0.5])

    assert res.parameters["quantiles"] == [0.9, 0.0, 1.0, 0.5]
    quantiles = res.columns["x"].quantiles
    assert list(quantiles) == [0.9, 0.0, 1.0, 0.5]
    assert list(quantiles.values()) == pytest.approx([3.7, 1.0, 4.0, 2.5])


def test_quantiles_take_numpy_arrays_scalars_and_integers():
    df = pd.DataFrame({"x": [4.0, 1.0, 3.0, 2.0]})

    # NumPy arrays and scalars aren't Sequence[float], but are accepted at runtime.
    from_array = ae.summarize_numeric(df, quantiles=np.linspace(0, 1, 3))  # ty: ignore[invalid-argument-type]  # see above
    from_scalar = ae.summarize_numeric(df, quantiles=np.float64(0.5))  # ty: ignore[invalid-argument-type]  # see above
    from_integers = ae.summarize_numeric(df, quantiles=[np.int64(0), np.int64(1)])  # ty: ignore[invalid-argument-type]  # see above

    assert from_array.columns["x"].quantiles == {0.0: 1.0, 0.5: 2.5, 1.0: 4.0}
    assert from_scalar.parameters["quantiles"] == [0.5]
    assert from_integers.columns["x"].quantiles == {0.0: 1.0, 1.0: 4.0}


@pytest.mark.parametrize("method", QUANTILE_METHODS)
@pytest.mark.parametrize("name", list(REFERENCE_SAMPLES))
def test_quantiles_equal_numpys_for_every_method(name, method):
    x = REFERENCE_SAMPLES[name]
    levels = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]

    entry = ae.summarize_numeric(
        reference_frame(), columns=[name], quantiles=levels, quantile_method=method
    ).columns[name]

    # The same numpy.quantile call on the same values: exact, not approximate.
    expected = np.quantile(x, levels, method=method)
    assert list(entry.quantiles.values()) == expected.tolist()


@pytest.mark.parametrize("name", list(REFERENCE_SAMPLES))
def test_median_equals_numpys_whatever_the_quantile_method(name):
    x = REFERENCE_SAMPLES[name]

    res = ae.summarize_numeric(
        reference_frame(), columns=[name], quantile_method="lower"
    )

    assert res.columns[name].median == np.median(x)


def test_median_of_an_even_count_is_the_mean_of_the_middle_values():
    df = pd.DataFrame({"x": [7.0, 1.0, 4.0, 2.0]})

    assert ae.summarize_numeric(df).columns["x"].median == 3.0


def test_trimmed_mean_cuts_a_tenth_from_each_end_by_default():
    # SciPy's trim_mean examples: int(0.1 * 10) = 1 value cut from each end.
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100]})

    res = ae.summarize_numeric(df)

    assert res.parameters["trim_fraction"] == 0.1
    assert res.columns["x"].trimmed_mean == 5.5


def test_trim_fraction_sets_how_many_values_are_cut():
    # 20 values with 0.15: int(3.0) = 3 cut from each end, leaving 3 to 16.
    x = np.r_[np.arange(17.0), 1000, 2000, 3000]
    df = pd.DataFrame({"x": x})

    assert ae.summarize_numeric(df, trim_fraction=0.15).columns["x"].trimmed_mean == 9.5
    assert ae.summarize_numeric(df, trim_fraction=0).columns["x"].trimmed_mean == (
        pytest.approx(x.mean(), rel=RTOL)
    )


@pytest.mark.parametrize("trim_fraction", [0.0, 0.1, 0.25, 0.49])
@pytest.mark.parametrize("name", list(REFERENCE_SAMPLES))
def test_trimmed_mean_matches_scipy(name, trim_fraction):
    x = REFERENCE_SAMPLES[name]

    entry = ae.summarize_numeric(
        reference_frame(), columns=[name], trim_fraction=trim_fraction
    ).columns[name]

    assert entry.trimmed_mean == pytest.approx(
        stats.trim_mean(x, trim_fraction), rel=RTOL
    )


def test_mad_is_raw_and_mad_scaled_multiplies_it_by_1_4826():
    # The median is 3; the absolute deviations 2, 1, 0, 1 and 97 have median 1.
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

    entry = ae.summarize_numeric(df).columns["x"]

    assert entry.mad == 1.0
    assert entry.mad_scaled == 1.4826


@pytest.mark.parametrize("name", list(REFERENCE_SAMPLES))
def test_mad_matches_scipy(name):
    x = REFERENCE_SAMPLES[name]

    entry = ae.summarize_numeric(reference_frame(), columns=[name]).columns[name]

    assert entry.mad == pytest.approx(stats.median_abs_deviation(x), rel=RTOL)
    # SciPy divides by its scale, so 1 / 1.4826 multiplies by 1.4826.
    assert entry.mad_scaled == pytest.approx(
        stats.median_abs_deviation(x, scale=1 / 1.4826), rel=RTOL
    )


def test_skewness_and_excess_kurtosis_of_scipys_example():
    # scipy.stats.skew's docstring example, bias-corrected (bias=False).
    df = pd.DataFrame({"x": [2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]})

    entry = ae.summarize_numeric(df).columns["x"]

    assert entry.skewness == pytest.approx(0.3305821804079746, rel=RTOL)
    assert entry.excess_kurtosis == pytest.approx(-2.098602258096087, rel=RTOL)


@pytest.mark.parametrize("name", list(REFERENCE_SAMPLES))
def test_skewness_and_excess_kurtosis_match_scipy_and_pandas(name):
    x = REFERENCE_SAMPLES[name]
    # Subtracting a constant changes neither statistic, and SciPy and pandas
    # keep all their digits only for data centred near zero: on large_offset
    # they lose 5e-9 of the exact value, which agent-eda keeps to 1e-14.
    centred = x - np.median(x)

    entry = ae.summarize_numeric(reference_frame(), columns=[name]).columns[name]

    assert entry.skewness == pytest.approx(stats.skew(centred, bias=False), rel=RTOL)
    assert entry.skewness == pytest.approx(pd.Series(centred).skew(), rel=RTOL)
    assert entry.excess_kurtosis == pytest.approx(
        stats.kurtosis(centred, fisher=True, bias=False), rel=RTOL
    )
    assert entry.excess_kurtosis == pytest.approx(pd.Series(centred).kurt(), rel=RTOL)


def test_skewness_and_excess_kurtosis_of_offset_data_match_exact_arithmetic():
    x = REFERENCE_SAMPLES["large_offset"]

    entry = ae.summarize_numeric(pd.DataFrame({"x": x})).columns["x"]

    exact = exact_moments(x)
    assert entry.skewness == pytest.approx(exact["skewness"], rel=RTOL)
    assert entry.excess_kurtosis == pytest.approx(exact["excess_kurtosis"], rel=RTOL)
    assert entry.std == pytest.approx(exact["std"], rel=RTOL)


TOO_FEW = (
    "Every statistic needs 1 finite value, std 2, skewness 3 and excess_kurtosis 4."
)


@pytest.mark.parametrize(
    ("values", "nulls"),
    [
        ([5.0], ["std", "skewness", "excess_kurtosis"]),
        ([5.0, 7.0, None, INF], ["skewness", "excess_kurtosis"]),
        ([5.0, 7.0, 6.0], ["excess_kurtosis"]),
    ],
    ids=["one-value", "two-finite-values", "three-values"],
)
def test_statistics_needing_more_values_are_null_with_a_limitation(values, nulls):
    df = pd.DataFrame({"x": values})

    res = ae.summarize_numeric(df)

    entry = res.columns["x"]
    assert [name for name in STATISTICS if getattr(entry, name) is None] == nulls
    [limitation] = res.limitations
    assert limitation.code == "too_few_values"
    assert limitation.fields == [f"columns['x'].{name}" for name in nulls]
    assert limitation.reason == (
        f"Too few finite values left some statistics null in 1 column: 'x'. {TOO_FEW}"
    )


def test_a_column_without_finite_values_has_only_null_statistics():
    df = pd.DataFrame({"x": [None, INF, -INF], "y": [1.0, 2.0, 3.0]})

    res = ae.summarize_numeric(df)

    entry = res.columns["x"]
    assert [getattr(entry, name) for name in STATISTICS] == [None] * len(STATISTICS)
    assert entry.quantiles == dict.fromkeys(DEFAULT_QUANTILES)
    [limitation] = res.limitations
    assert limitation.fields == [
        *(f"columns['x'].{name}" for name in STATISTICS),
        "columns['x'].quantiles",
        "columns['y'].excess_kurtosis",
    ]
    assert limitation.reason == (
        "Too few finite values left some statistics null in 2 columns: 'x' and "
        f"'y'. {TOO_FEW}"
    )


def test_skewness_and_kurtosis_of_equal_values_are_null_with_a_limitation():
    df = pd.DataFrame({"same": [3.0] * 5, "three_same": [2.0, 2.0, 2.0, None, None]})

    res = ae.summarize_numeric(df)

    same = res.columns["same"]
    assert (same.std, same.mad, same.skewness, same.excess_kurtosis) == (
        0.0,
        0.0,
        None,
        None,
    )
    too_few, zero_variance = res.limitations
    assert zero_variance.code == "zero_variance"
    assert zero_variance.fields == [
        "columns['same'].skewness",
        "columns['same'].excess_kurtosis",
        "columns['three_same'].skewness",
    ]
    assert zero_variance.reason == (
        "Every finite value is equal in 2 columns, so skewness and excess_kurtosis "
        "are undefined and null: 'same' and 'three_same'."
    )
    assert too_few.code == "too_few_values"
    assert too_few.fields == ["columns['three_same'].excess_kurtosis"]


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1, 2**53 + 1]),
        pd.Series([-(2**53) - 1, 5]),
        pd.Series([1, 2**64 - 1], dtype="uint64"),
        pd.Series([1, None, 2**60], dtype="Int64"),
        pd.Series([1, None, 2**60], dtype="int64[pyarrow]"),
    ],
    ids=["int64", "negative", "uint64", "Int64", "int64[pyarrow]"],
)
def test_integers_beyond_2_to_the_53_add_a_precision_limitation(values):
    df = pd.DataFrame({"id": values, "small": [1, 2, 3][: len(values)]})

    res = ae.summarize_numeric(df)

    precision = [lim for lim in res.limitations if lim.code == "float_precision"]
    assert len(precision) == 1
    assert precision[0].fields == ["columns['id']"]
    assert precision[0].reason == (
        "1 column holds integers or decimals beyond ±2^53 = 9,007,199,254,740,992, which "
        "float64 can't represent exactly, so its statistics may be rounded: 'id'."
    )


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1, 2**53, -(2**53)]),
        pd.Series([None, None], dtype="Int64"),
        pd.Series([1e17, 2.5]),
    ],
    ids=["exactly-2-to-the-53", "all-missing", "floats"],
)
def test_integers_within_float_precision_add_no_precision_limitation(values):
    res = ae.summarize_numeric(pd.DataFrame({"x": values}))

    assert "float_precision" not in [lim.code for lim in res.limitations]


def test_flags_constant_columns():
    df = pd.DataFrame(
        {
            "same": [2.0, 2.0, None, 2.0],
            "single": [None, 5.0, None, None],
            "varied": [1.0, 2.0, 3.0, 4.0],
            "zeros": [0.0, -0.0, 0.0, None],
        }
    )

    res = ae.summarize_numeric(df)

    [finding] = [f for f in res.findings if f.code == "constant_columns"]
    assert finding.assessment == "suspected"
    assert finding.columns == ["same", "zeros"]
    assert finding.fields == [
        "columns['same'].min",
        "columns['same'].max",
        "columns['zeros'].min",
        "columns['zeros'].max",
    ]
    assert finding.claim == (
        "2 columns hold a single distinct non-missing value: 'same' and 'zeros'."
    )


def test_constant_columns_are_the_ones_describe_dataset_flags():
    df = pd.DataFrame(
        {
            "same": [2.0, 2.0, None, 2.0],
            "single": [None, 5.0, None, None],
            "infinite": [INF, INF, None, INF],
            "mixed": [5.0, 5.0, INF, 5.0],
            "missing": [None, None, None, None],
            "zeros": [0.0, -0.0, 0.0, None],
            "integers": pd.Series([7, 7, None, 7], dtype="Int64"),
        }
    )

    flagged = [
        f.columns
        for f in ae.summarize_numeric(df).findings
        if f.code == "constant_columns"
    ]
    described = [
        f.columns
        for f in ae.describe_dataset(df).findings
        if f.code == "constant_columns"
    ]

    assert flagged == described == [["same", "infinite", "zeros", "integers"]]


def test_flags_infinite_values():
    df = pd.DataFrame(
        {"a": [1.0, INF, 2.0], "b": [3.0, 4.0, 5.0], "c": [-INF, None, -INF]}
    )

    res = ae.summarize_numeric(df)

    [finding] = [f for f in res.findings if f.code == "infinite_values"]
    assert finding.assessment == "suspected"
    assert finding.columns == ["a", "c"]
    assert finding.fields == [
        "columns['a'].infinite_count",
        "columns['c'].infinite_count",
    ]
    assert finding.claim == (
        "2 columns hold infinite values, which every statistic leaves out: 'a' and 'c'."
    )


def test_statistics_beyond_float64_are_null_with_a_limitation_and_no_warning():
    huge = 1.7e308  # float64's largest value is about 1.797e308
    df = pd.DataFrame({"big": [-huge, huge, huge, -huge], "small": [1.0, 2, 3, 4]})

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        res = ae.summarize_numeric(df)

    assert seen == []
    big = res.columns["big"]
    # The standard deviation, 1.963e308, and 1.4826 x the MAD, 2.52e308, don't
    # fit in float64; everything else does, even where a naive sum overflows.
    assert (big.std, big.mad_scaled) == (None, None)
    assert (big.mean, big.median, big.trimmed_mean, big.mad) == (0.0, 0.0, 0.0, huge)
    assert big.quantiles[0.5] == 0.0
    assert big.skewness == 0.0
    assert big.excess_kurtosis == pytest.approx(-6.0, rel=RTOL)
    [limitation] = [lim for lim in res.limitations if lim.code == "numerical_warning"]
    assert limitation.fields == ["columns['big'].std", "columns['big'].mad_scaled"]
    assert limitation.reason.startswith(
        "Some statistics overflowed float64 or failed numerically, so they are "
        "null in 1 column: 'big'. Warnings: "
    )


def test_moments_stay_exact_for_large_values_close_together():
    # Exactly representable: 1e15 + small integers. Shifting changes no moment.
    offsets = [0, 1, 2, 3, 10]
    df = pd.DataFrame({"x": [1e15 + v for v in offsets]})

    entry = ae.summarize_numeric(df).columns["x"]

    exact = exact_moments(offsets)
    assert entry.mean == 1e15 + 3.2
    assert entry.std == pytest.approx(exact["std"], rel=RTOL)
    assert entry.skewness == pytest.approx(exact["skewness"], rel=RTOL)
    assert entry.excess_kurtosis == pytest.approx(exact["excess_kurtosis"], rel=RTOL)


def exact_moments(values):
    """G1, G2 and the standard deviation in exact rational arithmetic."""
    x = [Fraction(float(v)) for v in values]
    n = len(x)
    mean = sum(x) / n
    m2, m3, m4 = (sum((v - mean) ** k for v in x) / n for k in (2, 3, 4))
    g1 = float(m3) / float(m2) ** 1.5
    g2 = float(m4 / m2**2) - 3
    return {
        "std": float(m2 * n / (n - 1)) ** 0.5,
        "skewness": (n * (n - 1)) ** 0.5 / (n - 2) * g1,
        "excess_kurtosis": (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6),
    }


NO_FINITE_VALUES = "There are no finite values to summarize."


def test_declares_its_statuses_and_finding_codes():
    [info] = [a for a in ae.list_analyses().analyses if a.name == "summarize_numeric"]

    assert info.module == "agent_eda.statistics.descriptive"
    assert info.statuses == [
        "completed",
        "partial",
        "insufficient_data",
        "not_applicable",
        "failed",
    ]
    assert info.finding_codes == ["constant_columns", "infinite_values"]
    assert info.parameters == [
        "data",
        "columns",
        "quantiles",
        "quantile_method",
        "trim_fraction",
    ]


def test_every_column_completes_when_each_has_a_finite_value():
    res = ae.summarize_numeric(pd.DataFrame({"a": [1.0, None], "b": [INF, 2.0]}))

    assert res.status == "completed"
    assert res.status_reason is None
    assert [(e.status, e.status_reason) for e in res.columns.values()] == [
        ("completed", None),
        ("completed", None),
    ]


def test_a_column_without_finite_values_makes_the_result_partial():
    df = pd.DataFrame({"a": [1.0, 2.0], "empty": [None, INF], "b": [3.0, 4.0]})

    res = ae.summarize_numeric(df)

    empty = res.columns["empty"]
    assert empty.status == "insufficient_data"
    assert empty.status_reason is not None
    assert (empty.status_reason.code, empty.status_reason.message) == (
        "no_finite_values",
        NO_FINITE_VALUES,
    )
    assert res.columns["a"].status == "completed"
    assert res.status == "partial"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "incomplete_columns",
        f"1 of 3 columns didn't complete: 'empty'. {NO_FINITE_VALUES}",
    )


def test_no_column_with_finite_values_gives_insufficient_data():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [INF, None], "c": ["x", "y"]})

    res = ae.summarize_numeric(df)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_finite_values",
        f"No column completed: 'a' and 'b'. {NO_FINITE_VALUES}",
    )
    assert [e.status for e in res.columns.values()] == ["insufficient_data"] * 2


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"x": pd.Series([], dtype="float64")}),
        pd.DataFrame(
            {"x": pd.Series([], dtype="Int64"), "t": pd.Series([], dtype=str)}
        ),
        pd.DataFrame({"t": pd.Series([], dtype=object)}),
    ],
    ids=["nothing", "numeric", "numeric-and-text", "text"],
)
def test_data_without_rows_gives_insufficient_data(df):
    res = ae.summarize_numeric(df)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_rows",
        "The data has no rows.",
    )
    assert all(e.status == "insufficient_data" for e in res.columns.values())


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"city": ["Lyon", "Oslo"], "member": [True, False]}),
        pd.DataFrame(index=range(3)),
    ],
    ids=["text-and-booleans", "no-columns"],
)
def test_data_without_numeric_columns_is_not_applicable(df):
    res = ae.summarize_numeric(df)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_numeric_columns",
        "The data has no numeric columns. Booleans count only when named in "
        "columns, which reads them as 0 and 1.",
    )
    assert res.columns == {}
    assert res.parameters["columns"] == []


def test_support_counts_the_values_each_column_left_out():
    df = pd.DataFrame({"x": [1.0, None, INF, 2.0, -INF], "y": [1, 2, 3, 4, 5]})

    support = ae.summarize_numeric(df).support

    assert (support.row_count, support.column_count) == (5, 2)
    assert [
        (
            c.column,
            c.eligible_count,
            c.evaluated_count,
            c.excluded_count,
            c.excluded_by_reason,
        )
        for c in support.columns
    ] == [
        ("x", 5, 2, 3, {"missing": 1, "infinite": 2}),
        ("y", 5, 5, 0, {}),
    ]


def survey():
    return pd.DataFrame(
        {
            "age": [34, 51, 29, 47, None, 38],
            "income": [52.5, 61.0, 48.25, 1e6, 70.0, 55.0],
            "member": [True, False, True, True, False, True],
        }
    )


def test_text_shows_the_main_statistics_of_each_column():
    res = ae.summarize_numeric(survey())

    assert str(res) == "\n".join(
        [
            "summarize_numeric: completed on 6 rows x 3 columns",
            "Statistics use the finite values. Quantile method linear; trimmed_mean "
            "cuts 10% from each end.",
            "Left out, not numeric or boolean: 'member'.",
            "res.to_frame() also has the quantiles, trimmed_mean, mad, mad_scaled, "
            "skewness and excess_kurtosis.",
            "Columns (2 of 2 shown):",
            "column  non_missing_count  missing_count       mean        std    min  median    max",
            "age                     5              1       39.8      9.094     29      38     51",
            "income                  6              0  1.667e+05  4.082e+05  48.25      58  1e+06",
            "Findings: none",
            "Limitations: none",
            "More: res.to_frame() has all 2 columns; res.findings, res.limitations "
            "and res.to_json() have the rest.",
        ]
    )


def test_text_of_named_columns_says_nothing_about_left_out_ones():
    res = ae.summarize_numeric(survey(), columns=["income"], trim_fraction=0.125)

    lines = str(res).splitlines()
    assert lines[1] == (
        "Statistics use the finite values. Quantile method linear; trimmed_mean "
        "cuts 12.5% from each end."
    )
    assert not any(line.startswith("Left out") for line in lines)


def test_text_of_a_partial_result_names_the_incomplete_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "empty": [np.nan, np.nan]})

    lines = str(ae.summarize_numeric(df)).splitlines()

    assert lines[:2] == [
        "summarize_numeric: partial on 2 rows x 2 columns",
        "Reason: incomplete_columns: 1 of 2 columns didn't complete: 'empty'. "
        "There are no finite values to summarize.",
    ]
    assert (
        "empty                   0              2  null    null  null    null  null"
        in lines
    )


def test_text_stays_within_40_lines_of_120_characters():
    df = pd.DataFrame(
        {f"{'a_long_column_name_' * 8}{i}": [1e300, -1e-300, None] for i in range(60)}
    )

    lines = str(ae.summarize_numeric(df)).split("\n")

    assert len(lines) <= 40
    assert max(len(line) for line in lines) <= 120


def test_to_frame_has_every_statistic_of_every_column():
    frame = ae.summarize_numeric(survey(), quantiles=[0.5, 0.9]).to_frame()

    assert list(frame.columns) == [
        "column",
        "status",
        "status_reason",
        "non_missing_count",
        "missing_count",
        "infinite_count",
        "zero_count",
        "negative_count",
        "mean",
        "std",
        "min",
        "max",
        "q0.5",
        "q0.9",
        "median",
        "trimmed_mean",
        "mad",
        "mad_scaled",
        "skewness",
        "excess_kurtosis",
    ]
    assert frame["column"].tolist() == ["age", "income"]
    assert frame["non_missing_count"].tolist() == [5, 6]
    assert frame["q0.5"].tolist() == [38.0, 58.0]
    assert frame["status_reason"].isna().all()


def test_to_frame_types_counts_as_integers_and_statistics_as_floats():
    df = pd.DataFrame({"empty": [np.nan, np.nan], "one": [1.0, np.nan]})

    frame = ae.summarize_numeric(df).to_frame()

    assert frame["non_missing_count"].dtype == "int64"
    assert frame["std"].dtype == "float64"
    assert frame["std"].isna().all()
    assert frame["status_reason"].tolist() == ["no_finite_values", None]


def test_to_frame_keeps_every_column_beyond_the_records_100():
    df = pd.DataFrame({f"c{i}": [float(i), 1.0] for i in range(150)})

    res = ae.summarize_numeric(df)
    record = res.to_dict()

    assert len(res.to_frame()) == 150
    assert list(record["columns"]) == [f"c{i}" for i in range(100)]
    assert record["columns_total"] == 150


def test_the_result_states_each_statistics_definition():
    definitions = ae.summarize_numeric(survey()).definitions

    assert list(definitions) == [
        "non_missing_count",
        "missing_count",
        "infinite_count",
        "zero_count",
        "negative_count",
        "mean",
        "std",
        "min",
        "max",
        "quantiles",
        "median",
        "trimmed_mean",
        "mad",
        "mad_scaled",
        "skewness",
        "excess_kurtosis",
    ]
    assert "ddof 1" in definitions["std"]
    assert "scipy.stats.trim_mean" in definitions["trimmed_mean"]
    assert "1.4826" in definitions["mad_scaled"]
    assert "bias=False" in definitions["skewness"]
    assert "bias=False" in definitions["excess_kurtosis"]


def test_the_record_is_strict_json_without_nan_or_infinity():
    df = pd.DataFrame(
        {
            "empty": [np.nan, np.nan, np.nan, np.nan],
            "infinite": [INF, -INF, INF, None],
            "same": [1.0, 1.0, 1.0, 1.0],
            "one": [None, 2.0, None, None],
        }
    )

    record = ae.summarize_numeric(df).to_json()

    columns = json.loads(record, parse_constant=reject_constant)["columns"]
    assert columns["empty"]["mean"] is None
    assert columns["infinite"]["quantiles"] == dict.fromkeys(
        ["0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99"]
    )
    assert columns["same"]["skewness"] is None


def reject_constant(name):
    raise AssertionError(f"non-strict JSON constant {name}")
