import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats.contingency import association

import agent_eda as ae

# Absolute tolerance against SciPy: the same definitions, summed in another order.
ATOL = 1e-12

# SciPy's docstring examples for spearmanr and kendalltau, with ties.
TIED_X = [7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4]
TIED_Y = [2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0]
# The example table in SciPy's contingency.association docstring.
SCIPY_TABLE = [[100, 150], [203, 322], [420, 700], [320, 210]]
# The music preferences table of the R package effectsize (reference page
# phi.html): rows Psych, Econ and Law, columns Pop, Rock, Jazz and Classic. Its
# cramers_v() with adjust = TRUE, Bergsma's correction, prints 0.23.
MUSIC_TABLE = [[150, 100, 165, 130], [50, 65, 35, 10], [2, 55, 40, 25]]


def table_frame(table):
    """Rows whose two categorical columns cross-tabulate to the table."""
    rows, columns = [], []
    for i, counts in enumerate(table):
        for j, count in enumerate(counts):
            rows += [f"r{i}"] * count
            columns += [f"c{j}"] * count
    return pd.DataFrame({"row": rows, "column": columns})


def pair(res, x, y):
    """The pair of columns x and y, in either order."""
    [found] = [p for p in res.pairs if {p.x_column, p.y_column} == {x, y}]
    return found


def test_pearson_of_two_numeric_columns_matches_scipy():
    # SciPy's pearsonr docstring example.
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7], "y": [10, 9, 2.5, 6, 4, 3, 2]})

    res = ae.compute_association_matrix(df)

    found = pair(res, "x", "y")
    assert found.method == "pearson"
    assert found.value == pytest.approx(
        stats.pearsonr(df["x"], df["y"]).statistic, abs=ATOL
    )
    assert found.value == pytest.approx(-0.828503883588428, abs=ATOL)
    assert found.pair_count == 7


def mixed_frame():
    return pd.DataFrame(
        {
            "age": [34, 51, 29, 47, 38, 62],
            "region": ["north", "south", "north", "east", "south", "east"],
            "zip": [1, 2, 1, 3, 2, 3],
            "income": [52.5, 61.0, 48.25, 70.0, 55.0, 80.5],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
        }
    )


def test_records_every_parameter_with_its_default():
    res = ae.compute_association_matrix(mixed_frame())

    assert res.parameters == {
        "data": {"row_count": 6, "column_count": 5},
        "columns": ["age", "region", "zip", "income"],
        "categorical_columns": [],
        "numeric_method": "pearson",
        "categorical_method": "cramers_v_bias_corrected",
        "max_categories": 50,
        "max_pairs": 5000,
    }


def test_records_the_parameters_as_given():
    res = ae.compute_association_matrix(
        mixed_frame(),
        columns=["income", "zip", "region"],
        categorical_columns=["zip"],
        numeric_method="kendall",
        categorical_method="cramers_v",
        max_categories=10,
        max_pairs=100,
    )

    assert res.parameters == {
        "data": {"row_count": 6, "column_count": 5},
        "columns": ["income", "zip", "region"],
        "categorical_columns": ["zip"],
        "numeric_method": "kendall",
        "categorical_method": "cramers_v",
        "max_categories": 10,
        "max_pairs": 100,
    }


def test_numbers_are_numeric_and_booleans_text_and_categories_are_categorical():
    df = pd.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float": [0.5, 1.5, 2.5, 4.0],
            "nullable": pd.array([1, None, 3, 4], dtype="Int64"),
            "arrow": pd.array([1.5, 2.5, None, 0.5], dtype="double[pyarrow]"),
            "flag": [True, False, True, False],
            "nullable_flag": pd.array([True, None, False, False], dtype="boolean"),
            "text": ["a", "b", "a", "b"],
            "object": pd.Series(["x", "y", "x", 1], dtype=object),
            "string": pd.array(["p", "q", None, "q"], dtype="string"),
            "category": pd.Categorical(["u", "v", "u", "v"]),
        }
    )

    res = ae.compute_association_matrix(df)

    assert {name: entry.type for name, entry in res.columns.items()} == {
        "int": "numeric",
        "float": "numeric",
        "nullable": "numeric",
        "arrow": "numeric",
        "flag": "categorical",
        "nullable_flag": "categorical",
        "text": "categorical",
        "object": "categorical",
        "string": "categorical",
        "category": "categorical",
    }
    assert res.skipped_columns == []


def test_numeric_codes_listed_as_categorical_count_as_categories():
    df = mixed_frame()

    res = ae.compute_association_matrix(df, categorical_columns=["zip"])

    assert res.columns["zip"].type == "categorical"
    assert pair(res, "zip", "age").method == "correlation_ratio"
    assert pair(res, "zip", "region").method == "cramers_v_bias_corrected"
    without = ae.compute_association_matrix(df)
    assert without.columns["zip"].type == "numeric"
    assert pair(without, "zip", "age").method == "pearson"


def test_skips_datetimes_timedeltas_and_columns_without_values_with_reasons():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "when": pd.to_datetime(["2024-01-01"] * 4),
            "took": pd.to_timedelta([1, 2, 3, 4], unit="s"),
            "empty": [None] * 4,
            "nothing": [np.nan] * 4,
            "infinite": [np.inf, -np.inf, np.nan, np.inf],
            "lists": pd.Series([[1], [2], None, [1]], dtype=object),
            "y": [2.0, 1.0, 4.0, 3.0],
        }
    )

    res = ae.compute_association_matrix(df)

    assert [(s.column, s.reason) for s in res.skipped_columns] == [
        ("when", "datetime"),
        ("took", "timedelta"),
        ("empty", "all_missing"),
        ("nothing", "all_missing"),
        ("infinite", "no_finite_values"),
        ("lists", "unhashable_values"),
    ]
    assert list(res.columns) == ["x", "y"]
    assert [(p.x_column, p.y_column) for p in res.pairs] == [("x", "y")]


def test_skips_categorical_columns_with_more_levels_than_max_categories():
    df = pd.DataFrame(
        {
            "few": ["a", "b", "a", "b", "a"],
            "many": ["a", "b", "c", "d", "e"],
            "codes": [1, 2, 3, 4, 4],
            "x": [1.0, 2.0, 3.0, 5.0, 4.0],
        }
    )

    res = ae.compute_association_matrix(
        df, categorical_columns=["codes"], max_categories=3
    )

    assert [(s.column, s.reason) for s in res.skipped_columns] == [
        ("many", "too_many_categories"),
        ("codes", "too_many_categories"),
    ]
    assert res.skipped_columns[0].message == (
        "It has more distinct values than max_categories=3; raise max_categories "
        "to pair it."
    )
    assert list(res.columns) == ["few", "x"]
    assert res.columns["few"].level_count == 2


@pytest.mark.parametrize(
    ("method", "expected", "reference"),
    [
        ("pearson", None, lambda x, y: stats.pearsonr(x, y).statistic),
        # Ties share the mean of their ranks; SciPy's docstring prints 0.7.
        ("spearman", 0.7, lambda x, y: stats.spearmanr(x, y).statistic),
        # tau-b by definition: 11 / sqrt(20 * 20); SciPy's docstring prints 0.55.
        ("kendall", 0.55, lambda x, y: stats.kendalltau(x, y).statistic),
    ],
)
def test_numeric_methods_match_scipy_on_scipys_examples(method, expected, reference):
    df = pd.DataFrame({"x": TIED_X, "y": TIED_Y})

    res = ae.compute_association_matrix(df, numeric_method=method)

    [found] = res.pairs
    assert found.method == method
    assert found.value == pytest.approx(reference(TIED_X, TIED_Y), abs=ATOL)
    if expected is not None:
        assert found.value == pytest.approx(expected, abs=ATOL)


def test_spearman_matches_the_published_example_without_ties():
    # Wikipedia, "Spearman's rank correlation coefficient": IQ against hours of
    # TV per week gives rho = -29/165.
    iq = [106, 100, 86, 101, 99, 103, 97, 113, 112, 110]
    tv = [7, 27, 2, 50, 28, 29, 20, 12, 6, 17]

    res = ae.compute_association_matrix(
        pd.DataFrame({"iq": iq, "tv": tv}), numeric_method="spearman"
    )

    assert res.pairs[0].value == pytest.approx(-29 / 165, abs=ATOL)


def test_kendall_matches_scipys_second_example():
    x1 = [12, 2, 1, 12, 2]
    x2 = [1, 4, 7, 1, 0]

    res = ae.compute_association_matrix(
        pd.DataFrame({"x1": x1, "x2": x2}), numeric_method="kendall"
    )

    assert res.pairs[0].value == pytest.approx(-0.4714045207910316, abs=ATOL)
    assert res.pairs[0].value == pytest.approx(
        stats.kendalltau(x1, x2).statistic, abs=ATOL
    )


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_rank_methods_rank_each_pairs_own_complete_rows(method):
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = x + rng.normal(size=60)
    z = rng.normal(size=60)
    x[[3, 8, 20]] = np.nan
    y[[5, 8, 41, 42]] = np.nan
    z[:30] = np.nan
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    reference = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }[method]

    res = ae.compute_association_matrix(df, numeric_method=method)

    for first, second in [("x", "y"), ("x", "z"), ("y", "z")]:
        complete = df[[first, second]].dropna()
        found = pair(res, first, second)
        assert found.pair_count == len(complete)
        assert found.value == pytest.approx(
            reference(complete[first], complete[second]).statistic, abs=ATOL
        )


def test_cramers_v_matches_scipys_association():
    df = table_frame(SCIPY_TABLE)

    res = ae.compute_association_matrix(df, categorical_method="cramers_v")

    [found] = res.pairs
    assert found.method == "cramers_v"
    assert found.pair_count == 2_425
    assert found.value == pytest.approx(
        association(np.array(SCIPY_TABLE), method="cramer"), abs=ATOL
    )
    assert found.value == pytest.approx(0.18617813077483678, abs=ATOL)


def test_bias_corrected_cramers_v_matches_the_published_example():
    df = table_frame(MUSIC_TABLE)

    corrected = ae.compute_association_matrix(df).pairs[0]
    plain = ae.compute_association_matrix(df, categorical_method="cramers_v").pairs[0]

    # effectsize prints 0.23 with the correction and 0.24 without it.
    assert corrected.method == "cramers_v_bias_corrected"
    assert corrected.value == pytest.approx(0.23, abs=0.005)
    assert plain.value == pytest.approx(0.24, abs=0.005)
    assert plain.value == pytest.approx(
        association(np.array(MUSIC_TABLE), method="cramer"), abs=ATOL
    )


def test_bias_corrected_cramers_v_follows_bergsmas_formula():
    # Bergsma (2013), with chi2 from SciPy: phi2 = chi2 / n, less its expected
    # value (k - 1)(r - 1) / (n - 1), over the corrected level counts.
    table = np.array(SCIPY_TABLE)
    n = table.sum()
    r, k = table.shape
    chi2 = stats.chi2_contingency(table, correction=False).statistic
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    corrected_r = r - (r - 1) ** 2 / (n - 1)
    corrected_k = k - (k - 1) ** 2 / (n - 1)
    expected = np.sqrt(phi2 / min(corrected_r - 1, corrected_k - 1))

    res = ae.compute_association_matrix(table_frame(SCIPY_TABLE))

    assert res.pairs[0].value == pytest.approx(expected, abs=ATOL)


def test_correlation_ratio_matches_the_published_example():
    # Wikipedia, "Correlation ratio": test scores per subject. The between-subject
    # sum of squares is 6780 and the total 9640, so eta = sqrt(6780 / 9640).
    scores = {
        "Algebra": [45, 70, 29, 15, 21],
        "Geometry": [40, 20, 30, 42],
        "Statistics": [65, 95, 80, 70, 85, 73],
    }
    df = pd.DataFrame(
        {
            "score": [s for values in scores.values() for s in values],
            "subject": [name for name, values in scores.items() for _ in values],
        }
    )
    f = stats.f_oneway(*scores.values()).statistic
    between, within = len(scores) - 1, len(df) - len(scores)

    res = ae.compute_association_matrix(df)

    [found] = res.pairs
    assert found.method == "correlation_ratio"
    assert (found.x_column, found.y_column) == ("subject", "score")
    assert (found.x_type, found.y_type) == ("categorical", "numeric")
    assert found.direction == "x_to_y"
    assert found.value == pytest.approx(np.sqrt(6780 / 9640), abs=ATOL)
    assert found.value is not None
    assert found.value**2 == pytest.approx(
        f * between / (f * between + within), abs=ATOL
    )


def test_symmetric_measures_say_so():
    res = ae.compute_association_matrix(
        mixed_frame(), columns=["age", "income", "region"]
    )

    assert pair(res, "age", "income").direction == "symmetric"
    assert pair(res, "region", "age").direction == "x_to_y"


def undefined_frame():
    """Pairs that are undefined for each reason, and one that isn't."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            # Constant where y is present, though not over all rows.
            "flat": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            "sparse": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0],
            "one": ["a", "a", "a", "a", "a", "a"],
            "two": ["a", "b", "a", "b", "a", "b"],
        }
    )


def test_each_pair_uses_its_own_complete_rows():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.inf],
            "y": [2.0, 1.0, 4.0, np.nan, 6.0, 5.0, 3.0],
            "g": ["a", None, "b", "b", "a", "b", "a"],
        }
    )

    res = ae.compute_association_matrix(df)

    # x is missing in row 2 and infinite in row 6; y in row 3; g in row 1.
    assert pair(res, "x", "y").pair_count == 4
    assert pair(res, "g", "x").pair_count == 4
    assert pair(res, "g", "y").pair_count == 5
    complete = df[["x", "y"]].iloc[[0, 1, 4, 5]]
    assert pair(res, "x", "y").value == pytest.approx(
        stats.pearsonr(complete["x"], complete["y"]).statistic, abs=ATOL
    )


def test_a_pair_without_a_value_is_null_with_its_reason():
    res = ae.compute_association_matrix(undefined_frame())

    reasons = {
        (p.x_column, p.y_column): (p.value, p.undefined_reason) for p in res.pairs
    }
    assert reasons[("x", "y")][1] is None
    assert reasons[("x", "flat")] == (None, "constant_column")
    assert reasons[("x", "sparse")] == (None, "too_few_rows")
    assert reasons[("one", "x")] == (None, "single_level")
    assert reasons[("one", "two")] == (None, "single_level")
    assert reasons[("two", "flat")] == (None, "constant_column")
    assert reasons[("two", "x")][1] is None


@pytest.mark.parametrize("method", ["cramers_v", "cramers_v_bias_corrected"])
def test_a_level_for_every_complete_row_leaves_the_pair_null(method):
    # A column whose every row is its own level, such as an ID, trivially
    # determines every other column: eta and the plain V would be 1, and the
    # bias-corrected V divides by zero.
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "group": ["u", "u", "v", "v", "u"],
            "score": [1.5, 2.0, 0.5, 3.0, 2.5],
        }
    )

    res = ae.compute_association_matrix(df, categorical_method=method)

    assert pair(res, "id", "group").undefined_reason == "one_row_per_level"
    assert pair(res, "id", "score").undefined_reason == "one_row_per_level"
    assert pair(res, "id", "score").value is None
    assert pair(res, "group", "score").value is not None
    assert res.findings == []


def test_limitations_name_the_null_pairs_and_why():
    res = ae.compute_association_matrix(undefined_frame())

    position = {(p.x_column, p.y_column): i for i, p in enumerate(res.pairs)}
    by_code = {lim.code: lim for lim in res.limitations}
    assert by_code["too_few_values"].fields == [
        f"pairs[{position[(x, y)]}].value"
        for x, y in [("x", "sparse"), ("y", "sparse"), ("flat", "sparse")]
    ] + [
        f"pairs[{position[(x, y)]}].value"
        for x, y in [("one", "sparse"), ("two", "sparse")]
    ]
    assert "Every measure needs at least 3" in by_code["too_few_values"].reason
    assert sorted(by_code["zero_variance"].fields) == sorted(
        f"pairs[{i}].value"
        for (x, y), i in position.items()
        if res.pairs[i].undefined_reason in ("constant_column", "single_level")
    )


def test_pairs_have_no_p_values_or_intervals():
    res = ae.compute_association_matrix(mixed_frame())

    record = res.to_dict()

    assert set(record["pairs"][0]) == {
        "x_column",
        "y_column",
        "x_type",
        "y_type",
        "method",
        "value",
        "pair_count",
        "direction",
        "undefined_reason",
    }


def strength_frame():
    """Pairs of known strength: -1, 0.8 and about 0, plus a constant column."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    return pd.DataFrame(
        {
            "flat": [1.0] * 8,
            "weak": [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            "x": x,
            "mirror": -x,
            "noisy": x + np.array([3.0, -3.0, 2.0, -2.0, 4.0, -4.0, 1.0, -1.0]),
        }
    )


def test_pairs_come_strongest_first_with_null_values_last():
    res = ae.compute_association_matrix(strength_frame())

    values = [p.value for p in res.pairs]
    defined = [abs(v) for v in values if v is not None]
    assert defined == sorted(defined, reverse=True)
    assert values[: len(defined)] == [v for v in values if v is not None]
    assert [(p.x_column, p.y_column) for p in res.pairs[:1]] == [("x", "mirror")]
    assert res.pairs[0].value == pytest.approx(-1.0, abs=ATOL)
    # Pairs of equal strength keep the selected column order.
    nulls = [(p.x_column, p.y_column) for p in res.pairs if p.value is None]
    assert nulls == [
        ("flat", "weak"),
        ("flat", "x"),
        ("flat", "mirror"),
        ("flat", "noisy"),
    ]


def test_the_record_keeps_the_100_strongest_pairs_with_the_total():
    rng = np.random.default_rng(7)
    base = rng.normal(size=40)
    df = pd.DataFrame(
        {f"c{i:02d}": base * (i / 16) + rng.normal(size=40) for i in range(16)}
    )

    res = ae.compute_association_matrix(df)

    record = res.to_dict()
    assert len(res.pairs) == 120
    assert record["pairs_total"] == 120
    assert record["pairs"] == [p.model_dump(mode="json") for p in res.pairs[:100]]
    weakest_kept = min(abs(p["value"]) for p in record["pairs"])
    assert all(abs(p.value or 0.0) <= weakest_kept for p in res.pairs[100:])


def test_to_frame_has_one_row_per_pair_strongest_first():
    res = ae.compute_association_matrix(mixed_frame())

    frame = res.to_frame()

    assert list(frame.columns) == [
        "x_column",
        "y_column",
        "x_type",
        "y_type",
        "method",
        "value",
        "pair_count",
        "direction",
        "undefined_reason",
    ]
    assert len(frame) == len(res.pairs) == 6
    assert frame["x_column"].tolist() == [p.x_column for p in res.pairs]
    assert frame["value"].tolist() == [p.value for p in res.pairs]
    assert frame["value"].dtype == "float64"
    assert frame["pair_count"].dtype == "int64"
    pd.testing.assert_frame_equal(res.to_frame(table="pairs"), frame)


def test_to_frame_matrix_reads_from_row_to_column():
    res = ae.compute_association_matrix(
        mixed_frame(), columns=["age", "region", "income"]
    )

    matrix = res.to_frame(table="matrix")

    assert list(matrix.index) == list(matrix.columns) == ["age", "region", "income"]
    assert (matrix.index.name, matrix.columns.name) == ("x_column", "y_column")
    r = pair(res, "age", "income").value
    assert matrix.loc["age", "income"] == matrix.loc["income", "age"] == r
    # The correlation ratio says how much region explains age, not the reverse.
    assert matrix.loc["region", "age"] == pair(res, "region", "age").value
    assert np.isnan(matrix.loc["age", "region"])
    assert np.isnan(np.diag(matrix.to_numpy())).all()


def test_to_frame_with_an_unknown_table_raises_with_the_choices():
    res = ae.compute_association_matrix(mixed_frame())

    with pytest.raises(ae.errors.InvalidArgumentError) as caught:
        res.to_frame(table="matirx")  # ty: ignore[invalid-argument-type]  # the typo is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "to_frame got 'matirx' for table, but table must be 'pairs' or 'matrix'. "
        "Did you mean 'matrix'?"
    )


def test_to_frame_with_data_for_a_table_name_shows_none_of_it():
    res = ae.compute_association_matrix(mixed_frame())
    secret = pd.Series(["Zaragoza-7f3", "Quimper-2b9"])

    with pytest.raises(ae.errors.InvalidArgumentError) as caught:
        res.to_frame(table=secret)  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    assert str(caught.value) == (
        "to_frame got a pandas Series of 2 values for table, but table must be "
        "'pairs' or 'matrix'."
    )


def test_prints_the_strongest_pairs_and_names_both_tables():
    res = ae.compute_association_matrix(mixed_frame())

    lines = str(res).splitlines()

    assert lines[0] == "compute_association_matrix: completed on 6 rows x 5 columns"
    assert lines[1:3] == [
        "Pairs, each on its own complete rows: 3 numeric (pearson), 0 categorical "
        "(cramers_v_bias_corrected),",
        "    3 mixed (correlation_ratio, from the categorical x_column to the numeric "
        "y_column).",
    ]
    assert lines[3] == "Skipped 1 column, see res.skipped_columns: 'when' (datetime)."
    assert lines[4] == (
        "Tables: res.to_frame() has every pair, strongest first; "
        "res.to_frame(table='matrix') has the 4 x 4 matrix."
    )
    assert lines[5] == "Pairs (6 of 6 shown):"
    assert lines[6].split() == [
        "x_column",
        "y_column",
        "method",
        "value",
        "pair_count",
        "direction",
    ]
    strongest = res.pairs[0]
    assert lines[7].split() == [
        strongest.x_column,
        strongest.y_column,
        strongest.method,
        f"{strongest.value:.4g}",
        str(strongest.pair_count),
        strongest.direction,
    ]
    assert lines[-1] == (
        "More: res.to_frame() has all 6 pairs; res.findings, res.limitations and "
        "res.to_json() have the rest."
    )


def test_flags_perfect_associations_as_suspected_duplicated_or_derived_columns():
    df = pd.DataFrame(
        {
            "price": [10.0, 12.5, 9.0, 14.0, 11.0, 13.5],
            "price_cents": [1000, 1250, 900, 1400, 1100, 1350],
            "level": ["low", "mid", "low", "high", "mid", "high"],
            "tier": ["b", "c", "b", "a", "c", "a"],
            "score": [3.0, 5.0, 3.0, 9.0, 5.0, 9.0],
            "noise": [0.3, -1.2, 0.8, 0.1, -0.4, 1.1],
        }
    )

    res = ae.compute_association_matrix(df)

    [finding] = res.findings
    perfect = [
        ("price", "price_cents"),
        ("level", "tier"),
        ("level", "score"),
        ("tier", "score"),
    ]
    paths = [
        f"pairs[{i}].value"
        for i, p in enumerate(res.pairs)
        if (p.x_column, p.y_column) in perfect
    ]
    assert finding.code == "perfect_associations"
    assert finding.assessment == "suspected"
    assert sorted(finding.fields) == sorted(paths)
    assert finding.columns == ["price", "price_cents", "level", "tier", "score"]
    assert finding.claim == (
        "4 pairs are perfectly associated, an absolute value of 1, so one column "
        "may duplicate or derive from the other: ('price', 'price_cents'), "
        "('level', 'tier'), ('level', 'score') and ('tier', 'score')."
    )


def test_a_nearly_perfect_association_is_not_flagged():
    x = np.arange(1.0, 11.0)
    df = pd.DataFrame({"x": x, "y": x + np.r_[0.01, np.zeros(9)]})

    res = ae.compute_association_matrix(df)

    value = res.pairs[0].value
    assert value is not None
    assert 0.99999 < value < 1 - 1e-9
    assert res.findings == []


def test_declares_its_statuses():
    [info] = [
        a for a in ae.list_analyses().analyses if a.name == "compute_association_matrix"
    ]

    assert info.statuses == [
        "completed",
        "budget_exceeded",
        "insufficient_data",
        "not_applicable",
        "failed",
    ]


def test_more_pairs_than_max_pairs_stops_before_computing_any():
    res = ae.compute_association_matrix(mixed_frame(), max_pairs=5)

    assert res.status == "budget_exceeded"
    assert res.status_reason is not None
    assert res.status_reason.code == "too_many_pairs"
    assert res.status_reason.message == (
        "The 4 paired columns make 6 pairs, more than max_pairs=5, so no pair was "
        "computed. Name fewer columns in columns, or pass max_pairs=6 to compute "
        "them all."
    )
    assert res.pairs == []
    assert list(res.columns) == ["age", "region", "zip", "income"]
    assert res.to_frame(table="matrix").shape == (4, 4)
    lines = str(res).splitlines()
    assert lines[1].startswith("Reason: too_many_pairs: The 4 paired columns make")
    assert "Pairs: none" in lines


def test_exactly_max_pairs_pairs_are_computed():
    res = ae.compute_association_matrix(mixed_frame(), max_pairs=6)

    assert res.status == "completed"
    assert len(res.pairs) == 6


def test_fewer_than_two_paired_columns_is_not_applicable():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "empty": [None, None, None],
        }
    )

    res = ae.compute_association_matrix(df)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert res.status_reason.code == "fewer_than_two_columns"
    assert res.status_reason.message == (
        "Fewer than two columns can be paired: numeric, boolean, text, object and "
        "categorical columns can, and skipped_columns says why the others can't."
    )
    assert res.pairs == []
    assert [s.column for s in res.skipped_columns] == ["when", "empty"]


def test_data_without_rows_is_insufficient_data():
    df = mixed_frame().iloc[:0]

    res = ae.compute_association_matrix(df)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_rows"
    assert res.pairs == []


def test_no_pair_with_a_value_is_insufficient_data():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 1.0], "g": ["a", "b"]})

    res = ae.compute_association_matrix(df)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_defined_pairs"
    assert res.status_reason.message == (
        "No pair has a value: each pair's undefined_reason says why, such as "
        "too_few_rows."
    )
    assert [p.undefined_reason for p in res.pairs] == ["too_few_rows"] * 3


def test_describe_analysis_lists_max_pairs_as_a_work_limit():
    description = ae.describe_analysis("compute_association_matrix")

    parameters = {p.name: p for p in description.parameters}
    assert description.work_limit_parameters == ["max_pairs"]
    assert parameters["max_pairs"].default == 5000
    assert parameters["max_pairs"].valid_range == "1 <= max_pairs"
    assert parameters["max_categories"].default == 50
    assert parameters["max_categories"].valid_range == "2 <= max_categories"
    assert parameters["numeric_method"].allowed_values == [
        "pearson",
        "spearman",
        "kendall",
    ]
    assert parameters["categorical_method"].allowed_values == [
        "cramers_v",
        "cramers_v_bias_corrected",
    ]


def test_shows_no_raw_data_values():
    df = pd.DataFrame(
        {
            "city": ["Zaragoza-7f3", "Quimper-2b9", "Zaragoza-7f3", "Quimper-2b9"],
            "code": pd.Categorical(["XQ-191", "XQ-191", "ZW-404", "ZW-404"]),
            "amount": [123456.789, 98765.4321, 55555.125, 31337.0625],
            "count": [9001, 4242, 7777, 1234],
            "when": pd.to_datetime(["1999-12-31"] * 4),
        }
    )
    raw = ["Zaragoza", "Quimper", "XQ-191", "ZW-404", "123456", "98765", "55555"]
    raw += ["31337", "9001", "4242", "7777", "1999"]

    res = ae.compute_association_matrix(df)

    shown = "\n".join(
        [
            str(res),
            res.to_json(),
            res.to_frame().to_csv(),
            res.to_frame(table="matrix").to_csv(),
        ]
    )
    assert [value for value in raw if value in shown] == []


def test_a_datetime_named_in_columns_raises_with_conversions():
    df = mixed_frame()

    with pytest.raises(ae.errors.ColumnTypeError) as caught:
        ae.compute_association_matrix(df, columns=["age", "when"])

    assert caught.value.code == "unsupported_column_type"
    assert str(caught.value) == (
        "compute_association_matrix can't use column 'when' with dtype "
        f"{df['when'].dtype}. It accepts numeric, boolean, text, object and "
        "categorical columns, nullable and pyarrow-backed ones included. Convert "
        "datetimes to numbers with (df['when'] - df['when'].min())"
        ".dt.total_seconds() or to labels with df['when'].dt.strftime('%Y-%m'), "
        "or leave 'when' out of columns."
    )


def test_categorical_columns_must_name_columns_the_call_pairs():
    df = mixed_frame()

    with pytest.raises(ae.errors.ColumnNotFoundError) as unknown:
        ae.compute_association_matrix(df, categorical_columns=["Zip"])
    with pytest.raises(ae.errors.ColumnTypeError) as datetime:
        ae.compute_association_matrix(df, categorical_columns=["when"])
    with pytest.raises(ae.errors.InvalidArgumentError) as outside:
        ae.compute_association_matrix(
            df, columns=["age", "income"], categorical_columns=["zip"]
        )

    assert str(unknown.value).startswith(
        "compute_association_matrix got an unknown column 'Zip' in "
        "categorical_columns. Did you mean 'zip'?"
    )
    assert "Convert datetimes to labels with df['when'].dt.strftime('%Y-%m')" in (
        str(datetime.value)
    )
    assert str(outside.value) == (
        "compute_association_matrix got 'zip' in categorical_columns, but columns "
        "leaves it out. Add it to columns, or leave it out of categorical_columns."
    )


def test_a_text_column_listed_as_categorical_stays_categorical():
    res = ae.compute_association_matrix(mixed_frame(), categorical_columns=["region"])

    assert res.columns["region"].type == "categorical"
    assert res.parameters["categorical_columns"] == ["region"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"max_pairs": 0},
            "compute_association_matrix got 0 for max_pairs, but max_pairs must be "
            "int with 1 <= max_pairs.",
        ),
        (
            {"max_categories": 1},
            "compute_association_matrix got 1 for max_categories, but "
            "max_categories must be int with 2 <= max_categories.",
        ),
        (
            {"numeric_method": "Spearman"},
            "compute_association_matrix got 'Spearman' for numeric_method, but "
            "numeric_method must be one of 'pearson', 'spearman' or 'kendall'. Did "
            "you mean 'spearman'?",
        ),
    ],
)
def test_invalid_options_raise_before_any_computation(arguments, message):
    with pytest.raises(ae.errors.InvalidArgumentError) as caught:
        ae.compute_association_matrix(mixed_frame(), **arguments)

    assert str(caught.value) == (
        f"{message} See help(ae.compute_association_matrix) for every argument."
    )


def test_support_counts_the_values_each_column_gives_and_leaves_out():
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, np.inf, 4.0, 2.0],
            "g": ["a", None, "b", "a", "b"],
            "when": pd.to_datetime(["2024-01-01"] * 5),
            "empty": [None] * 5,
            "y": [5.0, 4.0, 3.0, 1.0, 2.0],
        }
    )

    res = ae.compute_association_matrix(df)

    support = {c.column: c for c in res.support.columns}
    assert res.support.row_count == 5
    assert res.support.column_count == 4
    assert list(support) == ["x", "g", "empty", "y"]
    assert support["x"].model_dump() == {
        "column": "x",
        "eligible_count": 5,
        "evaluated_count": 3,
        "excluded_count": 2,
        "excluded_by_reason": {"missing": 1, "infinite": 1},
    }
    assert support["g"].excluded_by_reason == {"missing": 1}
    assert support["empty"].model_dump() == {
        "column": "empty",
        "eligible_count": 5,
        "evaluated_count": 0,
        "excluded_count": 5,
        "excluded_by_reason": {"all_missing": 5},
    }
    entry = res.columns["x"]
    assert (entry.value_count, entry.missing_count, entry.infinite_count) == (3, 1, 1)


def test_nan_stored_as_a_value_is_missing_in_numbers_read_as_categories():
    stored_nan = pd.arrays.FloatingArray(
        np.array([1.0, np.nan, 2.0, 1.0, 2.0, np.nan]), np.zeros(6, dtype=bool)
    )
    df = pd.DataFrame({"code": stored_nan, "group": ["a", "b", "a", "a", "b", "b"]})

    res = ae.compute_association_matrix(df, categorical_columns=["code"])

    entry = res.columns["code"]
    assert (entry.missing_count, entry.level_count) == (2, 2)
    [limitation] = [
        lim for lim in res.limitations if lim.code == "nan_counted_as_missing"
    ]
    assert limitation.fields == ["columns['code'].missing_count"]


def test_reading_numbers_as_float64_adds_the_shared_limitations():
    stored_nan = pd.arrays.FloatingArray(
        np.array([1.0, np.nan, 2.0, 0.5]), np.array([False, False, False, True])
    )
    df = pd.DataFrame(
        {
            "big": pd.array([2**60, 2**60 + 1, 3, 4], dtype="Int64"),
            "stored": stored_nan,
            "y": [1.0, 2.0, 3.0, 5.0],
        }
    )

    res = ae.compute_association_matrix(df)

    by_code = {lim.code: lim for lim in res.limitations}
    assert by_code["float_precision"].fields == ["columns['big']"]
    assert by_code["nan_counted_as_missing"].fields == [
        "columns['stored'].missing_count"
    ]
    assert res.columns["stored"].missing_count == 2


@pytest.mark.parametrize(
    ("method", "libraries"),
    [("pearson", ["numpy", "pandas"]), ("kendall", ["numpy", "pandas", "scipy"])],
)
def test_records_the_libraries_it_used(method, libraries):
    res = ae.compute_association_matrix(mixed_frame(), numeric_method=method)

    assert list(res.execution.versions) == ["agent_eda", "python", *libraries]


def reference_value(df, pair_entry):
    """The pair's measure computed independently on its complete rows.

    Numbers are shifted by their median first, which changes no measure and,
    for values close together far from 0, keeps the references' own sums exact.
    """
    x, y = pair_entry.x_column, pair_entry.y_column
    complete = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()

    def shifted(name):
        values = complete[name].astype(float)
        return values - values.median()

    if pair_entry.method == "pearson":
        return stats.pearsonr(shifted(x), shifted(y)).statistic
    if pair_entry.method == "cramers_v":
        table = pd.crosstab(complete[x], complete[y]).to_numpy()
        return association(table, method="cramer")
    if pair_entry.method == "cramers_v_bias_corrected":
        # Bergsma (2013), on SciPy's chi2.
        table = pd.crosstab(complete[x], complete[y]).to_numpy()
        n = table.sum()
        r, k = table.shape
        chi2 = stats.chi2_contingency(table, correction=False).statistic
        phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
        smaller = min(r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1))
        return np.sqrt(phi2 / (smaller - 1))
    assert pair_entry.method == "correlation_ratio", pair_entry.method
    # The correlation ratio, from pandas' group means.
    values = shifted(y)
    means = values.groupby(complete[x], observed=True).transform("mean")
    between = ((means - values.mean()) ** 2).sum()
    return np.sqrt(between / ((values - values.mean()) ** 2).sum())


def random_frame(rows, seed):
    """Numbers and categories with missing values, infinities and relations."""
    rng = np.random.default_rng(seed)
    group = rng.choice(["a", "b", "c", "d"], size=rows)
    level = rng.integers(0, 3, size=rows)
    x = rng.normal(5.0, 2.0, size=rows)
    y = 0.5 * x + rng.normal(size=rows) + (group == "b")
    z = 1e8 + rng.normal(size=rows) * 1e-3  # far from 0, small spread
    w = rng.lognormal(size=rows)
    for values in (x, y, z, w):
        values[rng.random(rows) < 0.1] = np.nan
    w[rng.random(rows) < 0.01] = np.inf
    df = pd.DataFrame(
        {
            "x": x,
            "group": pd.Series(group).where(rng.random(rows) > 0.05),
            "y": y,
            "z": z,
            "level": pd.array(level, dtype="Int64"),
            "flag": rng.random(rows) < 0.3,
            "w": w,
        }
    )
    df.loc[rng.random(rows) < 0.05, "level"] = pd.NA
    return df


@pytest.mark.parametrize("rows", [500, 70_000])
def test_every_pair_matches_an_independent_computation(rows):
    # 70,000 rows take more than one chunk of rows.
    df = random_frame(rows, seed=rows)

    res = ae.compute_association_matrix(
        df, categorical_columns=["level"], categorical_method="cramers_v"
    )

    assert len(res.pairs) == 21
    for found in res.pairs:
        expected = reference_value(df, found)
        assert found.value == pytest.approx(expected, abs=ATOL), found


@pytest.mark.parametrize("method", ["cramers_v", "cramers_v_bias_corrected"])
def test_tables_too_large_to_hold_whole_match_an_independent_computation(method):
    # Over 2,048 levels each: a dense table of every pair of levels would have
    # more than 4 million cells, almost all of them empty.
    rng = np.random.default_rng(11)
    first = rng.permutation(np.arange(6_300) % 2_100)
    second = (first + rng.integers(0, 3, first.size)) % 2_100
    df = pd.DataFrame({"first": first.astype(str), "second": second.astype(str)})

    res = ae.compute_association_matrix(
        df, max_categories=5_000, categorical_method=method
    )

    assert min(entry.level_count or 0 for entry in res.columns.values()) > 2_048
    [found] = res.pairs
    table = pd.crosstab(df["first"], df["second"]).to_numpy()
    plain = association(table, method="cramer")
    if method == "cramers_v":
        assert found.value == pytest.approx(plain, abs=ATOL)
    else:
        n = table.sum()
        r, k = table.shape
        phi2 = max(0.0, plain**2 * min(r - 1, k - 1) - (k - 1) * (r - 1) / (n - 1))
        corrected = min(r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)) - 1
        assert found.value == pytest.approx(np.sqrt(phi2 / corrected), abs=1e-9)


@pytest.mark.parametrize("method", ["cramers_v", "cramers_v_bias_corrected"])
def test_columns_with_few_and_many_levels_match_an_independent_computation(method):
    # From 3 to 70 levels: some tables come from one product, others are
    # counted pair by pair.
    rng = np.random.default_rng(5)
    rows = 3_000
    df = pd.DataFrame(
        {
            "wide": rng.integers(0, 70, rows).astype(str),
            "few": rng.integers(0, 3, rows).astype(str),
            "mid": rng.integers(0, 40, rows).astype(str),
            "some": rng.integers(0, 5, rows).astype(str),
            "narrow": rng.integers(0, 30, rows).astype(str),
            "x": rng.normal(size=rows),
        }
    )
    df.loc[rng.random(rows) < 0.1, "mid"] = None
    df.loc[rng.random(rows) < 0.1, "few"] = None
    df["x"] += df["narrow"].astype(int) / 30
    df.loc[df["few"] == "1", "some"] = "4"

    res = ae.compute_association_matrix(
        df, max_categories=100, categorical_method=method
    )

    assert len(res.pairs) == 15
    for found in res.pairs:
        expected = reference_value(df, found)
        assert found.value == pytest.approx(expected, abs=ATOL)


@pytest.mark.parametrize(
    "near",
    [
        lambda rng: 1e6 + rng.normal(size=100) * 1e-3,
        # Large integers close together, whose mean float64 rounds by a visible
        # share of their spread.
        lambda rng: 1e15 + rng.integers(0, 5, size=100),
    ],
    ids=["1e6", "1e15"],
)
def test_pairs_far_from_their_columns_centre_are_still_exact(near):
    # x is far from 0 where y and g are present and 0 elsewhere, so the pairs'
    # rows lie far from x's overall mean compared with their spread.
    rng = np.random.default_rng(3)
    x = np.r_[near(rng), np.zeros(100)]
    y = np.r_[rng.normal(size=100), np.full(100, np.nan)]
    g = np.r_[rng.choice(["u", "v", "w"], size=100), np.full(100, None)]
    df = pd.DataFrame({"x": x, "y": y, "g": g})

    res = ae.compute_association_matrix(df)

    for found in res.pairs:
        assert found.value == pytest.approx(reference_value(df, found), abs=ATOL)


def test_subnormal_values_are_scaled_without_overflow():
    # Multiples of the smallest subnormal number: scaling them up to [0.5, 1)
    # takes a power of two that float64 can't hold on its own.
    tiny = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * 5e-324
    df = pd.DataFrame({"tiny": tiny, "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]})

    res = ae.compute_association_matrix(df)

    assert res.limitations == []
    expected = stats.pearsonr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], df["y"]).statistic
    assert res.pairs[0].value == pytest.approx(expected, abs=ATOL)


def test_values_tiny_beside_their_columns_largest_are_still_exact():
    # Scaled for the 1.0 in rows the pair leaves out, the pair's own values
    # would square to subnormal numbers, which lose precision.
    rng = np.random.default_rng(13)
    x = rng.normal(size=50) * 1e-158
    x[:2] = [1.0, -1.0]
    y = rng.normal(size=50) + x * 1e158
    y[:2] = np.nan

    res = ae.compute_association_matrix(pd.DataFrame({"x": x, "y": y}))

    # Multiplying by 2^525 is exact and changes no correlation.
    expected = stats.pearsonr(np.ldexp(x[2:], 525), y[2:]).statistic
    assert res.pairs[0].value == pytest.approx(expected, abs=ATOL)


def test_a_column_constant_in_a_pairs_rows_gives_a_null_pair():
    df = pd.DataFrame(
        {
            "x": [5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan],
            "g": ["a", "b", "a", "b", None, None, None],
        }
    )

    res = ae.compute_association_matrix(df)

    assert pair(res, "x", "y").undefined_reason == "constant_column"
    assert pair(res, "g", "x").undefined_reason == "constant_column"
    assert pair(res, "g", "y").value is not None
