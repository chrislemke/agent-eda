"""Paths that ordinary generated data never reached: the edges of the parse rules,
categories of unusual types and numbers too large for a float."""

import datetime as dt

import pandas as pd
import pytest

import agent_eda as ae


def text_column(*values: str) -> pd.DataFrame:
    return pd.DataFrame({"x": pd.Series(list(values), dtype=object)})


def counts(entry):
    """An entry's counts: compatible, convertible, invalid and missing."""
    return (
        entry.compatible_count,
        entry.convertible_count,
        entry.invalid_count,
        entry.missing_count,
    )


@pytest.mark.parametrize(
    "text",
    [
        "2024-01-05T23:59:59",
        "2024-01-05T23:59:60",  # there is no second 60
        "2024-01-05T24:00:00",
        "2024-01-05T10:60:00",
        "2024-01-05T10:00:00+23:59",
        "2024-01-05T10:00:00+24:00",
        "2024-01-05T10:00:00-24:00",
        "2024-01-05T10:00:00+23:60",
        "2024-02-30T10:00:00",
        "2023-02-29",
        "2024-02-29",
        "2024-13-01",
    ],
)
def test_text_is_a_datetime_exactly_when_python_reads_it_as_iso_8601(text):
    res = ae.validate_column_types(text_column(text), schema={"x": "datetime"})

    try:
        dt.datetime.fromisoformat(text)
        reads = True
    except ValueError:
        reads = False
    assert counts(res.columns["x"]) == ((0, 1, 0, 0) if reads else (0, 0, 1, 0))


@pytest.mark.parametrize("text", ["2024-01-05T23:59:60", "2024-01-05T10:00:00+24:00"])
def test_text_that_is_no_datetime_is_not_inferred_as_one(text):
    res = ae.infer_column_semantics(text_column(text, text, text))

    assert res.columns["x"].inferred_logical_type != "datetime"


def test_a_categorical_of_mixed_values_is_checked_family_by_family():
    categories = ["N/A", -999, True, 2.5, dt.date(2020, 1, 1), "unknown"]
    values = categories * 3 + ["N/A", -999]
    mixed = pd.DataFrame({"x": pd.Categorical(values, categories=categories)})

    res = ae.detect_missing_sentinels(mixed)

    found = {(c.value, c.value_type) for c in res.candidates}
    assert ("N/A", "str") in found
    assert ("unknown", "str") in found
    # True is the boolean it is, not a number a rule suspects.
    assert not any(value is True for value, _ in found)


def test_a_categorical_whose_categories_are_python_strings_is_checked_as_text():
    categories = pd.Index(["N/A", "basic", "pro"], dtype=object)
    df = pd.DataFrame(
        {"plan": pd.Categorical(["basic", "N/A", "pro", "N/A"], categories=categories)}
    )

    res = ae.detect_missing_sentinels(df)

    assert [(c.value, c.count) for c in res.candidates] == [("N/A", 2)]


@pytest.mark.parametrize(
    "categories",
    [
        pytest.param([True, False], id="booleans"),
        pytest.param([dt.timedelta(1), dt.timedelta(2)], id="durations"),
        pytest.param(
            [dt.datetime(2020, 1, 1), dt.datetime(2021, 1, 1)], id="datetimes"
        ),
    ],
)
def test_a_categorical_of_booleans_durations_or_datetimes_has_no_sentinel_text(
    categories,
):
    values = [categories[i % 2] for i in range(10)]
    df = pd.DataFrame({"x": pd.Categorical(values, categories=categories)})

    res = ae.detect_missing_sentinels(df)

    assert res.status == "completed"
    assert [c.value for c in res.candidates if c.value_type == "str"] == []


def test_the_levels_kept_of_a_categorical_of_datetimes_are_the_earliest_when_counts_tie():
    start = dt.datetime(2020, 1, 1)
    days = [start + dt.timedelta(days=(i * 7) % 150) for i in range(150)]
    df = pd.DataFrame({"when": pd.Categorical(days)})

    res = ae.summarize_categorical(df, top_n=100)

    entry = res.columns["when"]
    assert entry.cardinality == 150
    # Every level occurs once, so the result keeps the first 100 by date.
    kept = [pd.Timestamp(lv.label) for lv in entry.levels]
    assert kept == [pd.Timestamp(day) for day in sorted(days)[:100]]
