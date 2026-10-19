import datetime as dt
import inspect
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st
from recipes import missing_value_arguments
from strategies import STR, data_frames

import agent_eda as ae
from agent_eda.errors import ColumnNotFoundError, ColumnTypeError, InvalidArgumentError

# The numbers the ticket lists as stand-ins for missing values.
LISTED_NUMBERS = [-9, -99, -999, -9999, 99, 999, 9999, 99999, 999999]


def suspected(res):
    """Each suspected candidate as (column, value, evidence)."""
    return [
        (c.column, c.value, c.evidence)
        for c in res.candidates
        if c.confirmation == "suspected"
    ]


def test_a_listed_number_at_the_minimum_is_a_suspected_candidate():
    df = pd.DataFrame({"age": [34, -999, 29, 51]})

    res = ae.detect_missing_sentinels(df)

    [candidate] = res.candidates
    assert candidate.column == "age"
    assert candidate.value == -999
    assert candidate.evidence == ["builtin_list"]
    assert candidate.confirmation == "suspected"


@pytest.mark.parametrize("number", LISTED_NUMBERS)
@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_every_listed_number_is_suspected_at_the_minimum_and_at_the_maximum(
    number, dtype
):
    low = pd.DataFrame({"x": [number, number + 5, number + 7]}, dtype=dtype)
    high = pd.DataFrame({"x": [number - 7, number - 5, number]}, dtype=dtype)

    assert suspected(ae.detect_missing_sentinels(low)) == [
        ("x", number, ["builtin_list"])
    ]
    assert suspected(ae.detect_missing_sentinels(high)) == [
        ("x", number, ["builtin_list"])
    ]


def test_a_listed_number_between_the_minimum_and_maximum_is_no_candidate():
    df = pd.DataFrame({"x": [-9999, -999, 5, 7, 99, 120]})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [("x", -9999, ["builtin_list"])]


def test_a_repeated_extreme_farther_from_the_rest_than_they_span_is_suspected():
    # -1 is 11 from the nearest other value, 10, and the others span 2;
    # 77 is 74 from 3, and the others span 12.
    df = pd.DataFrame({"low": [-1, 10, -1, 11, 12], "high": [1, 77, 2, 3, 77]})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("low", -1, ["isolated_extreme"]),
        ("high", 77, ["isolated_extreme"]),
    ]


def test_a_listed_number_that_is_also_an_isolated_extreme_has_both_evidence_codes():
    df = pd.DataFrame({"age": [-999, 34, -999, 29, 51]})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [("age", -999, ["builtin_list", "isolated_extreme"])]


@pytest.mark.parametrize(
    "values",
    [
        [0, 0, 18_000, 61_000, 49_000],  # zeros near the other incomes
        [-12, -3, 0, 4, 9],  # negative temperatures
        [1, 1, 1, 2, 2, 3, 7],  # a rare code, seen once
        [0, 1, 0, 1, 1],  # two values, both repeated
        [0, 0, 5, 10],  # 0 is as far from 5 as 5 is from 10
        [-1, 10, 11, 12],  # an extreme seen once
        [2.5, 2.5, 2.5],  # one value
    ],
    ids=[
        "legitimate-zeros",
        "negatives",
        "rare-code",
        "binary",
        "gap-equal-to-range",
        "extreme-seen-once",
        "constant",
    ],
)
def test_a_value_that_matches_no_rule_is_never_a_candidate(values):
    res = ae.detect_missing_sentinels(pd.DataFrame({"x": values}))

    assert res.candidates == []


def test_missing_and_infinite_values_are_left_out_of_the_rules():
    # Infinity can't be declared as a sentinel value, so it is no candidate,
    # and -999 is the smallest finite value.
    df = pd.DataFrame(
        {
            "score": [np.inf, -999.0, 1.0, np.nan, 2.0, np.inf, -np.inf],
            "visits": pd.array([None, 5, None, 9999, 6, 7, 5], dtype="Int64"),
            "ratio": pd.array(
                [0.5, None, np.inf, np.inf, 0.25, 0.5, 0.75], dtype="Float64"
            ),
        }
    )

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("score", -999.0, ["builtin_list"]),
        ("visits", 9999, ["builtin_list"]),
    ]
    assert [type(c.value) for c in res.candidates] == [float, int]


# The ticket's text stand-ins in several cases, empty and whitespace-only text.
LISTED_TEXT = [
    "",
    " ",
    "\t \n",
    "NA",
    "na",
    "N/A",
    "n/A",
    "NULL",
    "null",
    "None",
    "NONE",
    "nan",
    "NaN",
    "-",
    "?",
    "unknown",
    "Unknown",
    "missing",
    "MISSING",
]
# Text that only resembles them.
UNLISTED_TEXT = ["Lyon", "n.a.", "--", "??", "12", "none yet", "0"]
TEXT_DTYPES = [object, "string", "string[pyarrow]", STR, pd.ArrowDtype(pa.string())]


@pytest.mark.parametrize("dtype", TEXT_DTYPES, ids=str)
def test_listed_text_is_suspected_in_any_case_and_other_text_is_not(dtype):
    df = pd.DataFrame({"note": pd.Series([*LISTED_TEXT, *UNLISTED_TEXT], dtype=dtype)})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [("note", text, ["builtin_list"]) for text in LISTED_TEXT]


def test_listed_dates_at_midnight_are_suspected_and_other_times_are_not():
    times = [
        dt.datetime(1900, 1, 1),
        dt.datetime(2020, 5, 17),
        dt.datetime(1970, 1, 1),
        dt.datetime(1970, 1, 1, 0, 0, 1),
        dt.datetime(2099, 12, 31),
        dt.datetime(9999, 12, 31),
        dt.datetime(2099, 12, 31, 12),
    ]
    # Whole seconds reach the year 9999; nanoseconds stop in 2262.
    df = pd.DataFrame({"when": pd.Series(np.array(times, dtype="datetime64[s]"))})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("when", "1900-01-01T00:00:00", ["builtin_list"]),
        ("when", "1970-01-01T00:00:00", ["builtin_list"]),
        ("when", "2099-12-31T00:00:00", ["builtin_list"]),
        ("when", "9999-12-31T00:00:00", ["builtin_list"]),
    ]


def test_a_datetime_with_a_time_zone_counts_at_midnight_there_or_in_utc():
    local = pd.Series(
        pd.to_datetime(["2099-12-31 00:00", "1970-01-01 01:00", "1970-01-01 12:00"])
    ).dt.tz_localize("Europe/Berlin")

    res = ae.detect_missing_sentinels(pd.DataFrame({"seen": local}))

    # The second is midnight UTC on 1970-01-01, the Unix epoch.
    assert suspected(res) == [
        ("seen", "2099-12-31T00:00:00+01:00", ["builtin_list"]),
        ("seen", "1970-01-01T01:00:00+01:00", ["builtin_list"]),
    ]


def test_listed_dates_are_suspected_in_pyarrow_timestamps_and_dates():
    df = pd.DataFrame(
        {
            "stamp": pd.Series(
                [dt.datetime(1970, 1, 1), None, dt.datetime(1999, 1, 1)],
                dtype=pd.ArrowDtype(pa.timestamp("us")),
            ),
            "day": pd.Series(
                [dt.date(2024, 2, 1), dt.date(1900, 1, 1), None],
                dtype=pd.ArrowDtype(pa.date32()),
            ),
        }
    )

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("stamp", "1970-01-01T00:00:00", ["builtin_list"]),
        ("day", "1900-01-01", ["builtin_list"]),
    ]


def test_an_object_column_is_checked_value_by_value():
    # Its numbers are checked as numbers, its text as text and its dates as
    # dates; True is a boolean, not the number 1, and -999.0 equals -999.
    values = [5, "N/A", -999, True, dt.date(1970, 1, 1), None, "x", 6.5, -999.0]
    df = pd.DataFrame({"mixed": pd.Series(values, dtype=object)})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("mixed", -999, ["builtin_list", "isolated_extreme"]),
        ("mixed", "N/A", ["builtin_list"]),
        ("mixed", "1970-01-01", ["builtin_list"]),
    ]
    assert res.candidates[0].count == 2


@pytest.mark.parametrize(
    "values",
    [["\x00", "", "\x00x"], [5, "\x00", "", "\x00x"]],
    ids=["text", "mixed"],
)
def test_empty_text_stays_apart_from_text_that_starts_with_a_nul(values):
    # pandas' factorize takes '' and text starting with '\x00' for one value.
    df = pd.DataFrame({"x": pd.Series(values, dtype=object)})

    res = ae.detect_missing_sentinels(df)

    assert [(c.value, c.count, c.row_ids) for c in res.candidates] == [
        ("", 1, [values.index("")])
    ]


def test_numbers_of_any_python_type_compare_exactly_and_others_are_left_out():
    # Complex numbers have no order and infinity is no stand-in; decimals,
    # fractions, floats and NumPy numbers compare by value.
    values = [
        complex(1, 2),
        np.float32("inf"),
        Decimal("NaN"),
        Decimal("-999"),
        Fraction(1, 2),
        np.int8(3),
        -999.0,
    ]
    df = pd.DataFrame({"mixed": pd.Series(values, dtype=object)})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [("mixed", -999.0, ["builtin_list", "isolated_extreme"])]
    assert res.candidates[0].count == 2
    assert res.candidates[0].value_type == "Decimal"


def test_a_categorical_is_checked_by_the_categories_its_rows_take():
    df = pd.DataFrame(
        {
            # 'unknown' is a category no row takes.
            "plan": pd.Categorical(
                ["basic", "N/A", "pro", None, "N/A"],
                categories=["basic", "pro", "N/A", "unknown"],
            ),
            # 9 occurs twice, 6 above 3, while 1 to 3 span 2.
            "grade": pd.Categorical([1, 9, 9, 2, 3]),
        }
    )

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("plan", "N/A", ["builtin_list"]),
        ("grade", 9, ["isolated_extreme"]),
    ]
    assert [c.count for c in res.candidates] == [2, 2]


def test_pyarrow_decimals_are_checked_as_numbers():
    decimals = [Decimal("1.50"), Decimal("-999.00"), Decimal("2.25"), None]
    df = pd.DataFrame(
        {"price": pd.Series(decimals, dtype=pd.ArrowDtype(pa.decimal128(10, 2)))}
    )

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [("price", -999.0, ["builtin_list"])]


def test_a_column_of_one_listed_number_is_one_candidate():
    res = ae.detect_missing_sentinels(pd.DataFrame({"x": [-999, -999, -999]}))

    assert suspected(res) == [("x", -999, ["builtin_list"])]
    assert res.candidates[0].count == 3


def expected_number_suspects(numbers):
    """The ticket's number rules, written out plainly: each suspected number with
    its evidence, smallest first."""
    found = {}
    low, high = min(numbers), max(numbers)
    for end in sorted({low, high}):
        evidence = []
        if end in LISTED_NUMBERS:
            evidence.append("builtin_list")
        others = [n for n in numbers if n != end]
        nearest = min(others, key=lambda n: abs(n - end), default=None)
        if (
            numbers.count(end) >= 2
            and len(set(others)) >= 2
            and abs(nearest - end) > max(others) - min(others)
        ):
            evidence.append("isolated_extreme")
        if evidence:
            found[end] = evidence
    return found


SMALL_NUMBERS = st.sampled_from([-9999, -999, -99, -9, 0, 1, 2, 3, 99, 999, 9999])


@given(numbers=st.lists(SMALL_NUMBERS | st.integers(-50, 50), min_size=1, max_size=12))
def test_numbers_are_suspected_exactly_as_the_rules_say(numbers):
    res = ae.detect_missing_sentinels(pd.DataFrame({"x": numbers}))

    found = {value: evidence for _, value, evidence in suspected(res)}
    assert dict(sorted(found.items())) == expected_number_suspects(numbers)


NUMBER_DTYPES = ["int64", "Int64", "float64", "int64[pyarrow]", object, "category"]
TEXTS = st.sampled_from(["", " ", "NA", "n/a", "Null", "x", "Unknown", "-", "N/A "])
DAYS = st.sampled_from(
    [
        dt.datetime(1900, 1, 1),
        dt.datetime(1970, 1, 1),
        dt.datetime(1970, 1, 1, 0, 0, 1),
        dt.datetime(2020, 5, 17),
        dt.datetime(2099, 12, 31),
    ]
)
DATETIME_DTYPES = [
    "datetime64[ns]",
    object,
    "category",
    pd.ArrowDtype(pa.timestamp("us")),
]


def suspects_in_each_dtype(values, dtypes):
    """The suspected candidates of the values held in each dtype."""
    return [
        suspected(ae.detect_missing_sentinels(pd.DataFrame({"x": series})))
        for series in (pd.Series(values, dtype=dtype) for dtype in dtypes)
    ]


@given(
    numbers=st.lists(SMALL_NUMBERS, min_size=1, max_size=8),
    missing=st.lists(st.booleans(), min_size=8, max_size=8),
)
def test_numbers_are_suspected_alike_in_every_dtype(numbers, missing):
    values = [None if gap else n for n, gap in zip(numbers, missing, strict=False)]
    if all(value is None for value in values):
        values.append(0)
    # int64 can't hold a missing value.
    dtypes = NUMBER_DTYPES if None not in values else NUMBER_DTYPES[1:]

    first, *others = suspects_in_each_dtype(values, dtypes)

    assert all(other == first for other in others)


@given(texts=st.lists(TEXTS | st.none(), min_size=1, max_size=8))
def test_text_is_suspected_alike_in_every_dtype(texts):
    first, *others = suspects_in_each_dtype(texts, [*TEXT_DTYPES, "category"])

    assert all(other == first for other in others)


@given(days=st.lists(DAYS | st.none(), min_size=1, max_size=8))
def test_datetimes_are_suspected_alike_in_every_dtype(days):
    first, *others = suspects_in_each_dtype(days, DATETIME_DTYPES)

    assert all(other == first for other in others)


def test_a_candidate_reports_its_count_share_type_and_first_10_row_ids():
    df = pd.DataFrame(
        {"age": [-999] * 12 + [30, None, 50]},
        index=[f"r{i}" for i in range(15)],
    )

    res = ae.detect_missing_sentinels(df)

    [candidate] = res.candidates
    assert candidate.column == "age"
    assert candidate.value == -999.0
    assert candidate.value_type == "float"
    assert candidate.count == 12
    # Over all 15 rows, the missing one included.
    assert candidate.share == 0.8
    assert candidate.row_ids == [f"r{i}" for i in range(10)]
    record = res.to_dict()["candidates"][0]
    assert record["row_ids"] == [f"r{i}" for i in range(10)]
    assert record["row_ids_total"] == 12


def candidates(res):
    """Each candidate as (column, value, confirmation, evidence, count)."""
    return [
        (c.column, c.value, c.confirmation, c.evidence, c.count) for c in res.candidates
    ]


def test_a_declared_value_is_reported_as_declared_and_never_also_as_suspected():
    df = pd.DataFrame(
        {
            "age": [34, -999, 29, -999, 51],
            "city": ["Lyon", "N/A", "n/a", "Oslo", ""],
        }
    )

    res = ae.detect_missing_sentinels(
        df, sentinel_values={"age": [-999], "city": ["N/A"]}
    )

    # Declared values come first in their column; 'n/a' is another value.
    assert candidates(res) == [
        ("age", -999, "declared", ["declared"], 2),
        ("city", "N/A", "declared", ["declared"], 1),
        ("city", "n/a", "suspected", ["builtin_list"], 1),
        ("city", "", "suspected", ["builtin_list"], 1),
    ]


def test_the_rules_look_past_declared_values_as_if_they_were_missing():
    # Undeclared, -999 is the minimum and -998 isn't; declared, -998 is.
    df = pd.DataFrame({"x": [-999, -998, -998, *range(1, 11)]})

    before = ae.detect_missing_sentinels(df)
    after = ae.detect_missing_sentinels(df, sentinel_values={"x": [-999]})

    assert candidates(before) == [("x", -999, "suspected", ["builtin_list"], 1)]
    assert candidates(after) == [
        ("x", -999, "declared", ["declared"], 1),
        ("x", -998, "suspected", ["isolated_extreme"], 2),
    ]


def test_a_declared_value_the_column_lacks_is_no_candidate_but_stays_recorded():
    df = pd.DataFrame({"age": [34, 29, 51]})

    res = ae.detect_missing_sentinels(df, sentinel_values={"age": [-1]})

    assert res.candidates == []
    assert res.parameters["sentinel_values"] == {"age": [-1]}


def test_text_declared_for_an_integer_column_raises_naming_the_number():
    df = pd.DataFrame({"age": [34, -999, 29]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.detect_missing_sentinels(df, sentinel_values={"age": ["-999"]})

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "detect_missing_sentinels got '-999' for column 'age' in sentinel_values, "
        "but 'age' holds integers (dtype int64), so it can never match. Did you mean "
        "-999? Declare each value as the type its column holds, such as "
        "sentinel_values={'age': [-999]}."
    )


def people():
    return pd.DataFrame(
        {
            "age": [34, -999, 29],
            "city": ["Lyon", "N/A", None],
            "n": pd.Series([1, 2, 3], dtype="uint8"),
            "seen": pd.to_datetime(["2024-01-01", None, "1900-01-01"]),
        }
    )


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"sentinel_values": {"age": ["-999"]}}, InvalidArgumentError),
        ({"sentinel_values": {"city": [-1]}}, InvalidArgumentError),
        ({"sentinel_values": {"n": [300]}}, InvalidArgumentError),
        ({"sentinel_values": {"seen": ["1900-01-01"]}}, InvalidArgumentError),
        ({"sentinel_values": {"age": [np.nan]}}, InvalidArgumentError),
        ({"sentinel_values": {"age": [-np.inf]}}, InvalidArgumentError),
        ({"sentinel_values": {"age": [-999, -999.0]}}, InvalidArgumentError),
        ({"sentinel_values": {"Age": [-999]}}, ColumnNotFoundError),
        (
            {"columns": ["city"], "sentinel_values": {"age": [-999]}},
            InvalidArgumentError,
        ),
        ({"sentinel_values": {"age": [None]}}, InvalidArgumentError),
        ({"sentinel_values": [-999]}, InvalidArgumentError),
    ],
    ids=[
        "text-for-integers",
        "number-for-text",
        "beyond-uint8",
        "text-for-datetimes",
        "nan",
        "infinity",
        "listed-twice",
        "unknown-column",
        "column-left-out",
        "none",
        "no-mapping",
    ],
)
def test_declared_values_raise_as_in_profile_missing_values(arguments, error):
    with pytest.raises(error) as here:
        ae.detect_missing_sentinels(people(), **arguments)
    with pytest.raises(error) as there:
        ae.profile_missing_values(people(), **arguments)

    assert here.value.code == there.value.code
    assert str(here.value).replace("detect_missing_sentinels", "<analysis>") == str(
        there.value
    ).replace("profile_missing_values", "<analysis>")


@given(data=st.data())
def test_declared_values_count_as_profile_missing_values_counts_them(data):
    df = data.draw(data_frames(), "data")
    arguments = data.draw(missing_value_arguments(df), "arguments")

    here = ae.detect_missing_sentinels(df, **arguments)
    there = ae.profile_missing_values(df, **arguments)

    for name, entry in there.columns.items():
        counted = [(v.value, v.count) for v in entry.sentinel_values if v.count]
        assert counted == [
            (c.value, c.count)
            for c in here.candidates
            if c.column == name and c.confirmation == "declared"
        ]


def test_the_record_keeps_100_candidates_and_to_frame_has_all_of_them():
    df = pd.DataFrame({f"c{i:03d}": [-999, 1000 + i, 1001 + i] for i in range(120)})

    res = ae.detect_missing_sentinels(df)

    record = res.to_dict()
    assert len(res.candidates) == 120
    assert [c["column"] for c in record["candidates"]] == [
        f"c{i:03d}" for i in range(100)
    ]
    assert record["candidates_total"] == 120
    frame = res.to_frame()
    assert len(frame) == 120
    assert frame["column"].tolist() == [f"c{i:03d}" for i in range(120)]


def test_to_frame_has_one_row_per_candidate_in_record_order():
    df = pd.DataFrame(
        {"age": [34, -999, 29, -999], "city": ["N/A", "Lyon", "", "n/a"]},
        index=["a", "b", "c", "d"],
    )

    frame = ae.detect_missing_sentinels(
        df, sentinel_values={"city": ["N/A"]}
    ).to_frame()

    assert list(frame.columns) == [
        "column",
        "value",
        "value_type",
        "count",
        "share",
        "evidence",
        "confirmation",
        "row_ids",
    ]
    assert frame.to_dict("records") == [
        {
            "column": "age",
            "value": -999,
            "value_type": "int",
            "count": 2,
            "share": 0.5,
            "evidence": ["builtin_list", "isolated_extreme"],
            "confirmation": "suspected",
            "row_ids": ["b", "d"],
        },
        {
            "column": "city",
            "value": "N/A",
            "value_type": "str",
            "count": 1,
            "share": 0.25,
            "evidence": ["declared"],
            "confirmation": "declared",
            "row_ids": ["a"],
        },
        {
            "column": "city",
            "value": "",
            "value_type": "str",
            "count": 1,
            "share": 0.25,
            "evidence": ["builtin_list"],
            "confirmation": "suspected",
            "row_ids": ["c"],
        },
        {
            "column": "city",
            "value": "n/a",
            "value_type": "str",
            "count": 1,
            "share": 0.25,
            "evidence": ["builtin_list"],
            "confirmation": "suspected",
            "row_ids": ["d"],
        },
    ]
    assert frame["count"].dtype == "int64"
    assert frame["share"].dtype == "float64"


def test_declared_values_give_a_descriptive_finding_and_suspected_ones_a_suspected_one():
    df = pd.DataFrame(
        {
            "age": [-999, 34, 29],
            "city": ["N/A", "Lyon", "n/a"],
            "score": [0.5, 0.25, 0.75],
        }
    )

    res = ae.detect_missing_sentinels(df, sentinel_values={"city": ["N/A"]})

    declared, suspicion = res.findings
    assert declared.code == "sentinel_values_present"
    assert declared.assessment == "descriptive"
    assert declared.claim == "Declared sentinel values occur in 1 column: 'city'."
    assert declared.columns == ["city"]
    assert declared.fields == ["candidates[1]"]
    assert suspicion.code == "suspected_sentinel_values"
    assert suspicion.assessment == "suspected"
    assert suspicion.claim == (
        "2 values in 2 columns may stand in for missing values: 'age' and 'city'."
    )
    assert suspicion.columns == ["age", "city"]
    assert suspicion.fields == ["candidates[0]", "candidates[2]"]


def test_a_single_suspected_value_reads_in_the_singular():
    res = ae.detect_missing_sentinels(pd.DataFrame({"age": [-999, 34, 29]}))

    [finding] = res.findings
    assert (
        finding.claim == "1 value in 1 column may stand in for missing values: 'age'."
    )


def test_data_without_candidates_has_no_findings():
    res = ae.detect_missing_sentinels(pd.DataFrame({"age": [41, 34, 29]}))

    assert res.findings == []


def test_declares_the_four_statuses_the_ticket_names_and_its_finding_codes():
    [info] = [
        info
        for info in ae.list_analyses().analyses
        if info.name == "detect_missing_sentinels"
    ]

    assert info.module == "agent_eda.data.missingness"
    assert info.statuses == [
        "completed",
        "insufficient_data",
        "not_applicable",
        "failed",
    ]
    assert info.finding_codes == [
        "sentinel_values_present",
        "suspected_sentinel_values",
    ]


def test_data_with_rows_completes_whether_or_not_a_candidate_turns_up():
    res = ae.detect_missing_sentinels(pd.DataFrame({"age": [41, 34, 29]}))

    assert res.status == "completed"
    assert res.status_reason is None


def test_data_without_rows_gives_insufficient_data():
    df = pd.DataFrame({"age": pd.Series([], dtype="int64"), "city": []})

    res = ae.detect_missing_sentinels(df, sentinel_values={"age": [-999]})

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_rows"
    assert res.status_reason.message == "The data has no rows."
    assert res.candidates == []
    assert res.findings == []


@pytest.mark.parametrize(
    ("df", "id_column", "message"),
    [
        (pd.DataFrame(index=range(3)), None, "The data has no columns to check."),
        (
            pd.DataFrame({"id": [1, 2, 3]}),
            "id",
            "The data has no column to check besides id_column 'id', which "
            "the default columns leave out. Name the columns to check in columns.",
        ),
    ],
    ids=["no-columns", "only-id-column"],
)
def test_no_column_to_check_gives_not_applicable(df, id_column, message):
    res = ae.detect_missing_sentinels(df, id_column=id_column)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_columns_to_check"
    assert res.status_reason.message == message


def orders():
    """Five orders; -999 stands in for two unknown amounts."""
    return pd.DataFrame(
        {
            "order": [101, 102, 103, 104, 105],
            "amount": [9.5, -999.0, 20.0, -999.0, 12.0],
            "note": ["", "gift", "N/A", "", "late"],
        }
    )


def test_row_ids_are_the_id_columns_values_and_columns_none_leaves_it_out():
    res = ae.detect_missing_sentinels(orders(), id_column="order")

    assert res.parameters["columns"] == ["amount", "note"]
    assert res.parameters["id_column"] == "order"
    assert [(c.value, c.row_ids) for c in res.candidates] == [
        (-999.0, [102, 104]),
        ("", [101, 104]),
        ("N/A", [103]),
    ]


def test_a_named_id_column_is_checked_too():
    # Two orders carry the stand-in -999 as their ID.
    df = orders().assign(order=[101, -999, 103, -999, 105])

    res = ae.detect_missing_sentinels(df, columns=["order", "note"], id_column="order")

    assert res.parameters["columns"] == ["order", "note"]
    [stand_in] = [c for c in res.candidates if c.column == "order"]
    assert (stand_in.value, stand_in.count) == (-999, 2)
    # Both rows show the stand-in as their row ID, which the result says.
    assert stand_in.row_ids == [-999, -999]
    assert [lim.code for lim in res.limitations] == ["ambiguous_row_ids"]


def test_an_unknown_id_column_raises_with_the_closest_names():
    with pytest.raises(ColumnNotFoundError) as caught:
        ae.detect_missing_sentinels(orders(), id_column="Order ")

    assert caught.value.code == "column_not_found"
    assert str(caught.value) == (
        "detect_missing_sentinels got an unknown column 'Order ' in id_column. Did "
        "you mean 'order'? The data has 3 columns: 'order', 'amount' and 'note'."
    )


def test_an_id_column_whose_values_cant_be_hashed_raises():
    df = orders().assign(tags=[["a"], [], ["b"], ["a"], []])

    with pytest.raises(ColumnTypeError) as caught:
        ae.detect_missing_sentinels(df, id_column="tags")

    assert caught.value.code == "unsupported_column_type"
    assert "as id_column: it holds values that can't be hashed" in str(caught.value)


def test_repeated_index_labels_make_the_row_ids_that_repeat_ambiguous():
    df = orders().set_index(pd.Index(["a", "b", "c", "b", "e"]))

    res = ae.detect_missing_sentinels(df)

    [limitation] = res.limitations
    assert limitation.code == "ambiguous_row_ids"
    assert limitation.reason == (
        "2 rows share an index label, so a row ID can match several rows: use a "
        "unique id_column. Or make the index unique."
    )
    # -999.0 sits in rows b and b; '' in rows a and b.
    assert limitation.fields == ["candidates[0].row_ids", "candidates[1].row_ids"]


def test_timestamps_as_row_ids_become_iso_8601_text_in_the_record():
    df = orders().set_index(pd.date_range("2024-03-01", periods=5, freq="D"))

    record = ae.detect_missing_sentinels(df).to_dict()

    assert record["candidates"][0]["row_ids"] == [
        "2024-03-02T00:00:00",
        "2024-03-04T00:00:00",
    ]


def contacts():
    """Five contacts; email is sensitive and stands in 'N/A' and '' for unknown."""
    return pd.DataFrame(
        {
            "customer": ["c1", "c2", "c3", "c4", "c5"],
            "email": ["ann@x.org", "N/A", "", "N/A", "bob@x.org"],
            "age": [34, -999, 29, -999, 51],
        }
    )


def masking_project(*columns):
    return ae.Project(
        data_policy=ae.DataPolicy(sensitive_columns=list(columns), mask_sensitive=True)
    )


def everything_shown(res):
    """The printed text, the record and the table, as one string."""
    return str(res) + res.to_json() + res.to_frame().to_csv()


def flowing(text):
    """The text with its wrapped lines joined again."""
    return " ".join(line.strip() for line in text.splitlines())


def test_masking_puts_numbered_placeholders_in_place_of_sensitive_candidates():
    res = ae.detect_missing_sentinels(contacts(), project=masking_project("email"))

    assert [(c.column, c.value, c.value_type, c.count) for c in res.candidates] == [
        ("email", "<masked 1>", "str", 2),
        ("email", "<masked 2>", "str", 1),
        ("age", -999, "int", 2),
    ]
    assert "N/A" not in everything_shown(res)
    assert res.masking.applied is True
    assert res.masking.columns == ["email"]
    assert (
        "Masked (data_policy.mask_sensitive=True): the values of column 'email' show "
        "as <masked 1>, <masked 2> and so on; counts and shares are unaffected."
    ) in flowing(str(res))


def test_masking_puts_placeholders_in_place_of_row_ids_from_a_sensitive_id_column():
    res = ae.detect_missing_sentinels(
        contacts(), id_column="customer", project=masking_project("customer")
    )

    # Equal IDs share a placeholder, numbered in the order the result shows them.
    assert [c.row_ids for c in res.candidates] == [
        ["<masked 1>", "<masked 2>"],
        ["<masked 3>"],
        ["<masked 1>", "<masked 2>"],
    ]
    assert "c2" not in everything_shown(res)
    assert res.to_frame()["row_ids"].tolist()[1] == ["<masked 3>"]
    assert res.masking.columns == ["customer"]


def test_a_sensitive_id_column_that_is_also_checked_shares_its_placeholders():
    df = pd.DataFrame({"code": ["N/A", "x1", "N/A", "x2"], "visits": [1, 2, 3, 4]})

    res = ae.detect_missing_sentinels(
        df, columns=["code"], id_column="code", project=masking_project("code")
    )

    [candidate] = res.candidates
    assert candidate.value == "<masked 1>"
    assert candidate.row_ids == ["<masked 1>", "<masked 1>"]


def test_declared_values_of_a_masked_column_show_as_its_placeholders():
    res = ae.detect_missing_sentinels(
        contacts(),
        sentinel_values={"email": ["N/A"], "age": [-999]},
        project=masking_project("email"),
    )

    assert res.parameters["sentinel_values"] == {
        "email": ["<masked 1>"],
        "age": [-999],
    }
    assert [(c.value, c.confirmation) for c in res.candidates] == [
        ("<masked 1>", "declared"),
        ("<masked 2>", "suspected"),
        (-999, "declared"),
    ]
    assert "N/A" not in everything_shown(res)


def test_different_masked_values_get_different_placeholders():
    # pandas' factorize would take '' and '\x00a' for one value.
    df = pd.DataFrame({"code": ["\x00a", "", "b"], "age": [-999, -999, 5]})

    res = ae.detect_missing_sentinels(
        df, columns=["age"], id_column="code", project=masking_project("code")
    )

    assert res.candidates[0].row_ids == ["<masked 1>", "<masked 2>"]


def test_index_labels_are_never_masked():
    df = contacts().set_index("customer")

    res = ae.detect_missing_sentinels(df, project=masking_project("customer"))

    assert res.candidates[0].row_ids == ["c2", "c4"]
    assert res.masking.index_columns == ["customer"]


def test_with_masking_off_sensitive_values_show_as_stored_and_the_text_says_so():
    project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["email"]))

    res = ae.detect_missing_sentinels(contacts(), project=project)

    assert [c.value for c in res.candidates][:2] == ["N/A", ""]
    assert res.masking.applied is False
    assert res.masking.columns == ["email"]
    assert "Not masked (data_policy.mask_sensitive=False)" in str(res)


def test_no_rule_reads_booleans_durations_or_lists_and_the_result_names_them():
    df = pd.DataFrame(
        {
            "flag": [True, False, False, False],
            "took": pd.to_timedelta([0, 1, 999, 999], unit="D"),
            "age": [34, 51, 29, 41],
            "tags": pd.Series([["a"], [], None, ["b"]], dtype=object),
            "nothing": pd.Series([None] * 4, dtype=object),
            "answer": pd.Categorical([True, False, None, True]),
        }
    )

    res = ae.detect_missing_sentinels(df)

    assert res.candidates == []
    # A column without values has nothing to check, whatever its dtype.
    assert res.columns_without_rules == ["flag", "took", "tags", "answer"]


def test_the_printed_result_lists_the_candidates_most_frequent_first():
    res = ae.detect_missing_sentinels(orders(), id_column="order")

    assert str(res).splitlines() == [
        "detect_missing_sentinels: completed on 5 rows x 3 columns",
        "Checked 2 columns: 3 suspected candidates. Shares are of all rows; row IDs "
        "are values of id_column 'order'.",
        "Suspected values stay valid until declared in sentinel_values, which "
        "profile_missing_values counts as missing.",
        "Candidates, most frequent first (3 of 3 shown):",
        "column  value   count  share  confirmation  evidence                        "
        "row_ids",
        "amount  -999.0      2    0.4  suspected     builtin_list, isolated_extreme  "
        "102 and 104",
        "note    ''          2    0.4  suspected     builtin_list                    "
        "101 and 104",
        "note    'N/A'       1    0.2  suspected     builtin_list                    103",
        "Findings (1 of 1 shown):",
        "- suspected suspected_sentinel_values: 3 values in 2 columns may stand in for "
        "missing values: 'amount' and 'note'.",
        "Limitations: none",
        "More: res.to_frame() has all 3 candidates; res.findings, res.limitations and "
        "res.to_json() have the rest.",
    ]


def test_the_printed_result_names_3_row_ids_and_counts_the_rest():
    df = pd.DataFrame({"age": [-999] * 25 + [30, 40]})

    res = ae.detect_missing_sentinels(df)

    assert "0, 1, 2 and 22 more" in str(res)


def test_the_printed_result_without_candidates_says_so_and_names_columns_no_rule_reads():
    df = pd.DataFrame({"age": [41, 34, 29], "flag": [True, False, True]})

    lines = str(ae.detect_missing_sentinels(df)).splitlines()

    assert lines[1:4] == [
        "Checked 2 columns: no candidates. Shares are of all rows; row IDs are index "
        "labels.",
        "No built-in rule reads 1 column, so only declared values were looked for: "
        "'flag'.",
        "Candidates, most frequent first: none",
    ]


def test_the_printed_result_shows_placeholders_without_quotes():
    res = ae.detect_missing_sentinels(
        contacts(), id_column="customer", project=masking_project("customer", "email")
    )

    lines = str(res).splitlines()
    assert any(
        line.startswith("email   <masked 1>      2    0.4  suspected") for line in lines
    )
    assert any(line.endswith("<masked 1> and <masked 2>") for line in lines)


@pytest.mark.parametrize(
    "neighbour",
    [
        "profile_missing_values",
        "describe_dataset",
        "summarize_numeric",
        "summarize_categorical",
    ],
)
def test_nearby_analyses_point_to_it_for_stand_ins_for_missing_values(neighbour):
    doc = inspect.getdoc(getattr(ae, neighbour))
    assert doc

    assert "`detect_missing_sentinels`" in ae.describe_analysis(neighbour).when_to_use
    assert "    detect_missing_sentinels: " in doc.split("See Also:")[1]


def test_support_counts_the_values_checked_in_each_column():
    df = pd.DataFrame({"age": [34, None, -999], "flag": [True, False, True]})

    res = ae.detect_missing_sentinels(df)

    assert res.support.row_count == 3
    assert res.support.column_count == 2
    assert [
        (
            s.column,
            s.eligible_count,
            s.evaluated_count,
            s.excluded_count,
            s.excluded_by_reason,
        )
        for s in res.support.columns
    ] == [("age", 3, 2, 1, {"missing": 1})]


@pytest.mark.parametrize(
    "analysis", ["detect_missing_sentinels", "profile_missing_values"]
)
def test_true_and_one_can_both_be_declared_for_an_object_column(analysis):
    # Matching keeps them apart: True matches the True, and 1 matches 1 and 1.0.
    df = pd.DataFrame({"o": pd.Series([True, 1, 1.0, "1"], dtype=object)})

    res = getattr(ae, analysis)(df, sentinel_values={"o": [True, 1]})

    assert res.status == "completed"
    if analysis == "detect_missing_sentinels":
        assert [(c.value, c.count) for c in res.candidates] == [(True, 1), (1, 2)]
    else:
        declared = res.columns["o"].sentinel_values
        assert [(d.value, d.count) for d in declared] == [(True, 1), (1, 2)]


@pytest.mark.parametrize("values", [[1, 1.0], [True, True], [-999, -999]])
def test_a_value_declared_twice_still_raises_when_both_are_one_value(values):
    df = pd.DataFrame({"o": pd.Series([True, 1, 1.0, -999], dtype=object)})

    with pytest.raises(InvalidArgumentError, match="more than once in sentinel_values"):
        ae.detect_missing_sentinels(df, sentinel_values={"o": values})
