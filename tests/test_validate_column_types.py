import datetime as dt
import inspect
from decimal import Decimal
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import data_frames

import agent_eda as ae
from agent_eda.errors import ColumnNotFoundError, ColumnTypeError, InvalidArgumentError

NAN = float("nan")
DAY = dt.datetime(2024, 1, 5)


def objects(*values):
    return pd.Series(list(values), dtype=object)


# Each row: a column, the logical type declared for it, and the counts it must
# give: compatible, convertible, invalid and missing.
RULES = [
    # boolean
    ("bool", pd.Series([True, False, True]), "boolean", (3, 0, 0, 0)),
    (
        "nullable boolean",
        pd.Series([True, None], dtype="boolean"),
        "boolean",
        (1, 0, 0, 1),
    ),
    (
        "bool[pyarrow]",
        pd.Series([False, None], dtype="bool[pyarrow]"),
        "boolean",
        (1, 0, 0, 1),
    ),
    ("0/1 integers", pd.Series([0, 1, 2]), "boolean", (0, 2, 1, 0)),
    ("0/1 floats", pd.Series([0.0, 1.0, 0.5, NAN]), "boolean", (0, 2, 1, 1)),
    (
        "boolean words",
        objects("yes", "No", "1", "T", "maybe", None),
        "boolean",
        (0, 4, 1, 1),
    ),
    ("datetimes as booleans", pd.Series([DAY]), "boolean", (0, 0, 1, 0)),
    # integer
    ("Int64", pd.Series([1, None], dtype="Int64"), "integer", (1, 0, 0, 1)),
    ("uint8", pd.Series([1, 255], dtype="uint8"), "integer", (2, 0, 0, 0)),
    (
        "int64[pyarrow]",
        pd.Series([7, None], dtype="int64[pyarrow]"),
        "integer",
        (1, 0, 0, 1),
    ),
    ("Decimal objects", objects(Decimal("2"), Decimal("2.5")), "integer", (0, 1, 1, 0)),
    (
        "decimal128[pyarrow]",
        pd.Series(
            [Decimal("2.00"), Decimal("2.50"), None],
            dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
        ),
        "integer",
        (0, 1, 1, 1),
    ),
    ("datetimes as integers", pd.Series([DAY, None]), "integer", (0, 0, 1, 1)),
    # float
    ("float64", pd.Series([1.5, float("inf"), NAN]), "float", (2, 0, 0, 1)),
    ("float32", pd.Series([1.5, 2.0], dtype="float32"), "float", (2, 0, 0, 0)),
    ("int64 as floats", pd.Series([1, 2]), "float", (0, 2, 0, 0)),
    ("decimals as floats", objects(Decimal("0.1")), "float", (0, 1, 0, 0)),
    (
        "text as floats",
        objects("1.5", "1e3", "-.5", "inf", "1,5", "0x10"),
        "float",
        (0, 3, 3, 0),
    ),
    ("booleans as floats", pd.Series([True]), "float", (0, 0, 1, 0)),
    # datetime
    ("datetime64[ns]", pd.Series([DAY, None]), "datetime", (1, 0, 0, 1)),
    (
        "aware datetimes",
        pd.Series([DAY]).dt.tz_localize("UTC"),
        "datetime",
        (1, 0, 0, 0),
    ),
    (
        "date32[pyarrow]",
        pd.Series([dt.date(2024, 1, 5), None], dtype=pd.ArrowDtype(pa.date32())),
        "datetime",
        (1, 0, 0, 1),
    ),
    ("datetime objects", objects(DAY, dt.date(2024, 1, 5)), "datetime", (2, 0, 0, 0)),
    (
        "ISO 8601 text",
        objects("2024-01-05", "2024-01-05T10:00:00Z", "05/01/2024", "2024-02-30"),
        "datetime",
        (0, 2, 2, 0),
    ),
    ("numbers as datetimes", pd.Series([1704412800]), "datetime", (0, 0, 1, 0)),
    # timedelta
    ("timedelta64", pd.Series([dt.timedelta(days=1), None]), "timedelta", (1, 0, 0, 1)),
    (
        "duration[pyarrow]",
        pd.Series([dt.timedelta(hours=2)], dtype=pd.ArrowDtype(pa.duration("us"))),
        "timedelta",
        (1, 0, 0, 0),
    ),
    ("durations as text", objects("1 day", "P1D"), "timedelta", (0, 0, 2, 0)),
    ("numbers as durations", pd.Series([3.5]), "timedelta", (0, 0, 1, 0)),
    # categorical
    (
        "category",
        pd.Series(["a", None, "b"], dtype="category"),
        "categorical",
        (2, 0, 0, 1),
    ),
    ("text as categories", objects("a", "b", "a"), "categorical", (0, 3, 0, 0)),
    ("numbers as categories", pd.Series([1.5, 2.5, NAN]), "categorical", (0, 2, 0, 1)),
    ("lists as categories", objects(["a"], "b", None), "categorical", (0, 1, 1, 1)),
    # text
    ("object text", objects("a", " ", "", None), "text", (3, 0, 0, 1)),
    (
        "string[pyarrow]",
        pd.Series(["a", None], dtype="string[pyarrow]"),
        "text",
        (1, 0, 0, 1),
    ),
    (
        "arrow string",
        pd.Series(["a"], dtype=pd.ArrowDtype(pa.string())),
        "text",
        (1, 0, 0, 0),
    ),
    (
        "values of any type as text",
        objects("a", 1, 2.5, True, DAY, dt.timedelta(days=1)),
        "text",
        (1, 5, 0, 0),
    ),
    ("lists as text", objects(["a"], {"k": 1}), "text", (0, 0, 2, 0)),
    ("numbers as text", pd.Series([1, 2]), "text", (0, 2, 0, 0)),
    # A categorical stores categories: their values convert to the type they hold.
    (
        "category of integers",
        pd.Series([1, 2, None], dtype="category"),
        "integer",
        (0, 2, 0, 1),
    ),
    (
        "category of words",
        pd.Series(["yes", "no", "x"], dtype="category"),
        "boolean",
        (0, 2, 1, 0),
    ),
    ("category of text", pd.Series(["a", "b"], dtype="category"), "text", (0, 2, 0, 0)),
    # A category is read by its label, so a stored category is never compatible
    # with the other types, and each row counts as its category does.
    (
        "category of booleans as booleans",
        pd.Series([True, False, None, True], dtype="category"),
        "boolean",
        (0, 3, 0, 1),
    ),
    (
        "category of booleans as integers",
        pd.Series([True, False, None, True], dtype="category"),
        "integer",
        (0, 0, 3, 1),
    ),
    (
        "category of datetimes as datetimes",
        pd.Series([DAY, DAY, None, dt.datetime(2021, 1, 1)], dtype="category"),
        "datetime",
        (0, 3, 0, 1),
    ),
    (
        "category of datetimes as durations",
        pd.Series([DAY, DAY, None, dt.datetime(2021, 1, 1)], dtype="category"),
        "timedelta",
        (0, 0, 3, 1),
    ),
    (
        "category of durations as durations",
        pd.Series([dt.timedelta(1), dt.timedelta(2), None], dtype="category"),
        "timedelta",
        (0, 2, 0, 1),
    ),
    (
        "category of durations as text",
        pd.Series([dt.timedelta(1), dt.timedelta(2), None], dtype="category"),
        "text",
        (0, 2, 0, 1),
    ),
    (
        "category of mixed values as floats",
        pd.Series(["a", 1, 2.5, dt.date(2020, 1, 1)], dtype="category"),
        "float",
        (0, 2, 2, 0),
    ),
    (
        "category of mixed values as datetimes",
        pd.Series(["a", 1, 2.5, dt.date(2020, 1, 1)], dtype="category"),
        "datetime",
        (0, 1, 3, 0),
    ),
    (
        "category of mixed values as text",
        pd.Series(["a", 1, 2.5, dt.date(2020, 1, 1)], dtype="category"),
        "text",
        (0, 4, 0, 0),
    ),
    # Objects are read by their own type, though pandas counts 1, 1.0 and True as one.
    (
        "mixed objects as integers",
        objects(1, 1.0, True, "1", None),
        "integer",
        (1, 2, 1, 1),
    ),
    (
        "mixed objects as booleans",
        objects(1, 1.0, True, "1", None),
        "boolean",
        (1, 3, 0, 1),
    ),
    (
        "mixed objects as floats",
        objects(1, 1.0, True, "1", None),
        "float",
        (1, 2, 1, 1),
    ),
]


@pytest.mark.parametrize(
    ("values", "declared", "expected"),
    [
        pytest.param(values, declared, expected, id=name)
        for name, values, declared, expected in RULES
    ],
)
def test_each_value_is_compatible_convertible_invalid_or_missing(
    values, declared, expected
):
    df = pd.DataFrame({"x": values})

    res = ae.validate_column_types(df, schema={"x": declared})

    assert counts(res.columns["x"]) == expected


def test_text_in_the_declared_datetime_format_converts_as_iso_8601_does():
    df = pd.DataFrame(
        {"x": objects("05/01/2024", " 2024-01-05 ", "31/02/2024", "5.1.2024")}
    )

    res = ae.validate_column_types(
        df, schema={"x": ae.ColumnSpec("datetime", datetime_format="%d/%m/%Y")}
    )

    assert counts(res.columns["x"]) == (0, 2, 2, 0)
    assert res.columns["x"].datetime_format == "%d/%m/%Y"


def test_values_of_different_types_count_apart_as_invalid_values():
    df = pd.DataFrame({"x": objects(1, True, 1, "a")})

    res = ae.validate_column_types(df, schema={"x": "datetime"})

    entry = res.columns["x"]
    assert [(v.value, v.value_type, v.count) for v in entry.invalid_values] == [
        (1, "int", 2),
        (True, "bool", 1),
        ("a", "str", 1),
    ]
    assert entry.invalid_distinct_count == 3


def counts(entry):
    """An entry's four counts: compatible, convertible, invalid and missing."""
    return (
        entry.compatible_count,
        entry.convertible_count,
        entry.invalid_count,
        entry.missing_count,
    )


def test_values_stored_as_the_declared_logical_type_are_compatible():
    df = pd.DataFrame({"age": [34, 51, 29]})

    res = ae.validate_column_types(df, schema={"age": "integer"})

    entry = res.columns["age"]
    assert entry.declared_logical_type == "integer"
    assert entry.dtype == "int64"
    assert entry.present is True
    assert counts(entry) == (3, 0, 0, 0)
    assert res.status == "completed"


def test_schema_maps_names_to_a_logical_type_a_column_spec_or_its_fields():
    df = pd.DataFrame({"a": [1], "b": [2.5], "c": ["x"], 0: [True]})
    schema: dict[Any, Any] = {
        "a": "integer",
        "b": ae.ColumnSpec("float", nullable=False),
        "c": {"logical_type": "text", "nullable": True},
        # A label that isn't a string can be given as itself.
        0: "boolean",
    }

    res = ae.validate_column_types(df, schema=schema)

    assert [(e.declared_logical_type, e.nullable) for e in res.columns.values()] == [
        ("integer", None),
        ("float", False),
        ("text", True),
        ("boolean", None),
    ]
    assert list(res.columns) == ["a", "b", "c", "0"]
    assert res.to_dict()["parameters"]["schema"]["b"] == {
        "logical_type": "float",
        "nullable": False,
        "datetime_format": None,
    }


def test_leaving_out_schema_raises_with_what_it_takes():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.validate_column_types(pd.DataFrame({"a": [1]}))  # ty: ignore[missing-argument]  # leaving schema out is the test

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == (
        "validate_column_types is missing 1 required argument: schema. Pass it by "
        "keyword, such as ae.validate_column_types(df, schema=...), where each value "
        "in schema is one of 'boolean', 'integer', 'float', 'categorical', 'text', "
        "'datetime' or 'timedelta', or a ColumnSpec. See "
        "help(ae.validate_column_types) for every argument."
    )


def test_an_empty_schema_raises_since_it_declares_nothing_to_check():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.validate_column_types(pd.DataFrame({"a": [1]}), schema={})

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "validate_column_types got an empty schema, which declares no column to "
        "check. Declare the logical type of at least one column, such as "
        "schema={'age': 'integer'}."
    )


def test_an_unknown_logical_type_raises_with_the_closest_one():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.validate_column_types(pd.DataFrame({"a": [1]}), schema={"a": "int"})  # ty: ignore[invalid-argument-type]  # the unknown type is the test

    assert caught.value.code == "invalid_argument"
    assert "Did you mean 'integer'?" in str(caught.value)


def test_whole_floats_and_whole_numbers_written_as_text_convert_to_integer():
    df = pd.DataFrame(
        {
            "floats": [42.0, 7.0, None],
            "text": pd.Series(["42", " 7 ", "4.2e1"], dtype=object),
        }
    )

    res = ae.validate_column_types(df, schema={"floats": "integer", "text": "integer"})

    assert counts(res.columns["floats"]) == (0, 2, 0, 1)
    assert counts(res.columns["text"]) == (0, 3, 0, 0)


def test_values_that_neither_have_nor_convert_to_the_declared_type_are_invalid():
    df = pd.DataFrame(
        {
            "floats": [4.5, float("inf"), 3.0, None],
            # A whole number with a leading zero is a code: a number would drop it.
            "text": pd.Series(["4.5", "abc", "007", None], dtype=object),
            "flags": [True, False, True, False],
        }
    )

    res = ae.validate_column_types(
        df, schema={"floats": "integer", "text": "integer", "flags": "integer"}
    )

    assert counts(res.columns["floats"]) == (0, 1, 2, 1)
    assert counts(res.columns["text"]) == (0, 0, 3, 1)
    assert counts(res.columns["flags"]) == (0, 0, 4, 0)


def codes():
    """14 product codes; 12 aren't integers, 'n/a' three times and 'x' twice."""
    values = [
        "1",
        "n/a",
        "x",
        "n/a",
        "2",
        "y",
        "n/a",
        "z",
        "w",
        "v",
        "x",
        "u",
        "t",
        "s",
    ]
    index = [f"r{i}" for i in range(len(values))]
    return pd.DataFrame({"code": pd.Series(values, index=index, dtype=object)})


def test_invalid_values_come_with_the_first_10_row_ids_and_their_total():
    res = ae.validate_column_types(codes(), schema={"code": "integer"})

    entry = res.columns["code"]
    assert entry.invalid_count == 12
    assert entry.invalid_row_ids == [
        "r1",
        "r2",
        "r3",
        "r5",
        "r6",
        "r7",
        "r8",
        "r9",
        "r10",
        "r11",
    ]
    assert res.to_dict()["columns"]["code"]["invalid_row_ids_total"] == 12


def test_invalid_values_come_with_5_distinct_examples_most_frequent_first():
    res = ae.validate_column_types(codes(), schema={"code": "integer"})

    entry = res.columns["code"]
    # Ties go by value, in the order summarize_categorical gives levels.
    assert [(v.value, v.value_type, v.count) for v in entry.invalid_values] == [
        ("n/a", "str", 3),
        ("x", "str", 2),
        ("s", "str", 1),
        ("t", "str", 1),
        ("u", "str", 1),
    ]
    assert entry.invalid_distinct_count == 9
    assert res.to_dict()["columns"]["code"]["invalid_values_total"] == 9


def test_entries_follow_the_declared_order_and_undeclared_columns_are_listed():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})

    res = ae.validate_column_types(df, schema={"c": "integer", "a": "integer"})

    assert list(res.columns) == ["c", "a"]
    assert res.undeclared_columns == ["b", "d"]
    assert res.to_dict()["undeclared_columns_total"] == 2


def test_a_declared_column_missing_from_the_data_is_a_violation_not_an_error():
    df = pd.DataFrame({"email": ["ann@x.org", "bob@x.org"], "age": [34, 51]})

    res = ae.validate_column_types(df, schema={"emial": "text", "age": "integer"})

    entry = res.columns["emial"]
    assert entry.present is False
    assert entry.dtype is None
    assert counts(entry) == (None, None, None, None)
    assert (entry.invalid_row_ids, entry.invalid_values) == ([], [])
    assert res.status == "completed"
    [finding] = [f for f in res.findings if f.code == "declared_columns_missing"]
    assert finding.assessment == "violation"
    assert finding.columns == ["emial"]
    assert finding.claim == (
        "1 declared column is not in the data: 'emial' (did you mean 'email'?)."
    )
    assert finding.fields == ["columns['emial'].present"]


def findings_by_code(res):
    return {finding.code: finding for finding in res.findings}


def test_invalid_values_give_a_violation_naming_the_columns():
    res = ae.validate_column_types(codes(), schema={"code": "integer"})

    finding = findings_by_code(res)["invalid_values"]
    assert finding.assessment == "violation"
    assert finding.columns == ["code"]
    assert finding.claim == "1 column holds invalid values: 'code'."
    assert finding.fields == ["columns['code'].invalid_count"]
    assert res.columns["code"].assessment == "violation"


def test_missing_values_in_a_column_declared_non_nullable_give_a_violation():
    df = pd.DataFrame({"id": [1.0, None, 3.0], "note": ["a", None, "c"]})

    res = ae.validate_column_types(
        df,
        schema={
            "id": ae.ColumnSpec("float", nullable=False),
            "note": ae.ColumnSpec("text", nullable=True),
        },
    )

    finding = findings_by_code(res)["non_nullable_missing_values"]
    assert finding.assessment == "violation"
    assert finding.columns == ["id"]
    assert finding.claim == "1 column declared non-nullable has missing values: 'id'."
    assert finding.fields == ["columns['id'].missing_count"]
    assert res.columns["id"].assessment == "violation"
    assert res.columns["id"].nullable is False
    assert res.columns["note"].assessment == "passed"


def test_a_clean_column_passes_and_a_column_without_values_is_unknown():
    df = pd.DataFrame({"age": [34, 51], "notes": pd.Series([None, None], dtype=object)})

    res = ae.validate_column_types(df, schema={"age": "integer", "notes": "text"})

    found = findings_by_code(res)
    assert (found["valid_columns"].assessment, found["valid_columns"].columns) == (
        "passed",
        ["age"],
    )
    assert found["valid_columns"].claim == "1 column passes every check: 'age'."
    assert (
        found["unchecked_columns"].assessment,
        found["unchecked_columns"].columns,
    ) == (
        "unknown",
        ["notes"],
    )
    assert (
        found["unchecked_columns"].claim == "1 column has no value to check: 'notes'."
    )
    assert [res.columns[c].assessment for c in ("age", "notes")] == [
        "passed",
        "unknown",
    ]


def test_a_column_whose_every_value_is_invalid_was_checked():
    df = pd.DataFrame({"qty": objects("n/a", "none", None)})

    res = ae.validate_column_types(df, schema={"qty": "integer"})

    assert [f.code for f in res.findings] == ["invalid_values"]
    assert res.columns["qty"].assessment == "violation"


def test_values_that_need_converting_are_described_and_still_pass():
    df = pd.DataFrame({"n": pd.Series(["1", "2", "3"], dtype=object)})

    res = ae.validate_column_types(df, schema={"n": "integer"})

    found = findings_by_code(res)
    assert found["convertible_values"].assessment == "descriptive"
    assert (
        found["convertible_values"].claim == "1 column holds convertible values: 'n'."
    )
    assert found["convertible_values"].fields == ["columns['n'].convertible_count"]
    assert found["valid_columns"].columns == ["n"]


def test_row_ids_are_values_of_id_column_when_it_is_given():
    df = pd.DataFrame({"order": [101, 102, 103], "qty": objects("1", "two", "3")})

    res = ae.validate_column_types(df, schema={"qty": "integer"}, id_column="order")

    assert res.columns["qty"].invalid_row_ids == [102]
    assert res.parameters["id_column"] == "order"


def test_an_unknown_id_column_raises_with_the_closest_names():
    df = pd.DataFrame({"order": [101], "qty": [1]})

    with pytest.raises(ColumnNotFoundError) as caught:
        ae.validate_column_types(df, schema={"qty": "integer"}, id_column="Order")

    assert str(caught.value) == (
        "validate_column_types got an unknown column 'Order' in id_column. Did you "
        "mean 'order'? The data has 2 columns: 'order' and 'qty'."
    )


def test_an_id_column_whose_values_cant_be_hashed_raises():
    df = pd.DataFrame({"tags": objects(["a"], ["b"]), "qty": objects("1", "x")})

    with pytest.raises(ColumnTypeError) as caught:
        ae.validate_column_types(df, schema={"qty": "integer"}, id_column="tags")

    assert caught.value.code == "unsupported_column_type"
    assert "can't use column 'tags' with dtype object as id_column" in str(caught.value)


def test_repeated_row_ids_give_an_ambiguous_row_ids_limitation():
    df = pd.DataFrame({"qty": objects("1", "two", "x", "4")})
    df.index = [7, 7, 8, 9]

    res = ae.validate_column_types(df, schema={"qty": "integer"})

    assert res.columns["qty"].invalid_row_ids == [7, 8]
    [limitation] = res.limitations
    assert limitation.code == "ambiguous_row_ids"
    assert limitation.fields == ["columns['qty'].invalid_row_ids"]


def masked(*columns):
    """A project that masks the given columns."""
    policy = ae.DataPolicy(sensitive_columns=list(columns), mask_sensitive=True)
    return ae.Project(data_policy=policy)


def customers():
    """Four customers; email and phone hold personal data."""
    return pd.DataFrame(
        {
            "email": ["ann@x.org", "bob@x.org", "cy@x.org", "dan@x.org"],
            "age": objects("34", "unknown", "51", "n/a"),
            "phone": objects("555-1234", "555-1234", "12", "n/a"),
        }
    )


def test_with_masking_on_invalid_values_of_sensitive_columns_become_placeholders():
    res = ae.validate_column_types(
        customers(),
        schema={"phone": "integer", "age": "integer"},
        project=masked("phone"),
    )

    phone = res.columns["phone"]
    assert [(v.value, v.value_type, v.count) for v in phone.invalid_values] == [
        ("<masked 1>", "str", 2),
        ("<masked 2>", "str", 1),
    ]
    assert [v.value for v in res.columns["age"].invalid_values] == ["n/a", "unknown"]
    assert "555" not in str(res) + res.to_json() + res.to_frame().to_csv()
    assert (res.masking.applied, res.masking.columns) == (True, ["phone"])


def test_with_masking_on_row_ids_from_a_sensitive_id_column_become_placeholders():
    res = ae.validate_column_types(
        customers(),
        schema={"age": "integer"},
        id_column="email",
        project=masked("email"),
    )

    assert res.columns["age"].invalid_row_ids == ["<masked 1>", "<masked 2>"]
    assert "@" not in str(res) + res.to_json() + res.to_frame().to_csv()
    assert res.masking.columns == ["email"]


def test_index_labels_are_never_masked_even_when_they_hold_a_sensitive_column():
    df = customers().set_index("email")

    res = ae.validate_column_types(
        df, schema={"age": "integer"}, project=masked("email")
    )

    assert res.columns["age"].invalid_row_ids == ["bob@x.org", "dan@x.org"]
    assert res.masking.index_columns == ["email"]


def test_a_sensitive_id_column_shows_each_value_as_one_placeholder():
    df = pd.DataFrame(
        {"code": objects("a1", "7", "a1", "b2"), "qty": objects("x", "z", "y", "2")}
    )

    res = ae.validate_column_types(
        df,
        schema={"code": "integer", "qty": "integer"},
        id_column="code",
        project=masked("code"),
    )

    # The invalid values of code come first, in level order: 'a1' twice, 'b2'.
    assert [v.value for v in res.columns["code"].invalid_values] == [
        "<masked 1>",
        "<masked 2>",
    ]
    assert res.columns["code"].invalid_row_ids == [
        "<masked 1>",
        "<masked 1>",
        "<masked 2>",
    ]
    # Row IDs equal to an invalid value share its placeholder; '7' gets the next.
    assert res.columns["qty"].invalid_row_ids == [
        "<masked 1>",
        "<masked 3>",
        "<masked 1>",
    ]


def orders():
    """Five orders as read from a CSV file: every column holds text."""
    return pd.DataFrame(
        {
            "order_id": objects("1001", "1002", "1003", "1004", "1005"),
            "qty": objects("2", "1", "n/a", "3", "n/a"),
            "ordered": objects(
                "2024-01-05", "2024-01-06", None, "05/01/2024", "2024-01-09"
            ),
            "note": objects("gift", None, None, "", "rush"),
        }
    )


def test_prints_the_counts_per_declared_column_violations_first():
    res = ae.validate_column_types(
        orders(),
        schema={
            "order_id": ae.ColumnSpec("integer", nullable=False),
            "ordered": "datetime",
            "qty": "integer",
            "customer": "text",
        },
    )

    lines = str(res).splitlines()

    assert lines[0] == "validate_column_types: completed on 5 rows x 4 columns"
    assert "Not declared in schema, so not checked: 'note'." in lines
    start = lines.index("Declared columns, violations first (4 of 4 shown):")
    assert [line.split()[0] for line in lines[start + 2 : start + 6]] == [
        "ordered",
        "qty",
        "customer",
        "order_id",
    ]
    assert lines[start + 2].endswith("'05/01/2024': 1")
    # Findings print violation first, then descriptive, then passed.
    findings = [line for line in lines if line.startswith("- ")]
    assert [line.split()[1] for line in findings[:4]] == [
        "violation",
        "violation",
        "descriptive",
        "passed",
    ]
    assert (
        "- violation invalid_values: 2 columns hold invalid values: 'ordered' and "
        "'qty'."
    ) in lines
    assert "- column_not_in_data: 1 declared column isn't in the data, so" in " ".join(
        lines
    )
    assert max(len(line) for line in lines) <= 120


def test_to_frame_has_a_row_per_declared_column_and_a_table_of_invalid_values():
    res = ae.validate_column_types(
        orders(), schema={"qty": "integer", "ordered": "datetime"}
    )

    columns = res.to_frame()
    assert columns["column"].tolist() == ["qty", "ordered"]
    assert columns["invalid_count"].tolist() == [2, 1]
    assert str(columns["invalid_count"].dtype) == "Int64"
    assert columns["invalid_row_ids"].tolist() == [[2, 4], [3]]
    invalid = res.to_frame(table="invalid_values")
    assert invalid.columns.tolist() == ["column", "value", "value_type", "count"]
    assert invalid.to_numpy().tolist() == [
        ["qty", "n/a", "str", 2],
        ["ordered", "05/01/2024", "str", 1],
    ]


def test_with_masking_on_the_table_of_invalid_values_shows_placeholders():
    res = ae.validate_column_types(
        customers(), schema={"phone": "integer"}, project=masked("phone")
    )

    invalid = res.to_frame(table="invalid_values")
    assert invalid["value"].tolist() == ["<masked 1>", "<masked 2>"]
    assert invalid["count"].tolist() == [2, 1]
    assert "555" not in invalid.to_csv() + str(res)
    # Placeholders print without quotes, so they can't pass for stored text.
    assert "<masked 1>: 2, <masked 2>: 1" in str(res)


def test_support_counts_the_values_checked_in_each_declared_column_of_the_data():
    res = ae.validate_column_types(
        orders(), schema={"ordered": "datetime", "customer": "text"}
    )

    assert (res.support.row_count, res.support.column_count) == (5, 1)
    [support] = res.support.columns
    assert support.column == "ordered"
    assert (support.eligible_count, support.evaluated_count) == (5, 4)
    assert (support.excluded_count, support.excluded_by_reason) == (1, {"missing": 1})


def test_nan_stored_as_a_value_counts_as_missing_with_a_limitation():
    # double[pyarrow] keeps NaN, a value, apart from null, a missing value.
    stored = pa.array([1.5, NAN, None], type=pa.float64(), from_pandas=False)
    df = pd.DataFrame({"x": pd.Series(pd.arrays.ArrowExtensionArray(stored))})

    res = ae.validate_column_types(df, schema={"x": "float"})

    assert counts(res.columns["x"]) == (1, 0, 0, 2)
    [limitation] = res.limitations
    assert limitation.code == "nan_counted_as_missing"
    assert limitation.reason == (
        "1 column stores NaN as a value, which pandas isna() doesn't count as "
        "missing; missing_count counts it: 'x'."
    )
    assert limitation.fields == ["columns['x'].missing_count"]


# The share of infer_column_semantics' evidence that counts the values a
# declared logical type accepts, stored as it or convertible.
AGREEING_SHARES: dict[ae.LogicalType, str] = {
    "boolean": "boolean_share",
    "integer": "integer_share",
    "float": "number_share",
    "datetime": "datetime_share",
}


def assert_agrees_with_infer_column_semantics(df):
    inferred = ae.infer_column_semantics(df).columns
    for declared, share in AGREEING_SHARES.items():
        res = ae.validate_column_types(df, schema=dict.fromkeys(inferred, declared))
        for name, entry in res.columns.items():
            evidence = inferred[name].evidence
            compatible, convertible, invalid, _ = counts(entry)
            assert compatible is not None
            assert convertible is not None
            assert invalid is not None
            accepted = compatible + convertible
            checked = accepted + invalid
            assert checked == evidence.non_missing_count, (name, declared)
            if checked:
                assert accepted / checked == getattr(evidence, share), (name, declared)


@given(df=data_frames())
def test_accepts_the_values_infer_column_semantics_reads_as_each_type(df):
    assume(df.shape[1] > 0)

    assert_agrees_with_infer_column_semantics(df)


# Text near the edges of the parse rules.
TRICKY_TEXT = st.sampled_from(
    [
        *("007", "0", "1", "-0", "+4", ".5", "5.", "1e3", "4.2e1", "1E-2", "1_000"),
        *("1,000", "3,5", "0x10", "inf", "nan", " 42 ", "\t1\n", "١٢٣", "4.0e-1"),
        *("yes", "No", "TRUE", "t", "y", "n", "f", "2024-02-29", "2023-02-29"),
        *("2024-01-05T10:00:00+01:00", "2024-01-05 10:00", "2024-01-05Z", "9999-12-31"),
        *("2024-1-5", "05/01/2024", "5 Jan 2024", "2024-13-01", "0001-01-01"),
    ]
)


@given(values=st.lists(TRICKY_TEXT | st.text(max_size=6) | st.none(), max_size=12))
def test_parses_text_as_infer_column_semantics_does(values):
    df = pd.DataFrame({"x": pd.Series(values, dtype=object)})

    assert_agrees_with_infer_column_semantics(df)


def test_nothing_is_converted_and_the_data_never_changes():
    df = orders()
    df.attrs = {"source": "orders.csv"}
    before = df.copy(deep=True)

    ae.validate_column_types(
        df,
        schema={
            "order_id": "integer",
            "qty": "float",
            "ordered": ae.ColumnSpec("datetime", datetime_format="%d/%m/%Y"),
            "note": "categorical",
        },
    )

    pd.testing.assert_frame_equal(df, before, check_exact=True)
    assert df.attrs == {"source": "orders.csv"}
    assert df["qty"].tolist() == ["2", "1", "n/a", "3", "n/a"]


@pytest.mark.parametrize(
    "neighbour",
    ["infer_column_semantics", "describe_dataset", "profile_missing_values"],
)
def test_nearby_analyses_point_to_it_for_checking_declared_types(neighbour):
    doc = inspect.getdoc(getattr(ae, neighbour))
    assert doc

    assert "`validate_column_types`" in ae.describe_analysis(neighbour).when_to_use
    assert "    validate_column_types: " in doc.split("See Also:")[1]


def test_invalid_datetimes_that_tie_follow_their_string_forms():
    # Around the end of daylight saving time, local time runs back an hour:
    # 02:15+01:00 comes after 02:30+02:00 in time, but before it as text.
    instants = pd.Series(
        pd.to_datetime(
            ["2020-10-25 00:30:00", "2020-10-25 01:15:00", "2020-10-24 23:00:00"]
        )
    ).dt.tz_localize("UTC")
    df = pd.DataFrame(
        {
            "aware": instants.dt.tz_convert("Europe/Berlin"),
            "naive": instants.dt.tz_localize(None),
        }
    )

    res = ae.validate_column_types(df, schema={"aware": "integer", "naive": "integer"})

    assert [v.value for v in res.columns["aware"].invalid_values] == [
        "2020-10-25 01:00:00+02:00",
        "2020-10-25 02:15:00+01:00",
        "2020-10-25 02:30:00+02:00",
    ]
    assert [v.value for v in res.columns["naive"].invalid_values] == [
        "2020-10-24 23:00:00",
        "2020-10-25 00:30:00",
        "2020-10-25 01:15:00",
    ]
    invalid = res.to_frame(table="invalid_values")
    aware = invalid[invalid["column"] == "aware"]
    assert [str(value) for value in aware["value"]] == [
        v.value for v in res.columns["aware"].invalid_values
    ]
    assert aware["value_type"].tolist() == ["Timestamp"] * 3


def test_the_counts_of_a_column_missing_from_the_data_are_null_with_the_reason():
    df = pd.DataFrame({"age": [34, 51]})

    res = ae.validate_column_types(df, schema={"zip": "categorical"})

    [limitation] = res.limitations
    assert limitation.code == "column_not_in_data"
    assert limitation.fields == ["columns['zip']"]
    assert limitation.reason == (
        "1 declared column isn't in the data, so its dtype and counts are null: 'zip'."
    )
