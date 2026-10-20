"""GLOSSARY.md defines the terms the analyses rely on, as the code uses them."""

import re
from pathlib import Path

import pandas as pd

import agent_eda as ae

GLOSSARY = Path(__file__).resolve().parents[1] / "GLOSSARY.md"

# Each entry's bold heading, and the terms it must name.
REQUIRED = {
    "Limitation": ["Limitation"],
    "First look": ["First look"],
    "Level": ["Level"],
    "Native missing value": ["Native missing value"],
    "Compatible, convertible and invalid value": [
        "Compatible value",
        "convertible value",
        "invalid value",
    ],
    "Key, primary key and candidate key": [
        "Key",
        "primary key",
        "candidate key",
    ],
    "Target, class and label": ["Target", "class", "label"],
    "Sentinel candidate": ["Sentinel candidate"],
}


def entries():
    """The glossary's entries: heading line -> the lines that follow it."""
    found = {}
    lines = GLOSSARY.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        if re.match(r"^\*\*.+\*\*.*:$", line):
            body = []
            for text in lines[number + 1 :]:
                if not text.strip():
                    break
                body.append(text)
            found[line] = body
    return found


def entry_for(term):
    matches = [
        (heading, body)
        for heading, body in entries().items()
        if term.lower() in heading.lower()
    ]
    assert matches, f"GLOSSARY.md has no entry for {term!r}."
    return matches[0]


def test_the_glossary_defines_each_term_with_an_avoid_line():
    for terms in REQUIRED.values():
        for term in terms:
            heading, body = entry_for(term)
            assert len(body) >= 2, heading
            assert body[-1].startswith("_Avoid_: "), heading
            assert not body[0].startswith("_Avoid_"), heading


def test_a_level_is_a_distinct_non_missing_value_as_value_counts_tells_them_apart():
    values = pd.Series([1, 1.0, True, "Yes", "yes ", None, float("nan")], dtype=object)

    res = ae.summarize_categorical(pd.DataFrame({"x": values}))

    column = res.columns["x"]
    assert column.distinct_count == values.value_counts().size == 3
    assert column.missing_count == 2
    definition = " ".join(entry_for("Level")[1])
    assert "value_counts" in definition
    assert "missing" in definition


def test_a_native_missing_value_is_what_pandas_counts_as_missing_not_a_sentinel():
    data = pd.DataFrame({"x": [1.0, None, float("nan"), -999.0, 4.0]})

    res = ae.profile_missing_values(data, sentinel_values={"x": [-999]})

    column = res.columns["x"]
    assert column.native_missing_count == int(data["x"].isna().sum()) == 2
    assert column.sentinel_count == 1
    definition = " ".join(entry_for("Native missing value")[1])
    for word in ("isna", "None", "NaN", "NaT", "pd.NA"):
        assert word in definition


def test_compatible_convertible_and_invalid_values_are_the_three_groups_of_a_check():
    data = pd.DataFrame(
        {"x": pd.Series([1, 2.0, 2.5, "3", "three", None], dtype=object)}
    )

    res = ae.validate_column_types(data, schema={"x": "integer"})

    column = res.columns["x"]
    assert (column.compatible_count, column.convertible_count) == (1, 2)
    assert column.invalid_count == 2
    definition = " ".join(entry_for("Compatible value")[1]).lower()
    for word in ("stored", "exactly", "neither", "missing"):
        assert word in definition


def test_a_key_is_unique_a_primary_key_has_no_missing_value_a_candidate_key_is_minimal():
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, None, 4],
            "c": ["x", "x", "y", "y"],
            "d": [1, 2, 1, 2],
        }
    )

    res = ae.find_candidate_keys(data, max_key_size=2)

    assert [key.columns for key in res.keys] == [["a"], ["c", "d"]]
    assert [key.columns for key in res.keys_with_missing_values] == [["b"]]
    assert all(key.is_minimal for key in res.keys)
    text = " ".join(" ".join(entry_for("Key")[1]).split())
    for word in ("no two rows share", "missing", "minimal"):
        assert word in text


def test_a_target_has_classes_that_are_its_labels_other_than_missing():
    data = pd.DataFrame({"y": ["no", "yes", "yes", None]})

    res = ae.profile_target(
        data,
        target_column="y",
        task="binary_classification",
        expected_labels=["no", "yes"],
    )

    assert [(c.label, c.count) for c in res.classes or []] == [("yes", 2), ("no", 1)]
    assert res.missing_count == 1
    text = " ".join(" ".join(entry_for("Target")[1]).split())
    for word in ("predict", "class", "label"):
        assert word in text


def test_a_sentinel_candidate_is_declared_or_suspected_and_stays_a_value_until_declared():
    data = pd.DataFrame({"x": [1.0, 2.0, -999.0, 3.0, 99.0, 4.0]})

    res = ae.detect_missing_sentinels(data, sentinel_values={"x": [99]})

    confirmations = {c.value: c.confirmation for c in res.candidates}
    assert confirmations == {-999.0: "suspected", 99.0: "declared"}
    definition = " ".join(entry_for("Sentinel candidate")[1])
    for word in ("declared", "suspected"):
        assert word in definition


def test_a_limitation_names_a_code_a_reason_and_the_fields_it_affects():
    res = ae.profile_target(
        pd.DataFrame({"y": ["a", "b"]}), target_column="y", task="binary_classification"
    )

    assert res.limitations
    limitation = res.limitations[0]
    for field in (limitation.code, limitation.reason, limitation.fields):
        assert field
    definition = " ".join(entry_for("Limitation")[1])
    for word in ("code", "reason", "fields"):
        assert word in definition


def test_the_first_look_is_the_order_ae_guide_gives(capsys):
    ae.guide()
    guide = capsys.readouterr().out
    definition = " ".join(entry_for("First look")[1])

    assert "A first look at a new dataset, in this order:" in guide
    assert "guide" in definition
