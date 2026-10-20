"""One name per concept: a name retired by a rename doesn't come back."""

import inspect
import re
import typing
from pathlib import Path

import pandas as pd
import pydantic
import pytest

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]
INFOS = ae.list_analyses().analyses

# Retired name -> the name that replaced it.
RENAMED_ARGUMENTS = {"declared_schema": "schema"}
RENAMED_FINDING_CODES = {
    "declared_columns_missing": "missing_declared_columns",
    "valid_columns": "passing_columns",
}
# Retired field name -> the result class that had it and the replacement.
RENAMED_FIELDS = {
    "cardinality": ("SummarizeCategoricalColumn", "distinct_count"),
    "count": ("SummarizeNumericColumn", "non_missing_count"),
    "skipped_columns": ("ComputeAssociationMatrixResult", "left_out_columns"),
    "duplicate_row_count": ("CandidateKey", "duplicate_group_row_count"),
    "distinct_count": ("CandidateKey", "distinct_row_count"),
    "level_count": ("AssociationColumn", "distinct_count"),
}
# Words that no longer appear anywhere in the README.
RETIRED_WORDS = [
    "declared_schema",
    "cardinality",
    "skipped_columns",
    "duplicate_row_count",
    "declared_columns_missing",
    "valid_columns",
]


def models_in(annotation):
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return [annotation]
    return [m for arg in typing.get_args(annotation) for m in models_in(arg)]


def result_classes(name):
    """The analysis's result class and every model reachable through its fields."""
    found = {}
    pending = [inspect.signature(getattr(ae, name)).return_annotation]
    while pending:
        model = pending.pop()
        if model.__name__ in found:
            continue
        found[model.__name__] = model
        for field in model.model_fields.values():
            pending.extend(models_in(field.annotation))
    return found


ALL_CLASSES = {
    class_name: model
    for info in INFOS
    for class_name, model in result_classes(info.name).items()
}


@pytest.mark.parametrize("old", RENAMED_ARGUMENTS)
def test_a_retired_argument_name_is_in_no_signature(old):
    for info in INFOS:
        parameters = inspect.signature(getattr(ae, info.name)).parameters
        assert old not in parameters, (info.name, RENAMED_ARGUMENTS[old])
        assert old not in info.parameters, info.name


def test_both_analyses_that_take_declared_types_call_the_argument_schema():
    for name in ("infer_column_semantics", "validate_column_types"):
        assert "schema" in inspect.signature(getattr(ae, name)).parameters


@pytest.mark.parametrize("old", RENAMED_FINDING_CODES)
def test_a_retired_finding_code_is_declared_by_no_analysis(old):
    for info in INFOS:
        assert old not in info.finding_codes, (info.name, RENAMED_FINDING_CODES[old])
        assert old not in (inspect.getdoc(getattr(ae, info.name)) or ""), info.name


@pytest.mark.parametrize("old", RENAMED_FIELDS)
def test_a_retired_field_name_is_gone_from_the_class_that_had_it(old):
    class_name, new = RENAMED_FIELDS[old]

    fields = ALL_CLASSES[class_name].model_fields

    assert old not in fields
    assert new in fields


@pytest.mark.parametrize(
    "old", ["cardinality", "skipped_columns", "duplicate_row_count", "level_count"]
)
def test_a_retired_field_name_is_in_no_result_class_but_the_one_that_keeps_it(old):
    holding = [name for name, model in ALL_CLASSES.items() if old in model.model_fields]

    # level_count still names the levels of an index, which are not values.
    assert holding == (["DescribeDatasetIndex"] if old == "level_count" else [])


def test_the_numeric_summary_has_no_column_called_count():
    res = ae.summarize_numeric(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    assert "count" not in res.to_frame().columns
    assert "non_missing_count" in res.to_frame().columns
    assert "count" not in res.to_dict()["columns"]["x"]


def test_a_retired_word_is_not_in_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    found = [word for word in RETIRED_WORDS if re.search(rf"\b{word}\b", readme)]

    assert found == []


def test_the_changelog_and_the_glossary_list_each_rename():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    glossary = (ROOT / "GLOSSARY.md").read_text(encoding="utf-8")
    renames = [
        *RENAMED_ARGUMENTS.items(),
        *RENAMED_FINDING_CODES.items(),
        *((old, new) for old, (_, new) in RENAMED_FIELDS.items()),
    ]

    for old, new in renames:
        assert f"`{old}`" in changelog, old
        assert f"`{new}`" in changelog, new
    for old, new in renames:
        if old in ("count", "level_count", "distinct_count"):
            continue  # ordinary words, still right in other result classes
        assert f"`{old}`" in glossary, old
        assert f"`{new}`" in glossary, new
