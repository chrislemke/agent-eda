"""Statuses, finding codes and caps each have one source.

The docstring declares an analysis's statuses and finding codes, and registration
reads them. A cap's number lives in its marker on the result class, and the
analyses, the field descriptions and the check of the record all take it from there.
"""

import re
from pathlib import Path

import pytest
from registering import decorator_options, register_stand_in

import agent_eda as ae

DOCSTRING = """What does the stand-in see?

Returns:
    CountedResult: `seen`.

    - `rows_seen`: descriptive; the stand-in saw rows.
    - `declared`: an evidence code, which is no finding code.

Statuses:
    completed: Always.
    insufficient_data: No rows. Reason codes: `no_rows`.
"""


def test_the_decorator_takes_neither_statuses_nor_finding_codes():
    options = decorator_options()

    assert "statuses" not in options
    assert "finding_codes" not in options


def test_registration_reads_statuses_and_finding_codes_from_the_docstring():
    spec = register_stand_in(DOCSTRING)

    assert spec.statuses == ("completed", "insufficient_data")
    assert spec.finding_codes == ("rows_seen",)


def test_registration_raises_for_a_status_outside_the_status_list():
    broken = DOCSTRING.replace("insufficient_data:", "half_done:")

    with pytest.raises(TypeError, match="half_done") as raised:
        register_stand_in(broken)

    assert "completed" in str(raised.value), "the message lists the statuses allowed"


@pytest.mark.parametrize("code", ["Rows-Seen", "rows seen", "1rows", "RowsSeen"])
def test_registration_raises_for_a_malformed_finding_code(code):
    broken = DOCSTRING.replace("rows_seen", code)

    with pytest.raises(TypeError, match="malformed finding code"):
        register_stand_in(broken)


def test_every_analysis_registers_the_statuses_and_codes_its_docstring_declares():
    for info in ae.list_analyses().analyses:
        description = ae.describe_analysis(info.name)

        assert [s.status for s in description.statuses] == info.statuses
        assert [c.code for c in description.finding_codes] == info.finding_codes
        for code in description.finding_codes:
            assert code.description, f"{info.name}: `{code.code}` has no bullet"


def result_classes(base=ae.results.ResultModel):
    """Every record class of the package: the subclasses of ResultModel, found
    after the analyses are loaded."""
    ae.list_analyses()
    found = []
    for subclass in base.__subclasses__():
        found += [subclass, *result_classes(subclass)]
    return found


def capped_fields():
    """Each capped field as (class, field name, limit), asking the class."""
    found = []
    for cls in {c for c in result_classes() if c.__module__.startswith("agent_eda")}:
        for field in cls.model_fields:
            try:
                limit = cls.limit_of(field)
            except ValueError:
                continue
            found.append((cls, field, limit))
    return found


def test_a_result_class_gives_the_limit_of_each_capped_field():
    assert ae.results.DuplicateGroup.limit_of("row_ids") == 10
    assert ae.results.ValidateColumnTypesColumn.limit_of("invalid_values") == 5
    assert ae.results.ValidateColumnTypesColumn.limit_of("invalid_row_ids") == 10
    assert ae.results.AnalysisResult.limit_of("findings") == 100
    assert ae.results.AnalysisResult.limit_of("parameters") == 100


def test_asking_for_the_limit_of_an_uncapped_field_names_the_capped_ones():
    with pytest.raises(ValueError, match="parameters") as raised:
        ae.results.AnalysisResult.limit_of("status")

    assert "findings" in str(raised.value)
    with pytest.raises(ValueError, match="no_such_field"):
        ae.results.AnalysisResult.limit_of("no_such_field")


def test_the_cap_numbers_are_the_documented_ones():
    limits = {limit for _, _, limit in capped_fields()}

    assert limits == {10, 5, 100}


def test_a_capped_field_that_names_its_cap_says_the_markers_number():
    fields = capped_fields()
    assert fields

    named = 0
    for cls, field, limit in fields:
        description = cls.model_fields[field].description or ""
        numbers = re.findall(r"\b(?:first|keeps the) (\d+)\b", description)
        named += bool(numbers)
        assert set(numbers) <= {str(limit)}, f"{cls.__name__}.{field}: {description}"
        schema = cls.model_json_schema()["properties"][field]
        assert schema["description"] == description
    assert named >= 20, "the cap clauses are generated, so they exist"


def test_the_cap_clause_of_a_description_is_generated_from_its_marker():
    description = ae.results.DuplicateGroup.model_fields["row_ids"].description or ""

    assert "The record keeps the first 10 and adds row_ids_total." in description


def test_no_description_or_docstring_keeps_a_placeholder():
    for cls in result_classes():
        for field, info in cls.model_fields.items():
            assert "{" not in (info.description or ""), f"{cls.__name__}.{field}"
    for info in ae.list_analyses().analyses:
        assert "_kept}" not in str(getattr(ae, info.name).__doc__), info.name


def test_the_cap_numbers_are_written_only_in_the_markers():
    package = Path(ae.__file__).parent
    written = re.compile(
        r"\b(?:first|keeps the|keeps its first|strongest)\s+(?:10|5|100)\b"
        r"|\b(?:10|5|100) (?:row IDs|example values|entries of anything else)"
        r"|\b_Cap\(\d"
    )
    found = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "_result_base.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if written.search(line):
                found.append(
                    f"{path.relative_to(package)}:{number + 1}: {line.strip()}"
                )

    assert found == [], (
        "A cap's number lives in its marker in _result_base. Write {limit}, {cap} "
        "or {row_ids_kept}, {examples_kept}, {entries_kept} instead, or ask the "
        "result class with limit_of."
    )
