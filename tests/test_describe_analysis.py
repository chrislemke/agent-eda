import doctest
import inspect
import json
import textwrap

import pytest
from hypothesis import given
from hypothesis import strategies as st

import agent_eda as ae
from agent_eda.errors import AgentEDAError, InvalidArgumentError, UnknownAnalysisError

ANALYSES = [info.name for info in ae.list_analyses().analyses]


def test_describes_an_analysis_named_by_its_short_name():
    description = ae.describe_analysis("summarize_numeric")

    assert description.name == "summarize_numeric"
    assert description.module == "agent_eda.statistics.descriptive"
    assert description.summary == (
        "How are the values in each numeric column distributed?"
    )


def test_describes_an_analysis_named_by_its_fully_qualified_name():
    description = ae.describe_analysis("agent_eda.data.structure.describe_dataset")

    assert description.name == "describe_dataset"
    assert description.module == "agent_eda.data.structure"


@pytest.mark.parametrize(
    "name", ["agent_eda.summarize_numeric", "ae.summarize_numeric"]
)
def test_describes_an_analysis_named_as_the_package_exports_it(name):
    assert ae.describe_analysis(name).name == "summarize_numeric"


def test_an_analysis_passed_instead_of_its_name_raises_with_the_call_to_make():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_analysis(ae.summarize_numeric)  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "describe_analysis takes the name of an analysis as a string, but got the "
        "function summarize_numeric. Pass its name instead: "
        "ae.describe_analysis('summarize_numeric')."
    )


def test_a_name_of_another_type_raises_with_an_example():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_analysis(3)  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "describe_analysis takes the name of an analysis as a string, such as "
        "ae.describe_analysis('describe_dataset'), but got a value of type int. "
        "ae.list_analyses() lists every name."
    )


@pytest.mark.parametrize(
    ("name", "closest"),
    [
        ("summarise_numeric", "'summarize_numeric'"),
        ("Describe_Dataset", "'describe_dataset'"),
        ("agent_eda.data.summarize_numerics", "'summarize_numeric'"),
    ],
    ids=["misspelled", "wrong-case", "qualified-misspelled"],
)
def test_an_unknown_name_raises_with_the_closest_analysis_names(name, closest):
    with pytest.raises(UnknownAnalysisError) as caught:
        ae.describe_analysis(name)

    error = caught.value
    message = str(error)
    assert isinstance(error, LookupError)
    assert isinstance(error, AgentEDAError)
    assert error.code == "unknown_analysis"
    # Later analyses may add more suggestions after the closest one.
    assert message.startswith(
        f"describe_analysis got an unknown analysis {name!r}. Did you mean {closest}"
    )
    assert all(repr(analysis) in message for analysis in ANALYSES)
    assert "ae.search_analyses(" in message


@given(name=st.text() | st.sampled_from(ANALYSES))
def test_any_name_gives_a_description_or_an_unknown_analysis_error(name):
    described = description_or_error(name)

    if isinstance(described, UnknownAnalysisError):
        assert described.code == "unknown_analysis"
        assert str(described).startswith(
            f"describe_analysis got an unknown analysis {name!r}."
        )
    else:
        assert described.name in ANALYSES


def description_or_error(name):
    try:
        return ae.describe_analysis(name)
    except UnknownAnalysisError as error:
        return error


def test_a_name_without_a_close_match_lists_every_analysis():
    with pytest.raises(UnknownAnalysisError) as caught:
        ae.describe_analysis("histogram")

    message = str(caught.value)
    assert message.startswith(
        f"describe_analysis got an unknown analysis 'histogram'. agent-eda has "
        f"{len(ANALYSES)} analyses: "
    )
    assert "Did you mean" not in message
    assert all(repr(analysis) in message for analysis in ANALYSES)


def test_shows_the_signature_with_readable_types_and_defaults():
    numeric = ae.describe_analysis("summarize_numeric")
    dataset = ae.describe_analysis("describe_dataset")

    assert numeric.signature == (
        "summarize_numeric(data: pd.DataFrame, *, columns: Sequence[str] | None = "
        "None, quantiles: Sequence[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, "
        "0.99], quantile_method: str = 'linear', trim_fraction: float = 0.1) -> "
        "SummarizeNumericResult"
    )
    assert dataset.signature == (
        "describe_dataset(data: pd.DataFrame, *, source: str | None = None, "
        "source_version: str | None = None) -> DescribeDatasetResult"
    )


def test_describes_each_parameter_with_its_type_default_and_docstring_text():
    description = ae.describe_analysis("describe_dataset")

    data, source, source_version = description.parameters
    assert (data.name, data.type, data.required, data.default) == (
        "data",
        "pd.DataFrame",
        True,
        None,
    )
    assert data.description == (
        "The data to describe. Any column labels are accepted, including "
        "duplicated ones and MultiIndex columns. It is never modified."
    )
    assert (source.name, source.type, source.required, source.default) == (
        "source",
        "str | None",
        False,
        None,
    )
    assert source.description == (
        "Where the data came from, such as a file path, URL or table name, recorded "
        "exactly as given. None, the default, records the source as not supplied; "
        "it is never inferred."
    )
    assert source_version.name == "source_version"


def test_lists_the_allowed_quantile_methods_and_the_trim_fraction_range():
    description = ae.describe_analysis("summarize_numeric")

    parameters = {p.name: p for p in description.parameters}
    assert parameters["quantile_method"].allowed_values == [
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
    assert parameters["trim_fraction"].valid_range == "0 <= trim_fraction < 0.5"
    assert parameters["quantiles"].valid_range == "each value: 0 <= value <= 1"
    assert parameters["trim_fraction"].allowed_values is None
    assert parameters["quantile_method"].valid_range is None
    assert parameters["columns"].valid_range is None


def test_lists_the_declared_statuses_with_their_reasons():
    description = ae.describe_analysis("describe_dataset")

    assert [(s.status, s.reason) for s in description.statuses] == [
        (
            "completed",
            "The description finished, also for data with no rows, no columns or "
            "neither.",
        ),
        (
            "failed",
            "An anticipated computation failure stopped the description, and "
            "`status_reason` gives its code and message. No such failure is known today: a "
            "count that can't be computed is null, with a limitation, instead.",
        ),
    ]


def test_lists_the_finding_codes_with_their_meanings():
    description = ae.describe_analysis("summarize_numeric")

    assert [(f.code, f.description) for f in description.finding_codes] == [
        (
            "constant_columns",
            "suspected; a column holds a single distinct non-missing value in at "
            "least two rows, on stored values; NaN is missing.",
        ),
        (
            "infinite_values",
            "suspected; a column holds inf or -inf, which every statistic leaves out.",
        ),
    ]


def test_lists_the_result_fields_with_their_meanings():
    description = ae.describe_analysis("summarize_numeric")

    assert description.result_class == "SummarizeNumericResult"
    fields = {f.name: f for f in description.result_fields}
    assert list(fields) == ["columns", "left_out_columns", "definitions"]
    assert fields["left_out_columns"].type == "list[str]"
    assert fields["left_out_columns"].description.startswith(
        "Columns that the default column selection left out because they aren't "
        "numeric or are boolean, in DataFrame order;"
    )
    assert fields["columns"].type == "dict[str, SummarizeNumericColumn]"
    entry = {f.name: f for f in fields["columns"].fields}
    assert entry["median"].type == "float | None"
    assert entry["median"].description.startswith("Middle finite value")
    assert [f.name for f in entry["status_reason"].fields] == ["code", "message"]


def test_lists_the_fields_every_result_shares_apart():
    description = ae.describe_analysis("describe_dataset")

    shared = {f.name: f for f in description.shared_result_fields}
    assert list(shared) == [
        "analysis",
        "method",
        "status",
        "status_reason",
        "parameters",
        "support",
        "findings",
        "limitations",
        "execution",
    ]
    assert shared["findings"].type == "list[Finding]"
    assert [f.name for f in shared["findings"].fields] == [
        "code",
        "claim",
        "assessment",
        "columns",
        "fields",
    ]


def test_states_masking_support_work_limits_and_methods():
    description = ae.describe_analysis("summarize_numeric")

    assert description.supports_masking is False
    assert description.work_limit_parameters == []
    assert description.methods == []


@pytest.mark.parametrize("name", ANALYSES)
def test_agrees_with_what_list_analyses_records(name):
    [info] = [i for i in ae.list_analyses().analyses if i.name == name]

    description = ae.describe_analysis(name)

    assert description.supports_masking == info.supports_masking
    assert description.work_limit_parameters == info.work_limit_parameters
    assert description.methods == info.methods
    assert [p.name for p in description.parameters] == info.parameters
    assert [s.status for s in description.statuses] == info.statuses
    assert [f.code for f in description.finding_codes] == info.finding_codes
    assert description.result_class == info.result_class


@pytest.mark.parametrize("name", ANALYSES)
def test_the_example_is_the_docstring_doctest(name):
    # tests/test_contract.py runs each docstring's doctest, once.
    doc = inspect.getdoc(getattr(ae, name))
    assert doc
    section = doc.split("\nExamples:\n")[1].split("\nSee Also:\n")[0]

    example = ae.describe_analysis(name).example

    assert example == textwrap.dedent(section).strip("\n")
    assert doctest.DocTestParser().get_examples(example)


def test_the_record_has_a_json_schema_of_the_parameters_other_than_data():
    description = ae.describe_analysis("summarize_numeric")

    record = description.to_dict()

    schema = record["parameters_schema"]
    properties = schema["properties"]
    assert list(properties) == [
        "columns",
        "quantiles",
        "quantile_method",
        "trim_fraction",
    ]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert properties["trim_fraction"]["type"] == "number"
    assert properties["trim_fraction"]["minimum"] == 0
    assert properties["trim_fraction"]["exclusiveMaximum"] == 0.5
    assert properties["trim_fraction"]["default"] == 0.1
    assert properties["quantile_method"]["enum"][:2] == [
        "inverted_cdf",
        "averaged_inverted_cdf",
    ]
    assert properties["quantiles"]["items"] == {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }
    assert properties["trim_fraction"]["description"].startswith(
        "The share of the finite values cut from each end for the trimmed mean"
    )


def test_to_json_is_strict_json_of_the_record():
    description = ae.describe_analysis("describe_dataset")

    text = description.to_json()

    assert json.loads(text) == description.to_dict()
    assert description.to_dict()["name"] == "describe_dataset"
    assert ae.describe_analysis("describe_dataset").to_json() == text


def test_prints_each_part_compactly_in_plain_text():
    description = ae.describe_analysis("summarize_numeric")

    lines = str(description).splitlines()

    assert lines[0] == (
        "summarize_numeric: How are the values in each numeric column distributed?"
    )
    assert lines[1].startswith(
        "Signature: summarize_numeric(data: pd.DataFrame, *, columns: "
    )
    assert "Module: agent_eda.statistics.descriptive. Work limits: none." in lines
    assert any(line.startswith("Masking: not needed.") for line in lines)
    assert "When to use:" in lines
    assert "Parameters:" in lines
    assert "  data: pd.DataFrame (required)" in lines
    assert "  trim_fraction: float = 0.1; 0 <= trim_fraction < 0.5" in lines
    assert (
        "  quantiles: Sequence[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]; "
        "each value: 0 <= value <= 1"
    ) in lines
    method = lines.index("  quantile_method: str = 'linear'")
    assert lines[method + 1].startswith(
        "    Allowed: 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation',"
    )
    assert "  completed: Every selected column has a finite value." in lines
    assert (
        "  infinite_values: suspected; a column holds inf or -inf, which every "
        "statistic leaves out."
    ) in lines
    assert "  left_out_columns: list[str]" in lines
    example = lines.index("Example:")
    assert lines[example + 1 : example + 3] == [
        "  import pandas as pd",
        "  import agent_eda as ae",
    ]
    assert "  res = ae.summarize_numeric(df)" in lines
    # A one-line result of an expression follows it as a comment.
    assert '  res.columns["income"].median  # 58.0' in lines
    assert not [line for line in lines if line.lstrip().startswith(">>>")]
    assert lines[-1] == (
        "More: help(ae.summarize_numeric) has the full docstring; to_json() adds the "
        "parameters' JSON Schema."
    )


@pytest.mark.parametrize("name", ANALYSES)
def test_printed_lines_stay_within_120_characters(name):
    description = ae.describe_analysis(name)

    lines = str(description).splitlines()

    assert max(len(line) for line in lines) <= 120
    assert lines == [line.rstrip() for line in lines]
    assert repr(description) == str(description)


def test_a_default_is_shown_as_its_json_value():
    description = ae.describe_analysis("summarize_numeric")

    defaults = {p.name: p.default for p in description.parameters}
    assert defaults == {
        "data": None,
        "columns": None,
        "quantiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
        "quantile_method": "linear",
        "trim_fraction": 0.1,
    }
