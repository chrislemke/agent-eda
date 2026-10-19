"""The result API every analysis shares: tables by name, helpful errors, help text,
Rich display and the bridges that carry one result into the next analysis."""

import datetime as dt
import inspect
import io
import pydoc
import subprocess
import sys
import typing

import pandas as pd
import pytest
from recipes import recipe_for
from strategies import EDGE_FRAMES

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError

ANALYSES = [info.name for info in ae.list_analyses().analyses]


def call(name, df=None):
    df = EDGE_FRAMES["every kind"]() if df is None else df
    arguments = recipe_for(name).required(df)
    if arguments is None:
        pytest.skip(f"{name} takes no call on this frame")
    return getattr(ae, name)(df, **arguments)


@pytest.mark.parametrize("name", ANALYSES)
def test_every_result_accepts_its_tables_by_name(name):
    res = call(name)

    hints = typing.get_type_hints(res.to_frame)
    assert "table" in hints, f"{name}'s to_frame takes no table argument"
    names = typing.get_args(hints["table"])
    assert names
    pd.testing.assert_frame_equal(res.to_frame(table=names[0]), res.to_frame())
    for table in names:
        assert isinstance(res.to_frame(table=table), pd.DataFrame)


@pytest.mark.parametrize("name", ANALYSES)
def test_a_wrong_table_name_gives_the_error_with_the_choices(name):
    res = call(name)
    names = typing.get_args(typing.get_type_hints(res.to_frame)["table"])

    with pytest.raises(InvalidArgumentError) as error:
        res.to_frame(table="tabel")  # the bad name is the test

    message = str(error.value)
    assert "to_frame got 'tabel' for table, but table must be " in message
    assert repr(names[0]) in message
    assert f"res.to_frame() returns the {names[0]} table." in message


MAPPING_ANALYSES = [
    "summarize_numeric",
    "summarize_categorical",
    "profile_missing_values",
    "compute_association_matrix",
    "infer_column_semantics",
    "validate_column_types",
]


def frame_with_age():
    return pd.DataFrame(
        {
            "age": [31, 42, 25, 38, 29],
            "income": [1.5, 2.5, 3.5, 4.5, 5.5],
            "city": ["a", "b", "c", "d", "e"],
        }
    )


def mapping_result(name):
    df = frame_with_age()
    arguments = {
        "validate_column_types": {
            "schema": {
                "age": ae.ColumnSpec("integer"),
                "income": ae.ColumnSpec("float"),
            }
        },
    }.get(name, {})
    return getattr(ae, name)(df, **arguments)


@pytest.mark.parametrize("name", MAPPING_ANALYSES)
def test_a_mistyped_column_key_names_the_closest_columns(name):
    columns = mapping_result(name).columns
    key = next(iter(columns))
    mistyped = key.upper()

    assert key in columns
    with pytest.raises(KeyError) as error:
        columns[mistyped]

    message = error.value.args[0]
    assert f"no column {mistyped!r}" in message
    assert f"Did you mean {key!r}" in message
    assert str(error.value) == message
    assert columns.get(mistyped) is None
    assert mistyped not in columns


def test_describe_dataset_columns_are_found_by_name_with_the_same_message():
    res = ae.describe_dataset(frame_with_age())

    assert res.columns["age"].column == "age"  # ty: ignore[invalid-argument-type]  # the list takes a name too
    assert res.columns[0].column == "age"
    assert [c.column for c in res.columns[:2]] == ["age", "income"]
    with pytest.raises(KeyError) as error:
        res.columns["Age"]  # ty: ignore[invalid-argument-type]  # the list takes a name too
    assert "no column 'Age'" in error.value.args[0]
    assert "Did you mean 'age'?" in error.value.args[0]


def test_describe_dataset_columns_by_name_say_when_a_label_repeats():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])

    res = ae.describe_dataset(df)

    assert res.columns["b"].position == 1  # ty: ignore[invalid-argument-type]  # the list takes a name too
    with pytest.raises(KeyError) as error:
        res.columns["a"]  # ty: ignore[invalid-argument-type]  # the list takes a name too
    assert "'a' labels 2 columns" in error.value.args[0]
    assert "positions 0 and 2" in error.value.args[0]


def test_describe_dataset_documents_why_columns_is_a_list():
    description = ae.describe_analysis("describe_dataset")
    field = next(f for f in description.result_fields if f.name == "columns")

    assert "list, not a mapping" in field.description
    assert "labels can repeat" in field.description


def test_results_with_column_keys_still_compare_and_serialize_as_before():
    res = ae.summarize_numeric(frame_with_age())

    assert res == ae.summarize_numeric(frame_with_age())
    assert list(res.to_dict()["columns"]) == ["age", "income"]
    assert res.model_copy().columns["age"] == res.columns["age"]


@pytest.mark.parametrize(
    ("method", "arguments", "instead"),
    [
        ("json", (), "to_json()"),
        ("dict", (), "to_dict()"),
        ("schema", (), "record_schema()"),
        ("parse_obj", ({},), "model_validate"),
        ("copy", (), "model_copy()"),
    ],
)
@pytest.mark.parametrize("name", ["describe_dataset", "summarize_numeric"])
def test_pydantics_deprecated_methods_name_the_method_to_use(
    name, method, arguments, instead
):
    res = call(name)

    with pytest.raises(InvalidArgumentError) as error:
        getattr(res, method)(*arguments)

    assert f"{method}()" in str(error.value)
    assert instead in str(error.value)


def test_the_class_methods_schema_and_parse_obj_also_name_the_method_to_use():
    with pytest.raises(InvalidArgumentError, match="record_schema"):
        ae.results.SummarizeNumericResult.schema()
    with pytest.raises(InvalidArgumentError, match="model_validate"):
        ae.results.SummarizeNumericResult.parse_obj({})


@pytest.mark.parametrize("name", ANALYSES)
def test_help_of_a_result_shows_each_field_and_points_to_describe_analysis(name):
    res = call(name)

    text = pydoc.plain(pydoc.render_doc(type(res)))

    assert len(text) < 10_000, f"help({type(res).__name__}) is {len(text)} characters"
    for field, info in type(res).model_fields.items():
        assert field in text
        # A description is wrapped in the text, so compare its words.
        assert " ".join(info.description.split()[:6]) in " ".join(text.split())
    assert f"ae.describe_analysis('{name}')" in text
    for method in ("to_frame", "to_dict", "to_json", "record_schema"):
        assert method in text
    assert "Methods inherited from pydantic" not in text


def test_help_of_a_result_names_what_a_column_lookup_gives():
    text = pydoc.plain(pydoc.render_doc(ae.results.DescribeDatasetResult))

    assert "res.columns['age']" in text


@pytest.mark.parametrize("name", ANALYSES)
def test_rich_renders_the_capped_text_of_the_result(name):
    console_module = pytest.importorskip("rich.console")
    res = call(name)
    printed = io.StringIO()
    console = console_module.Console(
        file=printed, width=120, force_terminal=False, color_system=None
    )

    console.print(res)

    assert printed.getvalue() == str(res) + "\n"
    assert len(printed.getvalue().splitlines()) <= 40


def test_rich_shows_text_with_square_brackets_as_it_is():
    console_module = pytest.importorskip("rich.console")
    df = pd.DataFrame({"[red]x[/red]": [1.0, 2.0, 3.0]})
    res = ae.summarize_numeric(df)
    printed = io.StringIO()
    console = console_module.Console(
        file=printed, width=120, force_terminal=False, color_system=None
    )

    console.print(res)

    assert "[red]x[/red]" in printed.getvalue()


def test_inferred_schema_is_accepted_by_validate_column_types():
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "paid": ["yes", "no", "yes"],
            "ordered": ["01/02/2024", "03/02/2024", "04/02/2024"],
            "empty": [None, None, None],
        }
    )

    res = ae.infer_column_semantics(
        df, schema={"ordered": ae.ColumnSpec("datetime", datetime_format="%d/%m/%Y")}
    )
    schema = res.inferred_schema()

    assert "empty" not in schema
    assert res.columns["empty"].inferred_logical_type is None
    assert {name: spec for name, spec in schema.items() if name != "ordered"} == {
        "id": "integer",
        "paid": "boolean",
    }
    assert schema["ordered"] == ae.ColumnSpec("datetime", datetime_format="%d/%m/%Y")
    validated = ae.validate_column_types(df, schema=schema)
    assert validated.status == "completed"
    assert all(c.invalid_count == 0 for c in validated.columns.values())
    assert set(validated.columns) == set(schema)


def test_inferred_schema_of_data_without_a_type_is_empty():
    res = ae.infer_column_semantics(pd.DataFrame({"a": [None, None]}))

    assert res.inferred_schema() == {}


def sentinel_frame():
    return pd.DataFrame(
        {
            "when": pd.to_datetime(
                ["1900-01-01", "2020-01-01", "1900-01-01", "2021-05-05"]
            ),
            "day": [dt.date(2099, 12, 31), dt.date(2020, 1, 1)] * 2,
            "age": [-999, 31, -999, 40],
            "city": ["N/A", "Lyon", "N/A", "Oslo"],
        }
    )


def test_sentinel_values_are_accepted_by_profile_missing_values():
    df = sentinel_frame()

    res = ae.detect_missing_sentinels(df)
    declared = res.sentinel_values()

    assert declared["age"] == [-999]
    assert declared["city"] == ["N/A"]
    assert declared["when"] == [pd.Timestamp("1900-01-01")]
    assert declared["day"] == [dt.date(2099, 12, 31)]
    profile = ae.profile_missing_values(df, sentinel_values=declared)
    assert profile.status == "completed"
    assert profile.columns["age"].sentinel_count == 2
    assert profile.columns["when"].sentinel_count == 2
    assert profile.columns["day"].sentinel_count == 2
    assert declared.note is None


def test_sentinel_values_can_be_limited_to_some_columns():
    res = ae.detect_missing_sentinels(sentinel_frame())

    assert list(res.sentinel_values(columns=["city"])) == ["city"]
    with pytest.raises(ae.errors.ColumnNotFoundError) as error:
        res.sentinel_values(columns=["City"])
    assert "Did you mean 'city'?" in str(error.value)


def test_sentinel_values_leave_out_masked_candidates_with_a_note():
    df = sentinel_frame()
    project = ae.Project(
        name="p", data_policy=ae.DataPolicy(sensitive_columns=["city"])
    )

    res = ae.detect_missing_sentinels(df, project=project, mask_sensitive=True)
    declared = res.sentinel_values()

    assert "city" not in declared
    assert declared["age"] == [-999]
    assert declared.note is not None
    assert "'city'" in declared.note
    assert "masking is on" in declared.note


def test_the_first_look_flow_runs_from_one_result_to_the_next():
    df = sentinel_frame()

    schema = ae.infer_column_semantics(df).inferred_schema()
    sentinels = ae.detect_missing_sentinels(df).sentinel_values()

    assert ae.validate_column_types(df, schema=schema).status == "completed"
    assert (
        ae.profile_missing_values(df, sentinel_values=sentinels).status == "completed"
    )


@pytest.mark.parametrize(
    "submodule",
    ["data", "statistics", "discovery", "project", "mcp", "errors", "results"],
)
def test_public_submodules_resolve_on_a_fresh_import(submodule):
    code = (
        "import agent_eda as ae, types\n"
        f"assert isinstance(ae.{submodule}, types.ModuleType), ae.{submodule}\n"
        f"assert ae.{submodule}.__name__ == 'agent_eda.{submodule}'\n"
        f"assert '{submodule}' in dir(ae)\n"
    )

    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr


def test_importing_the_package_loads_no_submodule_it_does_not_need():
    code = (
        "import sys, agent_eda\n"
        "loaded = [m for m in ('agent_eda.mcp', 'agent_eda.statistics', 'mcp') "
        "if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr


def test_the_mcp_submodule_gives_the_install_command_without_the_extra():
    code = (
        "import sys\n"
        "sys.modules['mcp'] = None  # makes `import mcp` fail, as without the extra\n"
        "import agent_eda as ae\n"
        "try:\n"
        "    ae.mcp.create_server()\n"
        "except ae.errors.MissingDependencyError as error:\n"
        "    print(error)\n"
    )

    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    assert "pip install 'agent-eda[mcp]'" in done.stdout


def test_a_misspelled_exception_name_points_to_the_errors_module():
    with pytest.raises(AttributeError) as error:
        ae.AgentEDAError  # noqa: B018 # ty: ignore[unresolved-attribute]  # the attribute access of a misspelled name is the test

    assert "ae.errors.AgentEDAError" in str(error.value)
    with pytest.raises(AttributeError, match=r"ae\.errors\.InvalidArgumentError"):
        ae.InvalidArgumentError  # noqa: B018 # ty: ignore[unresolved-attribute]  # the attribute access of a misspelled name is the test


def test_a_result_class_name_points_to_the_results_module():
    with pytest.raises(AttributeError, match=r"ae\.results\.SummarizeNumericResult"):
        ae.SummarizeNumericResult  # noqa: B018 # ty: ignore[unresolved-attribute]  # the attribute access of a misspelled name is the test


def test_infer_and_detect_docstrings_show_the_flow_in_their_examples():
    infer = inspect.getdoc(ae.infer_column_semantics) or ""
    detect = inspect.getdoc(ae.detect_missing_sentinels) or ""

    assert "res.inferred_schema()" in infer.split("Examples:")[1]
    assert "ae.validate_column_types(df, schema=schema)" in infer.split("Examples:")[1]
    assert "res.sentinel_values()" in detect.split("Examples:")[1]
    assert (
        "ae.profile_missing_values(df, sentinel_values=" in detect.split("Examples:")[1]
    )
