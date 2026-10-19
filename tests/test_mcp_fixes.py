"""The fix in a mistake's message, tested as an MCP client and a Python caller meet it.

What is wrong and what is allowed read the same on both surfaces. The fix
differs: an MCP caller passes JSON arguments and files, so its message points to
them, and a Python caller's message keeps the Python fix.
"""

import warnings

import pytest

pytest.importorskip(
    "mcp",
    reason=(
        "the MCP server tests need the mcp extra: run `uv sync` (uv 0.13 or later) or "
        "uv run --extra mcp pytest in this repository, or "
        "pip install 'agent-eda[mcp]' for an installed package"
    ),
)

import anyio
import pandas as pd
from mcp import Client
from python_syntax import python_in

import agent_eda as ae
from agent_eda.errors import AgentEDAError
from agent_eda.mcp import create_server

CSV = "order,city,score\nA1,Oslo,1.5\nA2,Lyon,2.5\nA3,Oslo,\nA4,Oslo,4.0\n"

# Each mistake: the MCP tool, its arguments besides data_path and project_path,
# and each fix the Python message gives with the MCP fix that takes its place.
MISTAKES = {
    "value-outside-choices": (
        "summarize_numeric",
        {"quantile_method": "lineer"},
        {
            "See help(ae.summarize_numeric) for every argument.": (
                "The MCP tool's input schema describes every argument."
            ),
        },
    ),
    "column-spec-value-outside-choices": (
        "validate_column_types",
        {"schema": {"score": {"logical_type": "int"}}},
        {
            "See help(ae.ColumnSpec) for every field.": (
                "The ColumnSpec definition in the MCP tool's input schema "
                "describes every field."
            ),
        },
    ),
    "column-spec-fields-that-clash": (
        "validate_column_types",
        {"schema": {"score": {"logical_type": "float", "datetime_format": "%Y"}}},
        {"declare logical_type='datetime'.": 'declare {"logical_type": "datetime"}.'},
    ),
    "column-spec-without-logical-type": (
        "validate_column_types",
        {"schema": {"score": {"nullable": True}}},
        {
            "such as ae.ColumnSpec(logical_type='integer', nullable=False).": (
                'such as {"logical_type": "integer", "nullable": false}.'
            ),
        },
    ),
    "empty-schema": (
        "validate_column_types",
        {"schema": {}},
        {
            "such as schema={'age': 'integer'}.": 'such as {"schema": {"age": "integer"}}.'
        },
    ),
    "empty-column-list": (
        "summarize_numeric",
        {"columns": []},
        {
            "or pass columns=None for every numeric column": (
                "or leave columns out for every numeric column"
            ),
        },
    ),
    # Converting a column is Python; another analysis or leaving it out isn't.
    "column-of-a-type-the-analysis-cannot-use": (
        "summarize_numeric",
        {"columns": ["city"]},
        {
            "; convert numbers stored as text with pd.to_numeric(df['city'], "
            "errors='coerce'); or": ", or",
        },
    ),
    "column-labels-that-form-a-multiindex": (
        "summarize_numeric",
        {},
        {
            "Flatten them first, such as df.columns = ['_'.join(map(str, c)) for c "
            "in df.columns].": "Flatten them in the data file first.",
        },
    ),
    "id-column-of-values-that-cannot-be-hashed": (
        "find_duplicate_rows",
        {"id_column": "tags"},
        {
            "or leave id_column=None to use the index labels.": (
                "or leave id_column out to use the default row IDs."
            ),
        },
    ),
    "regression-target-of-text": (
        "profile_target",
        {"target_column": "city", "task": "regression"},
        {
            "pass task='binary_classification' or task='multiclass_classification'; "
            "or convert numbers stored as text with "
            "pd.to_numeric(df['city'], errors='coerce').": (
                'pass {"task": "binary_classification"} or '
                '{"task": "multiclass_classification"}.'
            ),
        },
    ),
    "regression-target-of-datetimes": (
        "profile_target",
        {"target_column": "joined", "task": "regression"},
        {
            "Convert datetimes to numbers with "
            "(df['joined'] - df['joined'].min()).dt.total_seconds().": (
                "Pass a column of numbers instead."
            ),
        },
    ),
    "classification-target-of-datetimes": (
        "profile_target",
        {"target_column": "joined", "task": "binary_classification"},
        {
            "Convert datetimes to labels with df['joined'].astype(str), or to months "
            "with df['joined'].dt.strftime('%Y-%m').": (
                "Pass a column of labels instead."
            ),
        },
    ),
    "expected-labels-for-a-regression-task": (
        "profile_target",
        {"target_column": "score", "task": "regression", "expected_labels": ["a"]},
        {
            "pass task='binary_classification' or task='multiclass_classification' "
            "if": 'pass {"task": "binary_classification"} or '
            '{"task": "multiclass_classification"} if',
        },
    ),
    "empty-expected-labels": (
        "profile_target",
        {
            "target_column": "city",
            "task": "binary_classification",
            "expected_labels": [],
        },
        {
            "or pass expected_labels=None to leave the check out.": (
                "or leave expected_labels out to skip the check."
            ),
        },
    ),
    "mask-sensitive-without-a-project": (
        "summarize_categorical",
        {"mask_sensitive": True},
        {
            "got mask_sensitive=True without": 'got {"mask_sensitive": true} without',
            "Pass a project that names the sensitive columns, such as "
            "project=ae.Project(data_policy=ae.DataPolicy(sensitive_columns=['email']))"
            ".": "Pass a project_path whose [data_policy] table names the sensitive "
            'columns, such as sensitive_columns = ["email"].',
        },
    ),
    "mask-sensitive-with-a-project-that-declares-none": (
        "summarize_categorical",
        {"mask_sensitive": True},
        {
            "got mask_sensitive=True, but": 'got {"mask_sensitive": true}, but',
            "Name them in its data policy, such as "
            "project=ae.Project(data_policy=ae.DataPolicy(sensitive_columns=['email']))"
            ".": "Name them in the [data_policy] table of its project file, such as "
            'sensitive_columns = ["email"].',
        },
    ),
    "project-file-that-is-not-toml": (
        "summarize_categorical",
        {},
        {
            "Fix the file, or write it again with project.save(path, overwrite=True) "
            "from a project built in code; help(ae.Project.load) shows the format.": (
                "Fix the file; the project_path description in the MCP tool's input "
                "schema shows the format."
            ),
        },
    ),
    "project-file-without-the-data-policy": (
        "summarize_categorical",
        {},
        {
            "as help(ae.Project.load) shows": (
                "as the project_path description in the MCP tool's input schema shows"
            ),
        },
    ),
    "project-file-without-a-key": (
        "summarize_categorical",
        {},
        {
            "as help(ae.Project.load) shows": (
                "as the project_path description in the MCP tool's input schema shows"
            ),
        },
    ),
    "project-file-that-masks-nothing": (
        "summarize_categorical",
        {},
        {
            "got mask_sensitive=True but": "got mask_sensitive = true but",
            "such as ae.DataPolicy(sensitive_columns=['email'], mask_sensitive=True). "
            "help(ae.Project.load) shows": 'such as sensitive_columns = ["email"]. The '
            "project_path description in the MCP tool's input schema shows",
        },
    ),
}

# The text of the project file a mistake passes, as project_path over MCP and
# as project=ae.Project.load(path) in Python.
PROJECT_FILES = {
    "mask-sensitive-with-a-project-that-declares-none": (
        "[data_policy]\nsensitive_columns = []\nmask_sensitive = false\n"
    ),
    "project-file-that-is-not-toml": "[data_policy\n",
    "project-file-without-the-data-policy": 'name = "shop"\n',
    "project-file-without-a-key": '[data_policy]\nsensitive_columns = ["city"]\n',
    "project-file-that-masks-nothing": (
        "[data_policy]\nsensitive_columns = []\nmask_sensitive = true\n"
    ),
}

# The mistakes whose data only a Parquet file can hold; the rest read CSV.
PARQUET = {
    "column-labels-that-form-a-multiindex": pd.DataFrame(
        [[1.5, 2.0]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
    ),
    "id-column-of-values-that-cannot-be-hashed": pd.DataFrame(
        {"tags": [["a"], ["b"]], "score": [1.5, 2.0]}
    ),
    "regression-target-of-datetimes": pd.DataFrame(
        {"joined": pd.to_datetime(["2024-01-05", "2024-02-05"])}
    ),
    "classification-target-of-datetimes": pd.DataFrame(
        {"joined": pd.to_datetime(["2024-01-05", "2024-02-05"])}
    ),
}


@pytest.mark.parametrize("mistake", list(MISTAKES))
def test_a_mistake_over_mcp_gives_the_python_message_with_an_mcp_fix(tmp_path, mistake):
    name, arguments, fixes = MISTAKES[mistake]
    path, data = data_file(tmp_path, PARQUET.get(mistake))
    arguments = {"data_path": str(path), **arguments}
    project_path = tmp_path / "project.toml"
    if mistake in PROJECT_FILES:
        project_path.write_text(PROJECT_FILES[mistake], encoding="utf-8")
        arguments["project_path"] = str(project_path)
    with pytest.raises(AgentEDAError) as raised:
        call_in_python(name, data, arguments)
    python = raised.value
    expected = str(python)
    for python_fix, mcp_fix in fixes.items():
        assert python_fix in expected
        expected = expected.replace(python_fix, mcp_fix)
    # A key of a mapping is written as a JSON pointer, schema/score.
    expected = expected.replace("schema['score']", "schema/score")
    shown = repr(str(project_path))
    if shown in expected:
        # The MCP server reads the file project_path leads to, not Project.load.
        expected = expected.replace("Project.load", "The MCP server").replace(
            shown, f"the project_path file {shown}"
        )

    response = anyio.run(call, name, arguments)

    assert response.is_error
    assert response.structured_content == {"code": python.code, "message": expected}
    assert [block.text for block in response.content] == [
        response.structured_content["message"]
    ]
    assert python_in(response.structured_content["message"]) == []


def call_in_python(name, data, arguments):
    """Make the Python call that matches the MCP call: the data read from
    data_path, and the project loaded from project_path."""
    options = {k: v for k, v in arguments.items() if not k.endswith("_path")}
    if "project_path" in arguments:
        options["project"] = ae.Project.load(arguments["project_path"])
    return getattr(ae, name)(data, **options)


def data_file(tmp_path, frame):
    """The data file, CSV unless a frame is given, and the DataFrame the MCP
    server reads from it."""
    if frame is None:
        path = tmp_path / "shop.csv"
        path.write_text(CSV, encoding="utf-8")
        return path, pd.DataFrame(
            {
                "order": ["A1", "A2", "A3", "A4"],
                "city": ["Oslo", "Lyon", "Oslo", "Oslo"],
                "score": [1.5, 2.5, None, 4.0],
            }
        )
    path = tmp_path / "shop.parquet"
    frame.to_parquet(path)
    with warnings.catch_warnings():
        # pandas 2.2 deprecates the internals pyarrow 14 builds DataFrames from.
        warnings.filterwarnings(
            "ignore",
            message="Passing a BlockManager to DataFrame is deprecated",
            category=DeprecationWarning,
        )
        return path, pd.read_parquet(path)


async def call(name, arguments):
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)
