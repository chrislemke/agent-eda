"""Saving a Project to a TOML file once and loading it in later scripts."""

import codecs
import doctest
import inspect
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pydantic
import pytest
from hypothesis import given
from hypothesis import strategies as st

import agent_eda as ae
from agent_eda.errors import (
    AgentEDAError,
    InvalidArgumentError,
    ProjectFileError,
    ProjectFileExistsError,
    ProjectFileNotFoundError,
)

EMAIL_MASKED = ae.Project(
    name="churn",
    data_policy=ae.DataPolicy(sensitive_columns=["email"], mask_sensitive=True),
)
HEADER = (
    "# An agent-eda Project. Load it with ae.Project.load(path).\nformat_version = 1\n"
)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def stands_alone(error):
    """Whether a traceback shows the error without another one chained to it."""
    return error.__cause__ is None and (
        error.__context__ is None or error.__suppress_context__
    )


@pytest.fixture
def project_file(tmp_path, monkeypatch):
    """Write project.toml in a new working directory and return its name."""
    monkeypatch.chdir(tmp_path)

    def write_project_file(text):
        write(tmp_path / "project.toml", text)
        return "project.toml"

    return write_project_file


def test_load_reads_a_name_and_a_data_policy_table(tmp_path):
    path = write(
        tmp_path / "project.toml",
        "# Written by hand\n"
        'name = "support tickets"\n'
        "\n"
        "[data_policy]\n"
        "sensitive_columns = ['customer', \"email\"]  # literal and basic strings\n"
        "mask_sensitive = true\n",
    )

    project = ae.Project.load(path)

    assert project == ae.Project(
        name="support tickets",
        data_policy=ae.DataPolicy(
            sensitive_columns=["customer", "email"], mask_sensitive=True
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        'name = "churn"\n'
        'data_policy = { sensitive_columns = ["email"], mask_sensitive = true }\n',
        'name = "churn"\n'
        'data_policy.sensitive_columns = ["email"]\n'
        "data_policy.mask_sensitive = true\n",
        "name = 'churn'\n"
        "[data_policy]\n"
        "mask_sensitive = true\n"
        "sensitive_columns = [\n"
        '  "email",  # the one sensitive column\n'
        "]\n",
        'name = "churn"\n[data_policy]\nsensitive_columns = "email"\nmask_sensitive = true\n',
    ],
    ids=["inline-table", "dotted-keys", "multi-line-array", "one-name"],
)
def test_load_reads_any_toml_spelling_of_the_same_keys(tmp_path, text):
    path = write(tmp_path / "project.toml", text)

    assert ae.Project.load(path) == EMAIL_MASKED


NO_POLICY = "[data_policy]\nsensitive_columns = []\nmask_sensitive = false\n"


def test_a_file_without_a_name_loads_an_unnamed_project(tmp_path):
    path = write(tmp_path / "project.toml", NO_POLICY)

    assert ae.Project.load(path) == ae.Project()


WHOLE_POLICY = (
    "A project file states its whole data policy, so that a file cut short can't "
    "leave masking off quietly"
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "",
            "Project.load found no [data_policy] table in 'project.toml'. "
            f"{WHOLE_POLICY}: add a [data_policy] line, with single brackets, "
            "followed by sensitive_columns and mask_sensitive, as "
            "help(ae.Project.load) shows.",
        ),
        (
            'name = "churn"\n',
            "Project.load found no [data_policy] table in 'project.toml'. "
            f"{WHOLE_POLICY}: add a [data_policy] line, with single brackets, "
            "followed by sensitive_columns and mask_sensitive, as "
            "help(ae.Project.load) shows.",
        ),
        (
            "[data_policy]\n",
            "Project.load found no sensitive_columns and mask_sensitive in the "
            f"[data_policy] table of 'project.toml'. {WHOLE_POLICY}: add "
            "sensitive_columns and mask_sensitive below the [data_policy] line, as "
            "help(ae.Project.load) shows.",
        ),
        (
            '[data_policy]\nsensitive_columns = ["email"]\n',
            "Project.load found no mask_sensitive in the [data_policy] table of "
            f"'project.toml'. {WHOLE_POLICY}: add mask_sensitive below the "
            "[data_policy] line, as help(ae.Project.load) shows.",
        ),
        (
            "[data_policy]\nmask_sensitive = false\n",
            "Project.load found no sensitive_columns in the [data_policy] table of "
            f"'project.toml'. {WHOLE_POLICY}: add sensitive_columns below the "
            "[data_policy] line, as help(ae.Project.load) shows.",
        ),
    ],
    ids=["empty", "no-table", "empty-table", "no-mask-key", "no-columns-key"],
)
def test_a_file_without_its_whole_data_policy_raises(project_file, text, message):
    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load(project_file(text))

    assert caught.value.code == "invalid_project_file"
    assert str(caught.value) == message
    assert stands_alone(caught.value)


@pytest.mark.parametrize("mask", [True, False])
def test_a_saved_file_cut_short_raises_instead_of_loading(tmp_path, mask):
    project = ae.Project(
        name="churn",
        data_policy=ae.DataPolicy(
            sensitive_columns=["email", "name"], mask_sensitive=mask
        ),
    )
    text = project.save(tmp_path / "project.toml").read_text(encoding="utf-8")
    cut = tmp_path / "cut.toml"

    for end in range(len(text.rstrip("\n"))):
        write(cut, text[:end])
        with pytest.raises(ProjectFileError):
            ae.Project.load(cut)


def test_load_takes_the_path_as_a_string_too(tmp_path, monkeypatch):
    write(tmp_path / "project.toml", 'name = "churn"\n' + NO_POLICY)
    monkeypatch.chdir(tmp_path)

    assert ae.Project.load("project.toml") == ae.Project(name="churn")
    assert ae.Project.load(str(tmp_path / "project.toml")) == ae.Project(name="churn")


def test_a_missing_file_raises_with_its_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ProjectFileNotFoundError) as caught:
        ae.Project.load("project.toml")

    error = caught.value
    assert isinstance(error, FileNotFoundError)
    assert isinstance(error, ProjectFileError)
    assert isinstance(error, AgentEDAError)
    assert error.code == "project_file_not_found"
    assert str(error) == (
        f"Project.load found no file at 'project.toml' ({tmp_path / 'project.toml'})."
        " Save a project there first with project.save('project.toml'), or pass the "
        "path of an existing project file."
    )
    assert stands_alone(error)


def test_save_writes_the_name_and_the_data_policy_table(tmp_path):
    path = tmp_path / "project.toml"
    project = ae.Project(
        name="churn",
        data_policy=ae.DataPolicy(
            sensitive_columns=["email", "name"], mask_sensitive=True
        ),
    )

    project.save(path)

    assert path.read_bytes().decode("utf-8") == (
        HEADER + 'name = "churn"\n'
        "\n"
        "[data_policy]\n"
        "sensitive_columns = [\n"
        '    "email",\n'
        '    "name",\n'
        "]\n"
        "mask_sensitive = true\n"
    )


def test_save_leaves_out_a_missing_name_and_writes_every_policy_key(tmp_path):
    path = tmp_path / "project.toml"

    ae.Project().save(path)

    assert path.read_bytes().decode("utf-8") == (
        HEADER + "\n[data_policy]\nsensitive_columns = []\nmask_sensitive = false\n"
    )


def test_equal_projects_save_to_identical_files(tmp_path):
    first = ae.Project(name="churn", data_policy={"sensitive_columns": ["b", "a"]})
    second = ae.Project.model_validate(
        {"name": "churn", "data_policy": {"sensitive_columns": ("b", "a")}}
    )

    first.save(tmp_path / "first.toml")
    second.save(tmp_path / "second.toml")

    assert (tmp_path / "first.toml").read_bytes() == (
        tmp_path / "second.toml"
    ).read_bytes()


def test_save_returns_the_absolute_path_to_load_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path = EMAIL_MASKED.save("project.toml")

    assert path == tmp_path / "project.toml"
    assert ae.Project.load(path) == EMAIL_MASKED


def test_save_refuses_to_overwrite_an_existing_file(tmp_path, monkeypatch):
    old = write(tmp_path / "project.toml", "name = 'old'\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ProjectFileExistsError) as caught:
        EMAIL_MASKED.save("project.toml")

    error = caught.value
    assert isinstance(error, FileExistsError)
    assert isinstance(error, ProjectFileError)
    assert isinstance(error, AgentEDAError)
    assert error.code == "project_file_exists"
    assert str(error) == (
        f"Project.save found a file at 'project.toml' ({old}) already, and replaces "
        "a file only when told to. Pass overwrite=True to replace it, or pass "
        "another path."
    )
    assert stands_alone(error)
    assert old.read_text(encoding="utf-8") == "name = 'old'\n"


def test_save_with_overwrite_replaces_an_existing_file(tmp_path):
    path = write(tmp_path / "project.toml", "name = 'old'\n")

    EMAIL_MASKED.save(path, overwrite=True)

    assert ae.Project.load(path) == EMAIL_MASKED


def test_save_and_load_use_no_toml_library_but_the_standard_one(tmp_path):
    path = tmp_path / "project.toml"
    code = (
        "import sys\n"
        "import agent_eda as ae\n"
        "project = ae.Project(name='churn')\n"
        f"assert ae.Project.load(project.save({str(path)!r})) == project\n"
        # coverage's subprocess patch loads its own copy of the config reader
        "print(' '.join(sorted(name for name in sys.modules\n"
        "    if 'toml' in name and not name.startswith('coverage'))))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    modules = completed.stdout.split()
    assert "tomllib" in modules
    assert all(name.split(".")[0] == "tomllib" for name in modules), modules


def test_a_project_saved_by_one_script_masks_in_the_next(tmp_path):
    first = (
        "import agent_eda as ae\n"
        "policy = ae.DataPolicy(sensitive_columns=['customer'], mask_sensitive=True)\n"
        "ae.Project(data_policy=policy).save('project.toml')\n"
    )
    second = (
        "import pandas as pd\n"
        "import agent_eda as ae\n"
        "project = ae.Project.load('project.toml')\n"
        "df = pd.DataFrame({'customer': ['Ann Lee', 'Bo Chu', 'Ann Lee']})\n"
        "print(ae.summarize_categorical(df, project=project))\n"
    )

    for script in [first, second]:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    assert "<masked 1>" in completed.stdout
    assert "Ann Lee" not in completed.stdout


# Text that a TOML string can't hold as it is, or that looks like TOML syntax.
AWKWARD_TEXT = [
    'say "hi"',
    "back\\slash",
    "\\u0041 as text",
    "line\nbreak",
    "tab\tstop",
    "carriage\rreturn",
    "form\ffeed",
    "back\bspace",
    "nul\x00",
    "escape\x1b",
    "delete\x7f",
    "next line\x85",
    "",
    "   ",
    "it's",
    "'''",
    '"""',
    "# not a comment",
    "a = b",
    "[data_policy]",
    "naïve café",
    "日本語",
    "😀",
    "zero\N{ZERO WIDTH SPACE}width",
    "no\xa0break",
    "\N{ZERO WIDTH NO-BREAK SPACE}bom",
    "line\N{LINE SEPARATOR}separator",
    "private" + chr(0xE000) + "use",
]


@pytest.mark.parametrize("text", AWKWARD_TEXT, ids=ascii)
def test_awkward_names_survive_a_save_and_load(tmp_path, text):
    project = ae.Project(
        name=text,
        data_policy=ae.DataPolicy(
            sensitive_columns=["email", text], mask_sensitive=True
        ),
    )

    loaded = ae.Project.load(project.save(tmp_path / "project.toml"))

    assert loaded == project


def test_save_escapes_what_a_toml_string_cannot_hold_as_it_is(tmp_path):
    project = ae.Project(
        name='say "hi"\tto\\all\n',
        data_policy={
            "sensitive_columns": ["nul\x00", "zero\N{ZERO WIDTH SPACE}width", "😀 ok"]
        },
    )

    path = project.save(tmp_path / "project.toml")

    assert path.read_bytes().decode("utf-8") == (
        HEADER + 'name = "say \\"hi\\"\\tto\\\\all\\n"\n'
        "\n"
        "[data_policy]\n"
        "sensitive_columns = [\n"
        '    "nul\\u0000",\n'
        '    "zero\\u200Bwidth",\n'
        '    "😀 ok",\n'
        "]\n"
        "mask_sensitive = false\n"
    )


@given(
    name=st.none() | st.text(),
    columns=st.lists(st.text(), unique=True, max_size=4),
    mask=st.booleans(),
)
def test_every_project_survives_a_save_and_load(name, columns, mask):
    project = ae.Project(
        name=name,
        data_policy=ae.DataPolicy(
            sensitive_columns=columns, mask_sensitive=mask and bool(columns)
        ),
    )

    with tempfile.TemporaryDirectory() as folder:
        loaded = ae.Project.load(project.save(Path(folder) / "project.toml"))

    assert loaded == project


SURROGATE = chr(0xDCE9)  # a lone surrogate, as errors="surrogateescape" leaves


@pytest.mark.parametrize(
    ("project", "field"),
    [
        (ae.Project(name="caf" + SURROGATE), "name"),
        (
            ae.Project(data_policy={"sensitive_columns": ["email", "caf" + SURROGATE]}),
            "data_policy.sensitive_columns[1]",
        ),
    ],
    ids=["name", "column"],
)
def test_text_toml_cannot_hold_raises_before_anything_is_written(
    tmp_path, project, field
):
    path = tmp_path / "project.toml"

    with pytest.raises(ProjectFileError) as caught:
        project.save(path)

    message = str(caught.value)
    assert caught.value.code == "invalid_project_file"
    assert message == (
        f"Project.save can't write {field}, {'caf' + SURROGATE!r}, to TOML, which "
        f"holds only Unicode text: {SURROGATE!r} is a lone surrogate, as left by "
        "decoding bytes with errors='surrogateescape'. Use text without it, or build "
        "the project in code in each script instead of saving it."
    )
    assert message.encode("utf-8")
    assert not path.exists()


ALLOWED = (
    "Allowed: name and data_policy at the top level; sensitive_columns and "
    "mask_sensitive in [data_policy]."
)
TABLES = (
    "Keys below a [data_policy] line belong to that table, and keys above it to the "
    "top level."
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            'nmae = "churn"\n',
            "Project.load found an unknown key 'nmae' at the top level of "
            f"'project.toml'. Did you mean 'name'? {ALLOWED}",
        ),
        (
            '[data_policy]\nsensitive_columns = ["email"]\nmask_sensitve = true\n',
            "Project.load found an unknown key 'mask_sensitve' in the [data_policy] "
            f"table of 'project.toml'. Did you mean 'mask_sensitive'? {ALLOWED}",
        ),
        (
            '[data_policy]\nSensitive_Columns = ["email"]\n',
            "Project.load found an unknown key 'Sensitive_Columns' in the "
            "[data_policy] table of 'project.toml'. Did you mean 'sensitive_columns'? "
            f"{ALLOWED}",
        ),
        (
            "[data_polcy]\nmask_sensitive = true\n",
            "Project.load found an unknown key 'data_polcy' at the top level of "
            f"'project.toml'. Did you mean 'data_policy'? {ALLOWED}",
        ),
        (
            'sensitive_columns = ["email"]\n',
            "Project.load found an unknown key 'sensitive_columns' at the top level "
            f"of 'project.toml'. It belongs in [data_policy]. {TABLES} {ALLOWED}",
        ),
        (
            '[data_policy]\nsensitive_columns = ["email"]\nname = "churn"\n',
            "Project.load found an unknown key 'name' in the [data_policy] table of "
            f"'project.toml'. It belongs at the top level. {TABLES} {ALLOWED}",
        ),
        (
            "sensitive_column = []\n",
            "Project.load found an unknown key 'sensitive_column' at the top level of "
            "'project.toml'. Did you mean 'sensitive_columns' in [data_policy]? "
            f"{TABLES} {ALLOWED}",
        ),
        (
            '[project]\nname = "churn"\n',
            "Project.load found an unknown key 'project' at the top level of "
            f"'project.toml'. {ALLOWED}",
        ),
    ],
    ids=[
        "misspelled",
        "misspelled-mask-key",
        "other-case",
        "misspelled-table",
        "policy-key-at-the-top",
        "name-in-the-table",
        "misspelled-policy-key-at-the-top",
        "no-close-key",
    ],
)
def test_an_unknown_key_raises_with_the_closest_key_and_the_allowed_ones(
    project_file, text, message
):
    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load(project_file(text))

    error = caught.value
    assert isinstance(error, ValueError)
    assert isinstance(error, AgentEDAError)
    assert error.code == "unknown_project_key"
    assert str(error) == message
    assert stands_alone(error)


def test_every_unknown_key_is_named_at_once(project_file):
    path = project_file(
        'nmae = "churn"\n'
        "roles = []\n"
        "[data_policy]\n"
        "mask_sensitve = true\n"
        'name = "churn"\n'
    )

    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load(path)

    assert caught.value.code == "unknown_project_key"
    assert str(caught.value) == (
        "Project.load found 4 unknown keys in 'project.toml': 'nmae' at the top "
        "level (did you mean 'name'?), 'roles' at the top level, 'mask_sensitve' in "
        "[data_policy] (did you mean 'mask_sensitive'?) and 'name' in [data_policy] "
        f"(it belongs at the top level). {TABLES} {ALLOWED}"
    )


REWRITE = (
    "Fix the file, or write it again with project.save(path, overwrite=True) from a "
    "project built in code; help(ae.Project.load) shows the format."
)
TOML_VALUES = (
    "TOML puts text in quotes, writes true and false in lower case and has no null, "
    "so leave out a key that has no value."
)


CUT_SHORT = (
    "The file stops in the middle of its last key, value or table line, such as an "
    "array without its closing ]."
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            'name = "churn"\n[data_policy]\nmask_sensitive = True\n',
            "Project.load found invalid TOML in 'project.toml' at line 3, column 18: "
            f"Invalid value. {TOML_VALUES} {REWRITE}",
        ),
        (
            "name = churn\r\n",
            "Project.load found invalid TOML in 'project.toml' at line 1, column 8: "
            f"Invalid value. {TOML_VALUES} {REWRITE}",
        ),
        (
            'name = "a"\nname = "b"\n',
            "Project.load found invalid TOML in 'project.toml' at line 2, column 11: "
            f"Cannot overwrite a value. {REWRITE}",
        ),
        (
            '[data_policy]\nsensitive_columns = [\n  "email",\n\n\n',
            "Project.load found invalid TOML in 'project.toml' at its end, after line "
            f"3: Invalid value. {CUT_SHORT} {REWRITE}",
        ),
        (
            'name = "churn"\nmask_sensitive',
            "Project.load found invalid TOML in 'project.toml' at its end, after line "
            f"2: Expected '=' after a key in a key/value pair. {CUT_SHORT} {REWRITE}",
        ),
        (
            "[data_policy",
            "Project.load found invalid TOML in 'project.toml' at its end, after line "
            "1: Expected ']' at the end of a table declaration. "
            f"{CUT_SHORT} {REWRITE}",
        ),
    ],
    ids=[
        "python-bool",
        "unquoted-text",
        "repeated-key",
        "open-array",
        "key-without-value",
        "open-table-line",
    ],
)
def test_invalid_toml_raises_with_the_line_at_fault(project_file, text, message):
    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load(project_file(text))

    error = caught.value
    assert isinstance(error, ValueError)
    assert error.code == "invalid_project_file"
    assert str(error) == message
    assert stands_alone(error)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '{"email": "ann.lee@example.com", "plan": "pro"}\n',
            "Project.load found invalid TOML in 'project.toml' at line 1, column 1: "
            f"Invalid statement. {REWRITE}",
        ),
        (
            "plan,email\npro,ann.lee@example.com\n",
            "Project.load found invalid TOML in 'project.toml' at line 1, column 5: "
            f"Expected '=' after a key in a key/value pair. {REWRITE}",
        ),
    ],
    ids=["json-lines", "csv"],
)
def test_a_data_file_loaded_by_mistake_keeps_its_values_out_of_the_message(
    project_file, text, message
):
    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load(project_file(text))

    assert str(caught.value) == message
    assert "example.com" not in str(caught.value)


def test_a_file_that_is_not_utf8_text_raises_with_the_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project.toml").write_bytes(b'name = "churn"\nname2 = "caf\xe9"\n')

    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load("project.toml")

    assert caught.value.code == "invalid_project_file"
    assert str(caught.value) == (
        "Project.load found bytes that aren't UTF-8 text in 'project.toml' at line "
        f"2. A project file is TOML, which is UTF-8 text. {REWRITE}"
    )
    assert stands_alone(caught.value)


def test_a_file_that_starts_with_a_byte_order_mark_loads(tmp_path):
    path = tmp_path / "project.toml"
    path.write_bytes(codecs.BOM_UTF8 + ('name = "churn"\n' + NO_POLICY).encode())

    assert ae.Project.load(path) == ae.Project(name="churn")


TABLE_FORM = (
    "a [data_policy] line, with single brackets, followed by sensitive_columns and "
    "mask_sensitive, as help(ae.Project.load) shows."
)
COLUMNS_FORM = 'an array of strings, such as sensitive_columns = ["email", "name"].'
CANT_BUILD = "Project.load can't build a project from 'project.toml': "
FORMAT = " help(ae.Project.load) shows how a project file writes each key."


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '[data_policy]\nsensitive_columns = ["email"]\nmask_sensitive = "yes"\n',
            "Project.load found a string for mask_sensitive in the [data_policy] "
            "table of 'project.toml', but mask_sensitive takes true or false, without "
            "quotes.",
        ),
        (
            "name = 3\n" + NO_POLICY,
            "Project.load found an integer for name at the top level of "
            "'project.toml', but name takes a string, such as name = \"churn\".",
        ),
        (
            "[data_policy]\nsensitive_columns = true\nmask_sensitive = true\n",
            "Project.load found a boolean for sensitive_columns in the [data_policy] "
            f"table of 'project.toml', but sensitive_columns takes {COLUMNS_FORM}",
        ),
        (
            '[data_policy]\nsensitive_columns = ["email", ["name"]]\n'
            "mask_sensitive = false\n",
            "Project.load found an array at sensitive_columns[1] in the [data_policy] "
            f"table of 'project.toml', but sensitive_columns takes {COLUMNS_FORM}",
        ),
        (
            "[data_policy]\nsensitive_columns = [0]\nmask_sensitive = false\n",
            "Project.load found an integer at sensitive_columns[0] in the "
            "[data_policy] table of 'project.toml', but sensitive_columns takes "
            f"{COLUMNS_FORM}",
        ),
        (
            "[[data_policy]]\nsensitive_columns = []\nmask_sensitive = false\n",
            "Project.load found an array of tables for data_policy in 'project.toml', "
            f"but data_policy is one table: {TABLE_FORM}",
        ),
        (
            'data_policy = "none"\n',
            "Project.load found a string for data_policy in 'project.toml', but "
            f"data_policy is one table: {TABLE_FORM}",
        ),
        (
            "[data_policy]\nsensitive_columns = []\nmask_sensitive = true\n",
            f"{CANT_BUILD}DataPolicy got mask_sensitive=True but no "
            "sensitive_columns, so there would be nothing to mask. Name the sensitive "
            "columns, such as ae.DataPolicy(sensitive_columns=['email'], "
            f"mask_sensitive=True).{FORMAT}",
        ),
        (
            '[data_policy]\nsensitive_columns = ["email", "email"]\n'
            "mask_sensitive = false\n",
            f"{CANT_BUILD}DataPolicy got 'email' more than once in sensitive_columns. "
            f"List each column once.{FORMAT}",
        ),
    ],
    ids=[
        "text-for-a-boolean",
        "number-for-the-name",
        "boolean-for-the-columns",
        "array-for-a-column",
        "number-for-a-column",
        "array-of-tables",
        "text-for-the-table",
        "nothing-to-mask",
        "column-listed-twice",
    ],
)
def test_a_value_a_project_cannot_take_raises_in_the_package_format(
    project_file, text, message
):
    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load(project_file(text))

    error = caught.value
    assert not isinstance(error, pydantic.ValidationError)
    assert isinstance(error, ValueError)
    assert error.code == "invalid_project_file"
    assert str(error) == message
    assert stands_alone(error)


@pytest.mark.parametrize(
    ("path", "shown"),
    [
        (3, "3"),
        (None, "None"),
        (b"project.toml", "a bytes"),
        (["project.toml"], "a list of 1 item"),
    ],
    ids=["number", "none", "bytes", "list"],
)
@pytest.mark.parametrize("call", ["load", "save"])
def test_a_path_of_the_wrong_type_raises_with_the_types_allowed(call, path, shown):
    method = ae.Project.load if call == "load" else ae.Project().save

    with pytest.raises(InvalidArgumentError) as caught:
        method(path)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"Project.{call} got {shown} for path, but path must be a str or a "
        "pathlib.Path, such as 'project.toml'. See help(ae.Project."
        f"{call}) for every argument."
    )


@pytest.mark.parametrize("call", ["load", "save"])
def test_a_directory_for_the_path_raises_with_a_file_path_in_it(
    tmp_path, monkeypatch, call
):
    (tmp_path / "config").mkdir()
    monkeypatch.chdir(tmp_path)
    method = ae.Project.load if call == "load" else ae.Project().save

    with pytest.raises(InvalidArgumentError) as caught:
        method("config")

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"Project.{call} got the directory 'config' ({tmp_path / 'config'}) for "
        "path, but path must name a file in it, such as "
        f"'{Path('config', 'project.toml')}'."
    )


def test_save_into_a_missing_directory_raises_with_how_to_create_it(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvalidArgumentError) as caught:
        ae.Project().save(Path("config", "project.toml"))

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"Project.save got '{Path('config', 'project.toml')}' for path, but its "
        f"directory {tmp_path / 'config'} doesn't exist. Create it first, such as "
        f"with os.makedirs({str(tmp_path / 'config')!r}), or pass a path in an "
        "existing directory."
    )
    assert not (tmp_path / "config").exists()


def test_save_into_a_file_as_if_it_were_a_directory_raises(tmp_path, monkeypatch):
    write(tmp_path / "notes.txt", "notes\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvalidArgumentError) as caught:
        ae.Project().save(Path("notes.txt", "project.toml"))

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"Project.save got '{Path('notes.txt', 'project.toml')}' for path, but "
        f"{tmp_path / 'notes.txt'} is a file, so it can't hold one. Pass a path in a "
        "directory."
    )


def test_load_from_a_file_as_if_it_were_a_directory_finds_no_file(
    tmp_path, monkeypatch
):
    write(tmp_path / "notes.txt", "notes\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ProjectFileNotFoundError) as caught:
        ae.Project.load(Path("notes.txt", "project.toml"))

    assert str(caught.value).startswith(
        f"Project.load found no file at '{Path('notes.txt', 'project.toml')}' "
        f"({tmp_path / 'notes.txt' / 'project.toml'})."
    )


@pytest.mark.parametrize("call", ["load", "save"])
def test_a_path_ending_in_a_separator_raises_as_it_names_a_directory(
    tmp_path, monkeypatch, call
):
    monkeypatch.chdir(tmp_path)
    method = ae.Project.load if call == "load" else ae.Project().save
    path = "config" + os.sep

    with pytest.raises(InvalidArgumentError) as caught:
        method(path)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"Project.{call} got {path!r} for path, whose last {os.sep!r} makes it name "
        "a directory, but path must name a file, such as "
        f"{os.path.join(path, 'project.toml')!r}."
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("", "got an empty path, but path must name a file, such as 'project.toml'."),
        (
            "project" + chr(0) + ".toml",
            f"got {'project' + chr(0) + '.toml'!r} for path, but a path can't hold a "
            "null character. Pass the path of a file, such as 'project.toml'.",
        ),
    ],
    ids=["empty", "null-character"],
)
@pytest.mark.parametrize("call", ["load", "save"])
def test_a_path_that_names_no_file_raises(tmp_path, monkeypatch, call, path, message):
    monkeypatch.chdir(tmp_path)
    method = ae.Project.load if call == "load" else ae.Project().save

    with pytest.raises(InvalidArgumentError) as caught:
        method(path)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == f"Project.{call} {message}"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("overwrite", ["yes", 1, None])
def test_overwrite_takes_only_true_or_false(tmp_path, overwrite):
    path = tmp_path / "project.toml"

    with pytest.raises(InvalidArgumentError) as caught:
        ae.Project().save(path, overwrite=overwrite)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"Project.save got {overwrite!r} for overwrite, but overwrite must be bool: "
        "True replaces an existing file, False leaves it. See help(ae.Project.save) "
        "for every argument."
    )
    assert not path.exists()


def test_a_path_starting_with_a_tilde_means_the_home_directory(tmp_path, monkeypatch):
    # os.path.expanduser reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    path = EMAIL_MASKED.save("~/project.toml")

    assert path == tmp_path / "project.toml"
    assert ae.Project.load("~/project.toml") == EMAIL_MASKED


def test_overwrite_is_keyword_only_and_off_by_default():
    overwrite = inspect.signature(ae.Project.save).parameters["overwrite"]

    assert overwrite.kind is inspect.Parameter.KEYWORD_ONLY
    assert overwrite.default is False


@pytest.mark.parametrize("method", ["load", "save"])
def test_the_docstring_examples_of_load_and_save_run(method):
    runner = doctest.DocTestRunner()
    report = []
    finder = doctest.DocTestFinder()
    function = getattr(ae.Project, method)
    for test in finder.find(function, name=f"Project.{method}", globs={}):
        runner.run(test, out=report.append)

    assert runner.tries > 0
    assert runner.failures == 0, "".join(report)


def test_help_on_load_shows_the_format_of_the_file():
    doc = inspect.getdoc(ae.Project.load)
    assert doc

    assert "[data_policy]" in doc
    for key in ["name", "sensitive_columns", "mask_sensitive"]:
        assert f"{key} = " in doc


def test_help_on_a_project_says_how_to_use_it_in_later_scripts():
    doc = inspect.getdoc(ae.Project)
    assert doc
    text = " ".join(doc.split())

    assert "project.save('project.toml')" in text
    assert "ae.Project.load('project.toml')" in text


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_named_pipe_raises_instead_of_waiting_for_a_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.mkfifo(tmp_path / "project.toml")

    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load("project.toml")

    assert caught.value.code == "invalid_project_file"
    assert str(caught.value) == (
        f"Project.load found 'project.toml' ({tmp_path / 'project.toml'}), but it "
        "isn't a regular file, such as a named pipe or a device, so it can't be a "
        "project file. Pass the path of a project TOML file, such as one "
        "project.save wrote."
    )
    assert stands_alone(caught.value)


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="needs /dev/zero")
def test_a_device_raises_instead_of_reading_without_end():
    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load("/dev/zero")

    assert caught.value.code == "invalid_project_file"
    assert "isn't a regular file" in str(caught.value)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that bind the user",
)
def test_a_file_without_read_permission_raises_with_the_fix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write(tmp_path / "project.toml", EMAIL_MASKED_TEXT)
    path.chmod(0)
    try:
        with pytest.raises(ProjectFileError) as caught:
            ae.Project.load("project.toml")
    finally:
        path.chmod(0o600)

    assert caught.value.code == "invalid_project_file"
    assert str(caught.value) == (
        f"Project.load can't read 'project.toml' ({tmp_path / 'project.toml'}): "
        "Permission denied. Give the user that runs this process read permission "
        "on the file, or pass the path of a project file it can read."
    )
    assert stands_alone(caught.value)


def test_a_file_too_large_for_a_project_raises_without_reading_it_all(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    with (tmp_path / "project.toml").open("wb") as file:
        file.truncate(1024 * 1024 + 1)

    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load("project.toml")

    assert caught.value.code == "invalid_project_file"
    assert str(caught.value) == (
        f"Project.load found more than 1 MiB in 'project.toml' "
        f"({tmp_path / 'project.toml'}), but a project file holds only a name and "
        "a data policy, which take a few KiB at most. Check that the path names a "
        "project file, not a data file."
    )


EMAIL_MASKED_TEXT = (
    '[data_policy]\nsensitive_columns = ["email"]\nmask_sensitive = true\n'
)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_save_refuses_a_named_pipe_instead_of_waiting_for_a_reader(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    os.mkfifo(tmp_path / "project.toml")

    with pytest.raises(AgentEDAError) as caught:
        EMAIL_MASKED.save("project.toml", overwrite=True)

    assert "isn't a regular file" in str(caught.value)
    assert "project.toml" in str(caught.value)
    assert stat.S_ISFIFO(os.stat(tmp_path / "project.toml").st_mode)
    assert stands_alone(caught.value)


@pytest.mark.skipif(not os.path.exists("/dev/null"), reason="needs /dev/null")
def test_save_refuses_a_device_and_leaves_it_alone():
    with pytest.raises(AgentEDAError) as caught:
        EMAIL_MASKED.save("/dev/null", overwrite=True)

    assert "isn't a regular file" in str(caught.value)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_save_through_a_link_to_a_named_pipe_refuses_too(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    (tmp_path / "project.toml").symlink_to(tmp_path / "pipe")

    with pytest.raises(AgentEDAError, match="isn't a regular file"):
        EMAIL_MASKED.save(tmp_path / "project.toml", overwrite=True)


def test_save_replaces_a_file_whole_and_leaves_no_temporary_file(tmp_path):
    path = write(tmp_path / "project.toml", "old content\n")
    inode = path.stat().st_ino

    saved = EMAIL_MASKED.save(path, overwrite=True)

    assert ae.Project.load(saved) == EMAIL_MASKED
    assert os.listdir(tmp_path) == ["project.toml"]
    # A rename puts a new file in place of the old one, which truncating would not.
    assert path.stat().st_ino != inode


def test_an_interrupted_save_keeps_the_old_file(tmp_path, monkeypatch):
    path = write(tmp_path / "project.toml", EMAIL_MASKED_TEXT)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    monkeypatch.setattr(os, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ae.Project(name="other").save(path, overwrite=True)

    assert path.read_text(encoding="utf-8") == EMAIL_MASKED_TEXT
    assert os.listdir(tmp_path) == ["project.toml"]


def test_save_through_a_symlink_replaces_the_file_it_points_to(tmp_path):
    target = write(tmp_path / "real.toml", "old\n")
    (tmp_path / "project.toml").symlink_to(target)

    EMAIL_MASKED.save(tmp_path / "project.toml", overwrite=True)

    assert (tmp_path / "project.toml").is_symlink()
    assert ae.Project.load(target) == EMAIL_MASKED


def test_save_without_overwrite_still_never_replaces_a_file(tmp_path):
    path = write(tmp_path / "project.toml", "keep me\n")

    with pytest.raises(ProjectFileExistsError):
        EMAIL_MASKED.save(path)

    assert path.read_text(encoding="utf-8") == "keep me\n"
    assert os.listdir(tmp_path) == ["project.toml"]


def test_save_refuses_a_project_that_load_would_refuse_for_its_size(tmp_path):
    columns = [f"column_{number:07d}" for number in range(100_000)]
    big = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=columns))

    with pytest.raises(ProjectFileError) as caught:
        big.save(tmp_path / "project.toml", overwrite=True)

    assert caught.value.code == "invalid_project_file"
    assert "1 MiB" in str(caught.value)
    assert "Project.load" in str(caught.value)
    assert os.listdir(tmp_path) == []


def test_a_saved_project_names_its_format_version(tmp_path):
    path = EMAIL_MASKED.save(tmp_path / "project.toml")

    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[:3] == [*HEADER.splitlines(), 'name = "churn"']
    assert ae.Project.load(path) == EMAIL_MASKED


def test_a_file_without_a_format_version_loads_as_version_1(tmp_path):
    path = write(tmp_path / "project.toml", EMAIL_MASKED_TEXT)

    assert ae.Project.load(path).data_policy.sensitive_columns == ("email",)


def test_a_file_of_a_newer_format_is_refused_with_its_version(tmp_path):
    path = write(tmp_path / "project.toml", "format_version = 2\n" + EMAIL_MASKED_TEXT)

    with pytest.raises(ProjectFileError) as caught:
        ae.Project.load(path)

    assert caught.value.code == "invalid_project_file"
    message = str(caught.value)
    assert "format_version = 2" in message
    assert "reads format_version 1" in message
    assert "Upgrade agent-eda" in message
    assert stands_alone(caught.value)


@pytest.mark.parametrize("version", ["0", '"1"', "1.0", "true", "-1"])
def test_a_format_version_that_is_not_a_positive_integer_raises(tmp_path, version):
    path = write(
        tmp_path / "project.toml", f"format_version = {version}\n" + EMAIL_MASKED_TEXT
    )

    with pytest.raises(ProjectFileError, match="format_version"):
        ae.Project.load(path)
