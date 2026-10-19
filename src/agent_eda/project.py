"""The Project: an optional declaration of a data science project's data policy.

Every analysis runs without a Project. The analyses whose results can show raw
values take one as project=, and its data policy says which columns are
sensitive and whether to mask them. The data policy is declared intent, not
access control: the package runs in the caller's own process, which can read
the data directly (ADR 0001).
"""

import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from agent_eda import _project_file
from agent_eda._declarations import Declaration, keyword_signature
from agent_eda._registry import checked_keywords, python_scalar
from agent_eda._run import check_distinct
from agent_eda._text import a_or_an
from agent_eda._wording import DeclareSensitive, Words, mistake
from agent_eda.errors import AgentEDAError, InvalidArgumentError


class DataPolicy(Declaration):
    """Which columns hold sensitive values, and whether results mask them.

    A sensitive column holds values that should not appear in results when
    masking is on, such as names or email addresses. Masking replaces the
    values of sensitive columns that a result would show, such as category
    labels and row IDs taken from a sensitive id_column, with numbered
    placeholders ('<masked 1>', '<masked 2>' and so on) that reveal neither
    content nor length. Counts, shares and statistics stay. Masking stays off,
    even with sensitive columns declared, until mask_sensitive is True here
    or in the call; a call's mask_sensitive argument wins.

    Masking is a guardrail against accidental exposure, such as raw values
    reaching your model provider through a printed result. It is not
    anonymization: counts and statistics stay, and the data itself never
    changes. It is not access control either: the package runs in your own
    process, which can read the data directly. Index labels are never
    masked, so when the index holds sensitive values, pass a non-sensitive
    id_column to the analyses that show row IDs.

    Args:
        sensitive_columns (Sequence[str] | None): Names of the columns that
            hold sensitive values: a list, a tuple, a set or a single name. A
            label that isn't a string can be given as itself, such as 0, or as
            its string form. Names the data lacks are allowed, since a project
            can span several datasets. None or an empty list, the default,
            declares none.
        mask_sensitive (bool): Whether analyses mask the values of the
            sensitive columns when the call's own mask_sensitive is None.
            Defaults to False. True needs sensitive_columns.

    Examples:
        >>> import agent_eda as ae
        >>> ae.DataPolicy(sensitive_columns=["email", "name"], mask_sensitive=True)
        DataPolicy(sensitive_columns=('email', 'name'), mask_sensitive=True)
    """

    # The docstring is long for a JSON Schema, which tool frameworks show.
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Which columns hold sensitive values, and whether "
            "results mask them with numbered placeholders. Masking is a "
            "guardrail against accidental exposure, not anonymization or access "
            "control."
        }
    )

    _example = DeclareSensitive(["email"], of="policy").render("python")
    _field_types: ClassVar[dict[str, object]] = {
        "sensitive_columns": Sequence[str] | None,
        "mask_sensitive": bool,
    }

    sensitive_columns: tuple[str, ...] = Field(
        default=(),
        description="Names of the columns that hold sensitive values; empty, the "
        "default, declares none.",
    )
    mask_sensitive: bool = Field(
        default=False,
        description="Whether analyses mask the values of the sensitive columns "
        "when the call's mask_sensitive is None. Defaults to false; true needs "
        "sensitive_columns.",
    )

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            sensitive_columns: Sequence[str] | None = (),
            mask_sensitive: bool = False,
        ) -> None: ...

    @field_validator("sensitive_columns", mode="before")
    @classmethod
    def _names(cls, value: Any) -> Any:
        """Column names as the analyses' columns argument takes them."""
        if value is None:
            return ()
        if isinstance(value, pd.Series | pd.DataFrame):
            # Its values would become names, and messages would show them.
            names = (
                "[series.name] for the column it came from"
                if isinstance(value, pd.Series)
                else "list(df.columns) for its columns"
            )
            raise InvalidArgumentError(
                f"DataPolicy got a pandas {type(value).__name__} for "
                "sensitive_columns, but sensitive_columns takes column names, not "
                f"data. Pass the names, such as sensitive_columns=['email'], or "
                f"{names}."
            )
        if isinstance(value, Mapping):
            raise InvalidArgumentError(
                f"DataPolicy got {a_or_an(type(value).__name__)} for sensitive_columns, but "
                "sensitive_columns must be a list of column names, such as "
                f"{cls._example}."
            )
        if isinstance(value, str | bytes) or not isinstance(value, Iterable):
            items = [value]
        else:
            items = value.tolist() if hasattr(value, "tolist") else list(value)
        names = [_name(item) for item in items]
        if isinstance(value, set | frozenset):
            names = sorted(names, key=str)
        # Validation rejects anything that isn't a name, naming its position.
        texts = [name for name in names if isinstance(name, str)]
        check_distinct("DataPolicy", "sensitive_columns", texts, "column")
        return tuple(names)

    @model_validator(mode="after")
    def _something_to_mask(self) -> Self:
        if self.mask_sensitive and not self.sensitive_columns:
            # An MCP caller's data policy comes from a project file, so its
            # message shows the TOML.
            raise mistake(
                InvalidArgumentError,
                "DataPolicy got ",
                Words("mask_sensitive=True", "mask_sensitive = true"),
                " but no sensitive_columns, so there would be nothing to mask. "
                "Name the sensitive columns, such as ",
                DeclareSensitive(["email"], of="policy"),
                ".",
            )
        return self


def _name(item: object) -> object:
    """A column label as the string form results show, if it is a label at all."""
    item = python_scalar(item)
    if isinstance(item, Hashable):
        return str(item)
    return item  # validation rejects it, naming its type


class Project(Declaration):
    """A data science project's declarations, given to the analyses that use them.

    A Project is optional, and every analysis runs without one. It holds a
    name and the data policy, which declares the sensitive columns and whether
    to mask them. Pass it as project= to the analyses whose results can show
    raw values; ae.list_analyses() marks them [supports masking]. Build it
    once and pass the same Project to every call. To use it in later scripts
    too, save it once with project.save('project.toml') and load it in each
    of them with ae.Project.load('project.toml'), whose help shows the file.

    The data policy is declared intent that the package checks to catch
    honest mistakes. Masking is a guardrail against accidental exposure, not
    anonymization and not access control: see help(ae.DataPolicy).

    Args:
        name (str | None): The project's name, such as 'churn'. None, the
            default, leaves it unnamed.
        data_policy (DataPolicy | Mapping[str, Any] | None): Which columns are
            sensitive and whether to mask them: a DataPolicy, a dict of its
            fields, or None. Defaults to DataPolicy(), which declares no
            sensitive column and leaves masking off.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> project = ae.Project(
        ...     name="support tickets",
        ...     data_policy=ae.DataPolicy(
        ...         sensitive_columns=["customer"], mask_sensitive=True
        ...     ),
        ... )
        >>> df = pd.DataFrame(
        ...     {
        ...         "customer": ["Ann Lee", "Bo Chu", "Ann Lee"],
        ...         "plan": ["pro", "pro", "team"],
        ...     }
        ... )
        >>> res = ae.summarize_categorical(df, project=project)
        >>> [(level.label, level.count) for level in res.columns["customer"].levels]
        [('<masked 1>', 2), ('<masked 2>', 1)]
        >>> res.masking.columns
        ['customer']
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "A data science project's declarations: its name and "
            "its data policy, which names the sensitive columns and whether to "
            "mask them."
        }
    )

    _example = (
        "ae.Project(name='churn', "
        "data_policy=ae.DataPolicy(sensitive_columns=['email']))"
    )
    _field_types: ClassVar[dict[str, object]] = {
        "name": str | None,
        "data_policy": DataPolicy | Mapping[str, Any] | None,
    }

    name: str | None = Field(
        default=None,
        description="The project's name; null, the default, leaves it unnamed.",
    )
    data_policy: DataPolicy = Field(
        default_factory=DataPolicy,
        description="Which columns are sensitive and whether to mask them. "
        "Defaults to a policy that declares no sensitive column.",
    )

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            name: str | None = None,
            data_policy: DataPolicy | Mapping[str, Any] | None = None,
        ) -> None: ...

    @classmethod
    def _unknown_field(cls, name: str) -> str:
        if name in DataPolicy.model_fields:
            return (
                f"Project has no field {name!r}. It is a field of the data policy: "
                f"ae.Project(data_policy=ae.DataPolicy({name}=...)). "
                f"Allowed: {', '.join(cls.model_fields)}."
            )
        return super()._unknown_field(name)

    @classmethod
    @checked_keywords
    def load(cls, path: str | os.PathLike[str]) -> Self:
        """Load a project from a TOML file, such as one that project.save() wrote.

        An agent that runs each script in a new process saves its project once
        and loads it at the start of each later script. The file holds the
        project's name at the top level and its data policy in a [data_policy]
        table, as project.save() writes it:

            format_version = 1
            name = "churn"

            [data_policy]
            sensitive_columns = ["email", "customer name"]
            mask_sensitive = true

        Leave name out for an unnamed project. The [data_policy] table and both
        of its keys are required, with sensitive_columns = [] for none, so that
        a file cut short raises instead of loading with masking off. A key
        that isn't one of these raises and names the closest one, so that a
        typo such as mask_sensitve can't leave masking off quietly either.
        format_version is the layout version of the file: project.save() writes
        1, and a file without the key loads as version 1. A file of a higher
        version, which a newer agent-eda wrote, raises and names it.

        Args:
            path (str | os.PathLike[str]): The file to read, such as
                'project.toml'. A leading ~ means the home directory.

        Returns:
            Project: The project the file declares. Loading the file that
            project.save() wrote gives a project equal to the one saved.

        Raises:
            ProjectFileNotFoundError: No file is at path (code
                project_file_not_found). It is a FileNotFoundError too.
            ProjectFileError: The file isn't TOML, and the message gives the
                line at fault; it lacks the [data_policy] table or one of its
                keys; or it holds a value a field can't take (code
                invalid_project_file). Or it can't be read: it isn't a regular
                file, such as a named pipe or a device; the operating system
                refuses it, such as without read permission; or it holds more
                than 1 MiB (code invalid_project_file). Or its format_version
                is newer than this agent-eda reads, and the message names the
                version (code invalid_project_file). Or it holds a key that
                isn't a field, and the message names the closest key (code
                unknown_project_key).
            InvalidArgumentError: path isn't a str or a path, is empty, or
                names a directory (code invalid_argument).

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> import agent_eda as ae
            >>> with tempfile.TemporaryDirectory() as folder:
            ...     path = Path(folder) / "project.toml"
            ...     _ = path.write_text(
            ...         'name = "churn"\\n'
            ...         "[data_policy]\\n"
            ...         'sensitive_columns = ["email"]\\n'
            ...         "mask_sensitive = true\\n"
            ...     )
            ...     project = ae.Project.load(path)
            >>> project
            Project(name='churn', data_policy=DataPolicy(sensitive_columns=('email',), mask_sensitive=True))
        """
        fields = _project_file.read(
            path,
            keys=list(cls.model_fields),
            table="data_policy",
            table_keys=list(DataPolicy.model_fields),
        )
        try:
            return cls.model_validate(fields)
        except AgentEDAError as error:
            raise _project_file.invalid_project_error(path, error) from None

    @checked_keywords
    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Save the project to a TOML file that Project.load() reads.

        An agent that runs each script in a new process saves its project once
        and loads it with ae.Project.load(path) at the start of each later
        script, instead of building it again. The file holds format_version = 1
        and the name at the top level, the name left out when there is none,
        and the data policy in a [data_policy] table. The save is atomic: it
        writes a temporary file next to the target and renames it into place,
        so an interrupted save leaves the file that was there. A new file is
        readable by its owner only, and a replaced file keeps its permissions.
        It is laid out
        the same way every time, so equal
        projects give identical files on the same Python version. Column names
        keep every character: quotes, backslashes and characters that print as
        nothing are escaped.

        Args:
            path (str | os.PathLike[str]): Where to write the file, such as
                'project.toml', in a directory that exists. A leading ~ means
                the home directory.
            overwrite (bool): Whether to replace a file that is already at
                path. Defaults to False, which raises instead, so that a saved
                project is never replaced by accident.

        Returns:
            Path: The file's absolute path, to pass to Project.load().

        Raises:
            ProjectFileExistsError: A file is already at path and overwrite is
                False (code project_file_exists). It is a FileExistsError too.
            ProjectFileError: The name or a column name holds a lone
                surrogate, which TOML can't hold, or the file would hold more
                than the 1 MiB that Project.load reads (code
                invalid_project_file).
            InvalidArgumentError: path isn't a str or a path, is empty, names
                a directory or a file that isn't a regular one, such as a
                named pipe, or lies in a directory that doesn't exist, or
                overwrite isn't a bool (code invalid_argument).
            OSError: The file can't be written, such as without permission.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> import agent_eda as ae
            >>> project = ae.Project(
            ...     name="churn",
            ...     data_policy=ae.DataPolicy(
            ...         sensitive_columns=["email"], mask_sensitive=True
            ...     ),
            ... )
            >>> with tempfile.TemporaryDirectory() as folder:
            ...     path = project.save(Path(folder) / "project.toml")
            ...     print(path.read_text(encoding="utf-8"))
            ...     loaded = ae.Project.load(path)
            # An agent-eda Project. Load it with ae.Project.load(path).
            format_version = 1
            name = "churn"
            <BLANKLINE>
            [data_policy]
            sensitive_columns = [
                "email",
            ]
            mask_sensitive = true
            <BLANKLINE>
            >>> loaded == project
            True
        """
        return _project_file.write(
            path, self.model_dump(mode="json"), overwrite=overwrite
        )

    @field_validator("data_policy", mode="before")
    @classmethod
    def _default_policy(cls, value: Any) -> Any:
        """None means the default policy, as the signature shows."""
        return DataPolicy() if value is None else value

    @model_validator(mode="before")
    @classmethod
    def _not_a_policy(cls, value: Any) -> Any:
        """A data policy where a project belongs, as in project=policy, raises."""
        if isinstance(value, DataPolicy):
            raise InvalidArgumentError(
                "project takes a Project, but got a DataPolicy. Put the policy in a "
                "project: project=ae.Project(data_policy=policy)."
            )
        return value


# help() and editors show the calls as they read, not as pydantic builds it.
DataPolicy.__signature__ = keyword_signature(
    DataPolicy, {"sensitive_columns": (), "mask_sensitive": False}
)
Project.__signature__ = keyword_signature(Project, {"name": None, "data_policy": None})
