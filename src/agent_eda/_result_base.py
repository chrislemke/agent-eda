"""The base of every result: the record, its text, tables and caps, and the shared parts."""

import ast
import inspect
import itertools
import json
import re
import textwrap
import typing
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal, Self

import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from agent_eda import _text
from agent_eda._annotations import TypeName
from agent_eda._written_form import record_integers, row_id_json
from agent_eda.errors import (
    InvalidArgumentError,
    ResultTableUnavailableError,
)

if typing.TYPE_CHECKING:
    from agent_eda._envelope import Settings

Status = Literal[
    "completed",
    "partial",
    "insufficient_data",
    "not_applicable",
    "missing_prerequisites",
    "budget_exceeded",
    "blocked_by_policy",
    "failed",
]

Assessment = Literal["descriptive", "suspected", "violation", "passed", "unknown"]

# The order the printed text lists findings in, so that a cap on the findings it
# shows never hides a violation behind suspected findings.
_TEXT_ORDER: tuple[Assessment, ...] = (
    "violation",
    "suspected",
    "descriptive",
    "passed",
    "unknown",
)

# What a column's values mean, as opposed to the dtype pandas stores them in.
# infer_column_semantics ranks these for each column, and a declared schema
# names one per column.
LogicalType = Literal[
    "boolean",
    "integer",
    "float",
    "categorical",
    "text",
    "datetime",
    "timedelta",
]

# What infer_column_semantics flags in a column that needs attention.
ColumnFlag = Literal[
    "identifier_like",
    "constant",
    "all_missing",
    "numbers_as_text",
    "dates_as_text",
    "non_iso_date_text",
    "boolean_like",
    "mixed_value_types",
]

# The package-wide list of status reason codes: why a run, or one column of it,
# ended with a status other than completed. Codes are public API: renaming one
# follows the rule for renamed names. Each analysis's Statuses: section names
# the codes it returns, and describe_analysis shows them with these meanings.
StatusReasonCode = Literal[
    # The data has no rows.
    "no_rows",
    # No column is left to work on: the data has none, the call's column
    # selection or the analysis's column rules leave none, or too few can be
    # paired. The message says which.
    "no_columns",
    # No value is left to work on in a column, the target or the pairs: every
    # value is missing, infinite or outside the declared labels, or no pair has
    # a value. The message says which.
    "no_values",
    # The data has one row, which every column identifies, so a key search has
    # nothing to find.
    "one_row",
    # The values, such as lists or dicts, can't be hashed, so they can't be
    # counted as levels.
    "unhashable_values",
    # No logical type fits the values: they are of other types, such as lists
    # or intervals, and aren't category labels repeated over rows.
    "unsupported_values",
    # Some columns didn't complete, so the result is partial; each column's
    # status_reason says why.
    "incomplete_columns",
    # The paired columns make more pairs than the max_pairs limit allows, so no
    # pair was computed.
    "too_many_pairs",
    # The complete key search covers more column sets than the max_combinations
    # limit allows, so it stopped early.
    "too_many_combinations",
    # An anticipated numerical failure stopped the analysis. The status failed
    # is the only one that carries it, and no analysis returns that status yet.
    "numerical_failure",
]

# Why a value was left out of the counts behind a column's result. One spelling
# per reason, in every analysis.
ExclusionReason = Literal[
    # The value is missing.
    "missing",
    # The value is infinite (inf or -inf).
    "infinite",
    # A target label isn't in the declared expected labels.
    "unexpected_label",
    # The values can't be hashed, such as lists or dicts, so they can't be
    # compared or counted as levels.
    "unhashable_values",
    # The column has more distinct values than the max_categories limit, so
    # the column was left out.
    "too_many_categories",
]

# The package-wide list of limitation codes. The first sentence of each comment
# is the code's short meaning, which the README's Reference prints. Codes are
# public API: renaming one follows the rule for renamed names.
LimitationCode = Literal[
    # Values can't be hashed, so counts of distinct values are null. Lists and
    # dicts are such values.
    "unhashable_values",
    # A key search left out float columns that hold fractions. They are
    # measurements rather than IDs, and the default columns don't select them.
    "non_integral_float_columns",
    # A statistic needs more values than the column has, so it is null. A
    # standard deviation of one value is an example.
    "too_few_values",
    # Every value is equal, so a statistic such as skewness is null.
    "zero_variance",
    # Numbers beyond ±2^53 were read as float64, which rounds them. Integers and
    # decimals are such numbers.
    "float_precision",
    # Booleans were read as numbers: 0 for False and 1 for True.
    "boolean_as_numeric",
    # A column stores NaN as a value, which every analysis counts as missing,
    # unlike isna().
    "nan_counted_as_missing",
    # A categorical column has one level per row of a pair, so its value is
    # null.
    "one_row_per_level",
    # A computation overflowed or failed, so the values it affects are null.
    # Division by zero, an invalid value and precision loss are examples; the
    # fields are empty when no value could be named.
    "numerical_warning",
    # Index labels or id_column values repeat, so a row ID can match several
    # rows.
    "ambiguous_row_ids",
    # No sentinel values were declared, so counts are null or candidates stay
    # suspected. Stand-ins such as -999 don't count as missing.
    "sentinel_values_not_declared",
    # A ranking stopped at its work limit before it was proven. An uncounted
    # set could outrank a listed one.
    "ranking_not_proven",
    # No expected labels were declared, so labels outside them weren't checked.
    # The fields that list those labels are null.
    "expected_labels_not_declared",
    # No rare_share was declared, so rare classes weren't checked. The fields
    # that list them are null.
    "rare_share_not_declared",
    # A declared column isn't in the data, so its dtype and counts are null.
    "column_not_in_data",
    # No schema was declared, so no column was compared with a declared type.
    # The fields that hold the comparison are null.
    "schema_not_declared",
    # nullable wasn't declared for a column, so its missing values weren't
    # checked and its nullable field is null.
    "nullable_not_declared",
    # A column has no inferred logical type, so the field is null. No value is
    # left after the missing ones, or none fits.
    "type_not_inferred",
    # A field belongs to the other task, so it is null. The tasks are
    # classification and regression.
    "not_applicable_to_task",
]


def entry_path(name: str) -> str:
    """The result path of a column's entry in a result's `columns`, such as
    "columns['age']"."""
    return f"columns[{name!r}]"


# The start of a field path that points at a table entry of a column, as a
# position, columns[2], or as a label, columns['age'].
_ENTRY_PATH = re.compile(
    r"columns\[(?:(?P<position>\d+)|(?P<label>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"))\]"
)


def _left_out_lines(
    names: Sequence[str], kinds: Mapping[str, str] | None, *, wanted: str, boolean: str
) -> list[str]:
    """Lines saying which columns the default left out and why.

    `wanted` names what the analysis reads, such as "numeric"; `boolean` says
    what happens to booleans, which some analyses read only when named. The
    columns are grouped by kind, in the order the kinds first appear. A result
    rebuilt from its record has no kinds and says only what the analysis reads.
    """
    if not names:
        return []
    if kinds is None:
        return [f"Left out, not {wanted}: {_text.join_labels(names)}."]
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(kinds.get(name, "other"), []).append(name)
    lines = []
    for kind, group in groups.items():
        why = boolean if kind == "boolean" else f"not {wanted} ({kind})"
        lines.append(f"Left out, {why}: {_text.join_labels(group)}.")
    return lines


class _Cap:
    """Marks a list or dict field that grows with the data.

    The record (to_dict and to_json) keeps the first `limit` entries and adds
    `<field>_total` with the full count. The result object keeps every entry.
    With `each_list`, a dict field stays whole and every list or dict inside it
    is capped instead, with `<key>_total` next to it, except the keys named in
    `whole`. With `total`, the field holds only the first entries even on the
    result object, and `<field>_total` takes the full count from the field that
    `total` names. A list whose entries have a natural key, such as a pair's two
    columns, takes `key`, which gives an entry's key as a tuple of strings:
    field paths then name its entries by key, as `pairs['a', 'b']`, and a
    capped record keeps the entries that findings and limitations point at.
    """

    def __init__(
        self,
        limit: int,
        *,
        each_list: bool = False,
        total: str | None = None,
        whole: tuple[str, ...] = (),
        key: Callable[[Any], tuple[str, ...]] | None = None,
    ) -> None:
        self.limit = limit
        self.each_list = each_list
        self.total = total
        self.whole = whole
        self.key = key

    def derive(self, **changes: Any) -> "_Cap":
        """A marker with this one's limit and the given options changed."""
        options: dict[str, Any] = {
            "each_list": self.each_list,
            "total": self.total,
            "whole": self.whole,
            "key": self.key,
        }
        return _Cap(self.limit, **{**options, **changes})


# The caps the spec sets for every list in the record that grows with the data.
# These three markers hold the only copy of each number: the other markers are
# derived from them, the analyses that cut their lists ask the result class
# (ResultModel.limit_of), and the descriptions and docstrings get the numbers
# from the markers (fill_caps).
_ROW_IDS = _Cap(10)
_EXAMPLE_VALUES = _Cap(5)
_ENTRIES = _Cap(100)
# For parameters: list and mapping arguments, but not data's shape or the
# project, which is recorded whole.
_LISTS_INSIDE = _ENTRIES.derive(each_list=True, whole=("data", "project"))

# What a docstring or description writes for each cap, until fill_caps puts the
# marker's number in: "{row_ids_kept}" for the row IDs of a list, and so on.
_CAP_NUMBERS = {
    "{row_ids_kept}": _ROW_IDS,
    "{examples_kept}": _EXAMPLE_VALUES,
    "{entries_kept}": _ENTRIES,
}


def fill_caps(text: str, *, field: str | None = None, cap: _Cap | None = None) -> str:
    """The text with each cap placeholder replaced by its marker's number.

    `{row_ids_kept}`, `{examples_kept}` and `{entries_kept}` become the three
    caps of the record. For the description of a capped field, which passes its
    own name and marker, `{limit}` becomes the marker's number and `{cap}` its
    clause: "The record keeps the first 100 and adds columns_total".
    """
    for placeholder, marker in _CAP_NUMBERS.items():
        text = text.replace(placeholder, str(marker.limit))
    if field is not None and cap is not None:
        clause = f"The record keeps the first {cap.limit} and adds {field}_total"
        text = text.replace("{cap}", clause).replace("{limit}", str(cap.limit))
    return text


# Said in every field description whose record can hold such an integer.
_BIG_INTEGERS = "The record writes an integer beyond ±2^53 as a string."

# Said in the output schema, for every field of the record at once.
_RECORD_INTEGERS = (
    "Every integer beyond ±2^53 (9,007,199,254,740,991), in row IDs, labels, "
    "values and parameters, is a string."
)

_RECORD = "agent_eda_record"
# In the serialization context: which entries of the result's capped fields the
# findings and limitations point at, by field name.
_POINTED = "agent_eda_pointed"


# Lone surrogates, which no UTF-8 output can hold.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _record_json(record: object) -> str:
    """A record as strict JSON that encodes as UTF-8.

    A lone surrogate, such as in a column label decoded with
    errors='surrogateescape', is written as its \\uXXXX escape, which JSON
    allows and json.loads reads back as the same text.
    """
    text = json.dumps(
        record, allow_nan=False, ensure_ascii=False, separators=(",", ":")
    )
    return _LONE_SURROGATE.sub(lambda found: f"\\u{ord(found[0]):04x}", text)


def _hint(name: object, names: Sequence[str]) -> str:
    """ " Did you mean 'x'?" for the closest of the names, or nothing."""
    closest = _text.closest_names(name, names)[:1] if isinstance(name, str) else []
    return f" Did you mean {closest[0]!r}?" if closest else ""


def _rich_text(text: str) -> Iterator[Any]:
    """The text as one Rich renderable, which keeps square brackets as they are."""
    from rich.text import Text  # rich is optional: only Rich calls this

    yield Text(text)


class _ColumnKeyError(KeyError):
    """A KeyError whose text is the message, not the message's repr."""

    def __str__(self) -> str:
        return str(self.args[0])


# A message lists this many column names, then counts the rest.
_NAMES_LISTED = 10


def _no_column(name: object, names: Sequence[str]) -> _ColumnKeyError:
    """The error for a column name the result doesn't hold: the closest names
    first, then the columns it holds."""
    closest = _text.closest_names(name, names)[:3] if isinstance(name, str) else []
    hint = (
        " Did you mean " + _text.join_alternatives([repr(c) for c in closest]) + "?"
        if closest
        else ""
    )
    shown = ", ".join(repr(column) for column in names[:_NAMES_LISTED])
    more = len(names) - _NAMES_LISTED
    rest = f" and {more} more" if more > 0 else ""
    held = (
        f"The result holds {shown}{rest}." if names else "The result holds no columns."
    )
    return _ColumnKeyError(f"no column {name!r} in this result.{hint} {held}")


class _ColumnsByName(dict[str, Any]):
    """A mapping from column name to entry whose KeyError suggests the closest
    names."""

    def __missing__(self, key: object) -> Any:
        raise _no_column(key, list(self))


class _ColumnList(list[Any]):
    """A list of column entries that also finds one by column name, with the same
    KeyError as _ColumnsByName for a name that no entry has."""

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            return super().__getitem__(key)
        found = [i for i, entry in enumerate(self) if entry.column == key]
        if not found:
            raise _no_column(key, [entry.column for entry in self])
        if len(found) > 1:
            where = ", ".join(map(str, found[:-1])) + f" and {found[-1]}"
            raise _ColumnKeyError(
                f"{key!r} labels {len(found)} columns, at positions {where}. "
                "Index the list by position."
            )
        return super().__getitem__(found[0])


def _by_name(columns: dict[str, Any]) -> dict[str, Any]:
    return _ColumnsByName(columns)


def _list_by_name(columns: list[Any]) -> list[Any]:
    return _ColumnList(columns)


# Marks a field keyed by column name, whose lookup suggests close names.
_BY_NAME = AfterValidator(_by_name)


def _tables_unavailable(result: "AnalysisResult") -> ResultTableUnavailableError:
    """The error for a result that was rebuilt from its record and so lost the
    tables that only the analysis call keeps."""
    return ResultTableUnavailableError(
        f"{type(result).__name__}.to_frame() needs the full tables that only a "
        "call of the analysis keeps, but this result was rebuilt from its "
        "record, such as by model_validate(res.model_dump()), which doesn't hold "
        f"them. Call ae.{result.analysis}(...) again, or copy the result with "
        "copy.deepcopy(res) or res.model_copy(), which keep the tables."
    )


# A row ID: an index label, or a value of the column id_column names. The result
# object holds it as pandas gives it, so df.loc[row_ids] finds the rows; the
# record converts it with row_id_json.
RowId = Annotated[
    Any,
    PlainSerializer(row_id_json, return_type=Any, when_used="json"),
    TypeName("RowId"),
]


def _head(values: list[Any] | dict[Any, Any], limit: int) -> list[Any] | dict[Any, Any]:
    """The first `limit` entries of a list or dict."""
    if isinstance(values, dict):
        return dict(itertools.islice(values.items(), limit))
    return values[:limit]


def _capped(
    values: dict[str, Any], limit: int, whole: tuple[str, ...], *, top: bool = False
) -> dict[str, Any]:
    """Keep the first `limit` entries of each list or dict among `values`, except
    the keys named in `whole`, and add its total as `<key>_total` beside it.

    The lists and dicts inside the kept entries are capped the same way, down
    to any depth, but get their total only when entries were dropped. A
    `<key>_total` that another key already holds gets underscores added until
    it is free, so no entry is overwritten.
    """
    capped: dict[str, Any] = {}
    for name, value in values.items():
        if name in whole or not isinstance(value, list | dict):
            capped[name] = value
            continue
        if isinstance(value, dict):
            capped[name] = _capped(
                dict(itertools.islice(value.items(), limit)), limit, ()
            )
        else:
            capped[name] = value[:limit]
        if top or len(value) > limit:
            total = f"{name}_total"
            while total in values:
                total += "_"
            capped[total] = len(value)
    return capped


def _entry_names(values: list[Any] | dict[Any, Any], cap: _Cap) -> list[Any]:
    """What a field path calls each entry: its key in a mapping, its key fields
    as a tuple in a keyed list, and its position in any other list."""
    if isinstance(values, dict):
        return list(values)
    if cap.key is not None:
        return [cap.key(entry) for entry in values]
    return list(range(len(values)))


def _kept_indexes(
    values: list[Any] | dict[Any, Any], cap: _Cap, pointed: set[Any]
) -> list[int] | None:
    """The positions of the entries a capped record keeps; None for all of them.

    The record keeps every entry that a finding or limitation points at, then
    the first of the others, `cap.limit` entries in all, in their own order.
    """
    if len(values) <= cap.limit:
        return None
    names = _entry_names(values, cap)
    marked = [i for i, name in enumerate(names) if name in pointed]
    if not marked:
        return list(range(cap.limit))
    keep = set(marked)
    others = (i for i in range(len(names)) if i not in keep)
    keep.update(itertools.islice(others, max(cap.limit - len(marked), 0)))
    return sorted(keep)


def _pick(
    values: list[Any] | dict[Any, Any], indexes: list[int]
) -> list[Any] | dict[Any, Any]:
    """The entries of a list or dict at these positions."""
    if isinstance(values, dict):
        items = list(values.items())
        return {items[i][0]: items[i][1] for i in indexes}
    return [values[i] for i in indexes]


def _subscript(path: str) -> tuple[str, Any] | None:
    """The field and the entry a field path starts with, such as ('columns',
    'age') for "columns['age'].min"; None when it names no entry."""
    try:
        node = ast.parse(path, mode="eval").body
        while True:
            if isinstance(node, ast.Attribute):
                node = node.value
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if isinstance(node.slice, ast.Slice):
                    return None
                return node.value.id, ast.literal_eval(node.slice)
            elif isinstance(node, ast.Subscript):
                node = node.value
            else:
                return None
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _renumbered(path: str, moves: dict[str, dict[int, int]]) -> str:
    """A field path with its list position changed to the entry's position in
    the record; `moves` maps each list field's object positions to those."""
    found = _subscript(path)
    if found is None or found[0] not in moves or not isinstance(found[1], int):
        return path
    name, position = found
    if position not in moves[name]:
        return path
    return f"{name}[{moves[name][position]}]{path[path.index(']') + 1 :]}"


def _models_in(
    annotation: object, found: dict[type[BaseModel], None]
) -> dict[type[BaseModel], None]:
    """Every pydantic model a field annotation reaches, in the order met."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation not in found:
            found[annotation] = None
            for field in annotation.model_fields.values():
                _models_in(field.annotation, found)
        return found
    for argument in typing.get_args(annotation):
        _models_in(argument, found)
    return found


def _is_optional(annotation: object) -> bool:
    return type(None) in typing.get_args(annotation)


# What the full table has besides the statistics the text shows, for
# summarize_numeric and a regression target.
_MORE_STATISTICS = (
    "the quantiles, trimmed_mean, mad, mad_scaled, skewness and excess_kurtosis"
)

# Nullable integers keep their type in to_frame() and null floats become NaN;
# other fields use pandas' default.
_FRAME_DTYPES: dict[Any, str] = {
    int: "int64",
    int | None: "Int64",
    float: "float64",
    float | None: "float64",
    bool: "bool",
}


def _reject_table(table: object, tables: Sequence[str]) -> None:
    """Raise unless `table` names one of a result's tables; the first is the default.

    Results with more than one table take the name in to_frame(table=...).
    """
    if isinstance(table, str) and table in tables:
        return
    shown = (
        repr(table) if isinstance(table, str) else _text.a_or_an(type(table).__name__)
    )
    closest = _text.closest_names(table, tables)[:1] if isinstance(table, str) else []
    hint = f" Did you mean {closest[0]!r}?" if closest else ""
    allowed = _text.join_alternatives([repr(name) for name in tables])
    allowed = allowed if len(tables) == 1 else f"one of {allowed}"
    raise InvalidArgumentError(
        f"to_frame got {shown} for table, but table must be {allowed}.{hint} "
        f"res.to_frame() returns the {tables[0]} table."
    )


def _entries_frame(
    entries: Sequence["ResultModel"], model: type["ResultModel"]
) -> pd.DataFrame:
    """One row per entry and one column per field of the entry model."""
    columns = {
        name: pd.Series(
            [getattr(entry, name) for entry in entries],
            dtype=_FRAME_DTYPES.get(field.annotation),
        )
        for name, field in model.model_fields.items()
    }
    return pd.DataFrame(columns, columns=list(model.model_fields))


# Where a result's help text wraps.
_HELP_WIDTH = 100


def _fields_doc(cls: Any) -> str:
    """The class docstring with one entry per field: its name and description."""
    lines = [cls.__plain_doc__.rstrip(), ""]
    if cls.__help_note__:
        lines += [*textwrap.wrap(cls.__help_note__, width=_HELP_WIDTH), ""]
    lines.append("Fields:")
    for name, field in cls.model_fields.items():
        lines += textwrap.wrap(
            f"{name}: {field.description or ''}".rstrip(),
            width=_HELP_WIDTH,
            initial_indent="    ",
            subsequent_indent="        ",
        )
    if "analysis" in cls.model_fields and cls.__name__ != "AnalysisResult":
        analysis = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__.removesuffix("Result"))
        lines += [
            "",
            f"ae.describe_analysis('{analysis.lower()}') shows how to call the "
            "analysis, how to read these fields and what each finding code means.",
        ]
    return "\n".join(lines)


# Members for the authors of an analysis, not for the readers of its result, so
# that dir() and help() of a result leave them out.
_FOR_AUTHORS = frozenset({"for_call", "limit_of", "settings"})


class _ResultMeta(type(BaseModel)):
    """The metaclass of the result classes. It writes each class's docstring with
    its fields' descriptions, and keeps pydantic's own members out of dir(cls),
    so that help(res) shows what a result holds instead of 29,000 characters of
    pydantic internals."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ):
        cls: Any = super().__new__(mcs, name, bases, namespace, **kwargs)
        if "analysis" in cls.model_fields:
            # An analysis result: help() shows the fields. The record's JSON
            # Schema keeps the plain docstring (record_schema puts it back).
            cls.__plain_doc__ = cls.__dict__.get("__doc__") or ""
            cls.__doc__ = _fields_doc(cls)
        if cls.model_fields:
            # help() prints the constructor's signature, which pydantic builds
            # from the annotations, with objects such as <_Cap object at 0x...>.
            cls.__signature__ = inspect.Signature(
                [
                    inspect.Parameter(field, inspect.Parameter.KEYWORD_ONLY)
                    for field in cls.model_fields
                ]
            )
        return cls

    def __dir__(cls) -> list[str]:
        owners = {
            name: next(c for c in cls.__mro__ if name in vars(c))
            for name in super().__dir__()
            if any(name in vars(c) for c in cls.__mro__)
        }
        return [
            name
            for name, owner in owners.items()
            if owner.__module__.startswith("agent_eda")
            and not name.startswith(("_", "model_"))
            and name not in _FOR_AUTHORS
        ]


class ResultModel(BaseModel, metaclass=_ResultMeta):
    """Base for every record inside an analysis result.

    Frozen, with unknown fields forbidden and no NaN or infinity.
    """

    # An analysis result's docstring as written, before the fields were added to
    # the class docstring that help() shows.
    __plain_doc__: ClassVar[str] = ""
    # A note for help() only, such as how to look an entry up in Python.
    __help_note__: ClassVar[str] = ""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Give each capped field's description its cap clause from its marker."""
        super().__pydantic_init_subclass__(**kwargs)
        changed = False
        for name, field in cls.model_fields.items():
            cap = next((m for m in field.metadata if isinstance(m, _Cap)), None)
            if cap is None or not field.description:
                continue
            filled = fill_caps(field.description, field=name, cap=cap)
            if filled != field.description:
                field.description = filled
                changed = True
        if changed and cls.__pydantic_complete__:
            # pydantic builds a field's JSON Schema from the description it had
            # when the class was built.
            cls.model_rebuild(force=True)

    @classmethod
    def limit_of(cls, field: str) -> int:
        """The most entries the record keeps of a capped list or mapping field.

        Raises ValueError for a field that isn't capped, naming the ones that are.
        """
        capped = {
            name: marker
            for name, info in cls.model_fields.items()
            for marker in info.metadata
            if isinstance(marker, _Cap)
        }
        if field not in capped:
            raise ValueError(
                f"{cls.__name__}.{field} is not a capped field. The capped fields "
                f"are {list(capped)}."
            )
        return capped[field].limit

    @model_serializer(mode="wrap")
    def _cap_growing_fields(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        if not (info.context and info.context.get(_RECORD)):
            return handler(self)
        caps = {
            name: marker
            for name, field in type(self).model_fields.items()
            for marker in field.metadata
            if isinstance(marker, _Cap)
        }
        if not caps:
            return handler(self)
        limits = {name: cap.limit for name, cap in caps.items() if not cap.each_list}
        # A list that is null, such as a check that didn't run, has no length.
        lengths = {
            name: None if getattr(self, name) is None else len(getattr(self, name))
            for name in limits
        }
        # A cap with `total` names the field that holds the full count, since
        # the result object keeps only the first entries of that list.
        totals = {
            name: getattr(self, total) if (total := caps[name].total) else length
            for name, length in lengths.items()
        }
        # Serialize only the entries the record keeps, which saves converting
        # what it would drop, such as the row IDs of 500,000 duplicate groups.
        # Findings and limitations name entries of the result's own fields; the
        # record keeps those.
        pointed = (
            info.context.get(_POINTED, {}) if isinstance(self, AnalysisResult) else {}
        )
        heads: dict[str, Any] = {}
        for name, limit in limits.items():
            length = lengths[name]
            if length is None or length <= limit:
                continue
            values = getattr(self, name)
            if name in pointed and caps[name].total is None:
                kept = _kept_indexes(values, caps[name], pointed[name])
                heads[name] = values if kept is None else _pick(values, kept)
            else:
                heads[name] = _head(values, limit)
        data = handler(self.model_copy(update=heads) if heads else self)
        record: dict[str, Any] = {}
        for name, value in data.items():
            record[name] = value
            if name in totals:
                record[f"{name}_total"] = totals[name]
        for name, cap in caps.items():
            inner = record.get(name)
            if cap.each_list and isinstance(inner, dict):
                record[name] = _capped(inner, cap.limit, cap.whole, top=True)
        return record

    # pydantic's deprecated methods give records that differ from the result's own
    # (no totals for capped lists, a JSON Schema that reads like an inferred
    # schema), so each points to the method to use. They sit last in the class
    # body, since a method named dict would otherwise shadow the type below it.

    def json(self, *args: object, **kwargs: object) -> typing.NoReturn:
        """Not available: use to_json()."""
        raise _use_instead("json()", "to_json()", "model_dump_json()")

    def dict(self, *args: object, **kwargs: object) -> typing.NoReturn:
        """Not available: use to_dict()."""
        raise _use_instead("dict()", "to_dict()", "model_dump()")

    def copy(self, *args: object, **kwargs: object) -> typing.NoReturn:
        """Not available: use model_copy()."""
        raise _use_instead("copy()", "model_copy()")

    @classmethod
    def schema(cls, *args: object, **kwargs: object) -> typing.NoReturn:
        """Not available: use record_schema()."""
        raise _use_instead("schema()", "record_schema()", "model_json_schema()")

    @classmethod
    def parse_obj(cls, *args: object, **kwargs: object) -> typing.NoReturn:
        """Not available: use model_validate()."""
        raise _use_instead("parse_obj()", "model_validate()")


def _use_instead(
    method: str, instead: str, complete: str | None = None
) -> InvalidArgumentError:
    """The error for pydantic's deprecated `method`, naming the one to use."""
    other = (
        f" {complete} gives the complete record, with every list in full."
        if complete
        else ""
    )
    return InvalidArgumentError(
        f"{method} is a deprecated pydantic method that this result doesn't "
        f"support: it would give a record that differs from the result's own. "
        f"Use {instead}.{other}"
    )


class Finding(ResultModel):
    """A claim about the data, backed by result fields and carrying an assessment."""

    code: str = Field(description="Stable code naming the kind of finding.")
    claim: str = Field(description="One sentence from a fixed template.")
    assessment: Assessment = Field(
        description=(
            "descriptive: a neutral fact. suspected: a possible problem to judge. "
            "violation: the data breaks something declared. passed: a declared "
            "check held. unknown: a declared check couldn't be evaluated."
        )
    )
    columns: Annotated[list[str], _ENTRIES] = Field(
        description="The columns the finding concerns."
    )
    fields: Annotated[list[str], _ENTRIES] = Field(
        description="Paths of the result fields behind the claim, such as "
        "'columns[2].missing_count'. A mapping entry is named by its key, as "
        "in columns['age'], and a pair by its two columns, as in "
        "pairs['age', 'income']. In the record, every path resolves."
    )


class Limitation(ResultModel):
    """Something that limits how far a result can be read."""

    code: LimitationCode = Field(
        description="Stable code from the package-wide list, LimitationCode."
    )
    reason: str = Field(description="One sentence on what is limited and why.")
    fields: Annotated[list[str], _ENTRIES] = Field(
        description="Paths of the result fields it affects."
    )


class StatusReason(ResultModel):
    """Why a run ended with a status other than completed."""

    code: StatusReasonCode = Field(
        description="Stable code from the package-wide list, StatusReasonCode. "
        "Each analysis's Statuses: section names the codes it returns."
    )
    message: str = Field(description="One sentence on what happened.")


class ColumnSupport(ResultModel):
    """The counts behind one column's results."""

    column: str = Field(description="The column label as a string.")
    eligible_count: int = Field(description="Values eligible for the analysis.")
    evaluated_count: int = Field(description="Values the analysis evaluated.")
    excluded_count: int = Field(description="Eligible values left out.")
    excluded_by_reason: dict[ExclusionReason, int] = Field(
        description="Excluded values per reason: missing, infinite, "
        "unexpected_label, unhashable_values or too_many_categories."
    )

    @classmethod
    def of(
        cls,
        column: str,
        eligible: int,
        excluded_by_reason: Mapping[ExclusionReason, int] | None = None,
    ) -> Self:
        """The support of a column with `eligible` values, `excluded_by_reason`
        of them left out; the other counts follow, so they always add up.

        Raises ValueError when the exclusions outnumber the eligible values.
        """
        reasons = dict(excluded_by_reason or {})
        excluded = sum(reasons.values())
        if excluded > eligible:
            raise ValueError(
                f"column {column!r} excludes {excluded} of {eligible} eligible "
                f"values, by reason {reasons}"
            )
        return cls(
            column=column,
            eligible_count=eligible,
            evaluated_count=eligible - excluded,
            excluded_count=excluded,
            excluded_by_reason=reasons,
        )


class Support(ResultModel):
    """The counts behind a result: rows given and per-column counts."""

    row_count: int = Field(description="Rows in the data given to the analysis.")
    column_count: int = Field(description="Columns the analysis selected.")
    columns: Annotated[list[ColumnSupport], _ENTRIES] = Field(
        description="The selected columns where something was excluded, in the "
        "selected order. A column not listed had every row eligible and "
        "evaluated: eligible_count and evaluated_count equal row_count, and "
        "excluded_count is 0."
    )

    @field_validator("columns", mode="after")
    @classmethod
    def _only_where_something_was_excluded(
        cls, columns: list[ColumnSupport]
    ) -> list[ColumnSupport]:
        return [column for column in columns if column.excluded_count]


class Execution(ResultModel):
    """How the run was executed. Holds no timestamps or durations."""

    versions: dict[str, str] = Field(
        description="Versions of agent_eda, Python and every library the analysis "
        "used, keyed by import name."
    )


class Masking(ResultModel):
    """Whether a result masks the values of sensitive columns, and which ones.

    Masking is a guardrail against accidental exposure, not anonymization and
    not access control.
    """

    applied: bool = Field(
        description="Whether masking was on: every value of a sensitive column "
        "that the result shows, such as a category label or a row ID taken from "
        "id_column, is a numbered placeholder, '<masked 1>', '<masked 2>' and so "
        "on, numbered per column in result order. Placeholders reveal neither "
        "content nor length; counts, shares and statistics are unchanged."
    )
    decided_by: Literal["mask_sensitive", "project", "default"] = Field(
        description="What switched masking on or off: the call's mask_sensitive "
        "argument; the project's data_policy.mask_sensitive, when the call leaves "
        "mask_sensitive out; or default, without either, which leaves masking off."
    )
    columns: Annotated[list[str], _ENTRIES] = Field(
        description="The sensitive columns whose values the result shows, in "
        "result order: as placeholders when applied is true, as stored "
        "otherwise. Empty when the result shows no value of a sensitive column."
    )
    index_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Sensitive columns that the data holds as its index, by "
        "level name, when the result shows index labels as row IDs. Index "
        "labels are never masked, so these values show as stored: pass a "
        "non-sensitive id_column instead."
    )
    absent_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Sensitive columns the project declares that the data lacks "
        "as a column or an index level, in the project's order. Masking matches "
        "column names exactly, so check them for typos; a project spanning "
        "several datasets has some anyway."
    )
    near_miss_columns: Annotated[dict[str, str], _ENTRIES] = Field(
        description="The absent sensitive columns whose names match a column or "
        "index level of the data once case and Unicode form are ignored (casefold "
        "and NFC), each with that name as the data spells it, such as "
        "'Email' for the declared 'email'. Masking matches names exactly, so the "
        "values of such a column are not masked: declare it by the name the data "
        "spells."
    )


def _masking_text(
    masking: Masking, surface: _text.Surface, statistics_columns: Sequence[str] = ()
) -> list[str]:
    """The line on masking that a result prints, when it has something to say.

    `statistics_columns` are sensitive columns whose numeric statistics the
    result shows, which masking never changes, such as a regression target.
    Python text spells settings as Python; MCP text as the MCP tool's JSON
    arguments and the project file's TOML.
    """
    if masking.decided_by == "default":
        return []
    python = surface == "python"
    if masking.decided_by == "mask_sensitive":
        setting = _setting("mask_sensitive", masking.applied, python, ": ")
    else:
        setting = _setting("data_policy.mask_sensitive", masking.applied, python, " = ")
    names = _text.join_labels(masking.columns)
    columns = "column" if len(masking.columns) == 1 else "columns"
    if not masking.applied:
        if not masking.columns:
            return []
        text = (
            f"Not masked ({setting}): the values of sensitive {columns} {names} "
            "show as stored. Pass "
            f"{'mask_sensitive=True' if python else 'mask_sensitive: true'} to "
            "mask them."
        )
    else:
        if masking.columns:
            text = (
                f"Masked ({setting}): the values of {columns} {names} show as "
                "<masked 1>, <masked 2> and so on; counts and shares are "
                "unaffected."
            )
        elif statistics_columns:
            listed = _text.join_labels(statistics_columns)
            one = len(statistics_columns) == 1
            text = (
                f"Masking on ({setting}), but numeric statistics are never masked: "
                f"those of sensitive {'column' if one else 'columns'} {listed}, such "
                f"as {'its' if one else 'their'} minimum and maximum, show as "
                "computed."
            )
        elif not masking.index_columns and not masking.near_miss_columns:
            text = (
                f"Masking on ({setting}), but the result shows no value of a "
                "sensitive column."
            )
        else:
            text = f"Masking on ({setting}), but"
        if masking.index_columns:
            index = _text.join_labels(masking.index_columns)
            held = "column" if len(masking.index_columns) == 1 else "columns"
            text += (
                f"{' And' if masking.columns else ''} index labels are never "
                f"masked, and the index holds sensitive {held} {index}, whose "
                "values show as row IDs. Pass a non-sensitive id_column"
                f"{', such as after df.reset_index()' if python else ''}."
            )
        for declared, spelled in masking.near_miss_columns.items():
            sentence = (
                f"sensitive column {declared!r} isn't a column of the data, and "
                f"masking matches names exactly: did you mean {spelled!r}? The "
                f"values of {spelled!r} are not masked."
            )
            # After "but" the sentence continues; otherwise it starts a new one.
            if not text.endswith("but"):
                sentence = f"{sentence[0].upper()}{sentence[1:]}"
            text += f" {sentence}"
        lacking = [
            name
            for name in masking.absent_columns
            if name not in masking.near_miss_columns
        ]
        if lacking:
            text += f" Sensitive columns the data lacks: {_text.join_labels(lacking)}."
    return _text.wrap(text, first="", indent="    ")


def _setting(name: str, value: bool, python: bool, mcp_separator: str) -> str:
    """A setting as the caller writes it: name=True in Python; over MCP, the
    JSON argument (name: true) or the project file's TOML (name = true)."""
    if python:
        return f"{name}={value}"
    return f"{name}{mcp_separator}{str(value).lower()}"


def _check_status_reason(status: Status, reason: StatusReason | None) -> None:
    if (status == "completed") != (reason is None):
        raise ValueError(
            "status_reason must be null when status is completed and set otherwise"
        )


class ColumnResult(ResultModel):
    """One column's entry in the result of an analysis over several columns."""

    column: str = Field(description="The column name: the label's string form.")
    status: Status = Field(
        description="How the run ended for this column, in the same terms as the "
        "result's status."
    )
    status_reason: StatusReason | None = Field(
        description="Why this column didn't complete; null when its status is "
        "completed."
    )

    @model_validator(mode="after")
    def _reason_matches_status(self) -> Self:
        _check_status_reason(self.status, self.status_reason)
        return self


class _PrintedRecord(ResultModel):
    """A record that prints as plain text and gives its full record as JSON.

    What `print`, `repr`, Rich and devtools show is the text, never pydantic's
    dump of every field, which would bypass the text's caps.
    """

    def _render(self, surface: _text.Surface) -> str:
        """The text, worded for the surface the caller reads it on."""
        raise NotImplementedError(f"{type(self).__name__} has no text")

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON-compatible Python objects."""
        raise NotImplementedError(f"{type(self).__name__} has no record")

    def __str__(self) -> str:
        return self._render("python")

    def __repr__(self) -> str:
        return self.__str__()

    def __rich_console__(self, console: Any, options: Any) -> Iterator[Any]:
        """Rich prints the text that print(res) gives, not pydantic's field dump."""
        yield from _rich_text(str(self))

    def __rich_repr__(self) -> None:  # ty: ignore[invalid-method-override]  # None makes Rich use repr(), the text, instead of listing every field
        return None

    def __pretty__(self, fmt: Any, **kwargs: Any) -> Generator[Any, None, None]:
        """devtools prints the text, as it does for print(res)."""
        yield str(self)

    def to_json(self) -> str:
        """The record from to_dict() as strict JSON, without NaN or Infinity."""
        return _record_json(self.to_dict())


class AnalysisResult(_PrintedRecord):
    """The record an analysis returns.

    Every analysis returns its own subclass, named after it, with its typed fields
    next to these shared ones.
    """

    analysis: str = Field(
        description="Name of the analysis, such as 'describe_dataset'."
    )
    method: str | None = Field(
        description="The method used, or null when the analysis has no methods."
    )
    status: Status = Field(
        description="How the run ended. Each analysis's Statuses: section lists "
        "the ones it can return and why. It says nothing about whether the data "
        "passed a check."
    )
    status_reason: StatusReason | None = Field(
        description="Why the run didn't complete; null when status is completed."
    )
    parameters: Annotated[dict[str, JsonValue], _LISTS_INSIDE] = Field(
        description="The resolved arguments by name, defaults included. data "
        "appears only as its row_count and column_count, and project whole. In "
        "the record, a list or mapping argument keeps its first {limit} entries and "
        "gains <name>_total; a list inside a mapping argument keeps its first "
        "{limit} entries too, and gains <key>_total beside it when it lost some. "
        + _BIG_INTEGERS
    )
    support: Support = Field(description="The counts behind the result.")
    findings: Annotated[list[Finding], _ENTRIES] = Field(
        description="Claims about the data."
    )
    limitations: Annotated[list[Limitation], _ENTRIES] = Field(
        description="What limits how far the result can be read."
    )
    execution: Execution = Field(description="Library versions used for the run.")

    # The tables to_frame can return, by name: the first is the default and the
    # name to_frame accepts as `table`. Each table after the first has a clause
    # for the printed text's footer, such as "has every candidate", where
    # {name} fills in a value from _table_footer_fields; the key table's
    # footer comes from its own text table, so its clause is None. A result
    # with one table declares it too, so the check names it in its message.
    TABLES: ClassVar[Mapping[str, str | None]] = {}

    @model_validator(mode="after")
    def _reason_matches_status(self) -> Self:
        _check_status_reason(self.status, self.status_reason)
        return self

    @classmethod
    def for_call(
        cls,
        *,
        parameters: Mapping[str, Any] | None = None,
        libraries: Sequence[str] = (),
        **fields: Any,
    ) -> Self:
        """The result of the registered analysis that is running, with its
        envelope filled in: the analysis's name, its method, its parameters and
        its execution come from the call, so an analysis passes only the rest.

        `parameters` overrides the recorded value of the arguments whose value
        the result shows differently, such as the columns the analysis
        resolved, or declared values it masked. An argument that holds data
        values, such as sentinel_values, must be overridden when it is given:
        the registry raises instead of recording it as given. `libraries` names
        those the analysis used beyond numpy and pandas.

        Raises RuntimeError outside a registered analysis.
        """
        from agent_eda._envelope import envelope

        return cls(**envelope(parameters, libraries), **fields)

    @property
    def settings(self) -> "Settings":
        """The arguments of the call as the result recorded them, each typed as
        the analysis declares it, such as `res.settings.top_n`. An argument the
        record lacks is None."""
        from agent_eda._envelope import Settings

        return Settings(self.analysis, self.parameters)

    def __eq__(self, other: object) -> bool:
        """Whether both are results of the same class with the same record.

        The record is every field, with row IDs as to_dict() writes them but no
        list cut short. The tables that to_frame() builds from the analysis call
        don't take part, since they follow from the record.
        """
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, AnalysisResult)
        return json.dumps(self.model_dump(mode="json")) == json.dumps(
            other.model_dump(mode="json")
        )

    def _render(self, surface: _text.Surface, reserve: int = 0) -> str:
        """The printed result, worded for the surface the caller reads it on;
        print(res) shows the Python one, an MCP tool's response the MCP one.
        `reserve` lines of the 40 are left free for the caller to add."""
        reason = self.status_reason
        # Wrapped, not cut: a reason ends with how to fix the call.
        facts = (
            _text.wrap(
                f"Reason: {reason.code}: {reason.message}", indent="    ", first=""
            )
            if reason
            else []
        )
        return _text.render(
            heading=self._heading(),
            facts=[
                *facts,
                *self._masking_facts(surface),
                *self._text_facts(surface),
            ],
            table=self._text_table(),
            findings=[
                _text.Item(f"{f.assessment} {f.code}", f.claim)
                for f in sorted(
                    self.findings, key=lambda f: _TEXT_ORDER.index(f.assessment)
                )
            ],
            limitations=[_text.Item(lim.code, lim.reason) for lim in self.limitations],
            other_tables=self._text_other_tables(),
            surface=surface,
            reserve=reserve,
        )

    def _heading(self) -> str:
        shape = self.parameters.get("data")
        rows = shape.get("row_count") if isinstance(shape, dict) else None
        columns = shape.get("column_count") if isinstance(shape, dict) else None
        heading = f"{self.analysis}: {self.status}"
        if isinstance(rows, int) and isinstance(columns, int):
            heading += (
                f" on {_text.plural(rows, 'row', 'rows')} x "
                f"{_text.plural(columns, 'column', 'columns')}"
            )
        if self.method is not None:
            heading += f", method {self.method}"
        return heading

    def to_frame(self) -> pd.DataFrame:
        """The complete main table of the result as a DataFrame."""
        raise NotImplementedError(f"{type(self).__name__} has no table")

    def _check_table(self, table: object) -> None:
        """Raise unless `table` is one of the names in TABLES."""
        _reject_table(table, tuple(self.TABLES))

    def _table_footer_fields(self) -> dict[str, str]:
        """The values the footer clauses in TABLES fill in, by field name."""
        return {}

    def _by_interest[T](self, entries: Sequence[T], names: Sequence[str]) -> list[T]:
        """The entries of a table of columns, those with a finding or a limitation
        first and the rest after them, each group in its own order.

        `names` holds each entry's column name. A finding points at a column by
        name, a limitation by the path of its entry, "columns['age']" or
        "columns[2]".
        """
        named = {name for finding in self.findings for name in finding.columns}
        flagged = {i for i, name in enumerate(names) if name in named}
        for limitation in self.limitations:
            for path in limitation.fields:
                match = _ENTRY_PATH.match(path)
                if match is None:
                    continue
                if match["position"] is not None:
                    flagged.add(int(match["position"]))
                else:
                    label = ast.literal_eval(match["label"])
                    flagged.update(i for i, name in enumerate(names) if name == label)
        first = [entry for i, entry in enumerate(entries) if i in flagged]
        rest = [entry for i, entry in enumerate(entries) if i not in flagged]
        return first + rest

    def _masking_facts(self, surface: _text.Surface) -> list[str]:
        """The line on masking, for results that support it."""
        return []

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        """Lines of key facts under the first line; analyses override this."""
        return []

    def _text_table(self) -> _text.Table | None:
        """The key table; analyses override this."""
        return None

    def _text_other_tables(self) -> list[str]:
        """What to_frame(table=...) returns besides the key table, as clauses for
        the footer: one for each table after the first that has a clause."""
        fields = self._table_footer_fields()
        return [
            f"res.to_frame(table={name!r}) {clause.format(**fields)}"
            for name, clause in list(self.TABLES.items())[1:]
            if clause is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        """The result record as JSON-compatible Python objects.

        Every list that grows with the data keeps its first entries ({row_ids_kept} row
        IDs, {examples_kept} example values, {entries_kept} entries of anything else, and top_n levels
        per column in summarize_categorical) and gains a `<field>_total` key
        with the full count. A capped list of entries that findings or
        limitations point at keeps those too, so every field path in a finding
        or limitation resolves in the record; a path that names an entry by
        position then gives its position in the record. The result object keeps
        every entry, except in the eight fields that hold only their first
        entries on the object too: levels, row IDs, invalid values, classes and
        unexpected or rare labels. to_frame() has every one of them, row IDs
        included.
        """
        pointed = self._pointed_entries()
        record = self.model_dump(
            mode="json", context={_RECORD: True, _POINTED: pointed}
        )
        self._renumber_paths(record, pointed)
        return record_integers(record)

    def _capped_entry_fields(self) -> dict[str, _Cap]:
        """The result's own lists and mappings that the record caps at their
        first entries, where findings and limitations can point at later ones."""
        return {
            name: marker
            for name, field in type(self).model_fields.items()
            for marker in field.metadata
            if isinstance(marker, _Cap) and not marker.each_list and not marker.total
        }

    def _pointed_entries(self) -> dict[str, set[Any]]:
        """The entries of the capped fields that the record's findings and
        limitations name, by field: positions, keys or key tuples."""
        capped = self._capped_entry_fields()
        pointed: dict[str, set[Any]] = {}
        for item in [
            *self.findings[: _ENTRIES.limit],
            *self.limitations[: _ENTRIES.limit],
        ]:
            for path in item.fields[: _ENTRIES.limit]:
                if (found := _subscript(path)) is None or found[0] not in capped:
                    continue
                name, entry = found
                if capped[name].key is not None and not isinstance(entry, tuple):
                    entry = (entry,)
                pointed.setdefault(name, set()).add(entry)
        return pointed

    def _renumber_paths(
        self, record: dict[str, Any], pointed: dict[str, set[Any]]
    ) -> None:
        """Rewrite the paths into lists that name entries by position, when the
        record dropped entries before them, so each still resolves."""
        capped = self._capped_entry_fields()
        moves: dict[str, dict[int, int]] = {}
        for name in pointed:
            cap, values = capped[name], getattr(self, name)
            if cap.key is not None or isinstance(values, dict):
                continue
            kept = _kept_indexes(values, cap, pointed[name])
            if kept is not None:
                moves[name] = {old: new for new, old in enumerate(kept)}
        if not moves:
            return
        for kind in ("findings", "limitations"):
            for item in record[kind]:
                item["fields"] = [_renumbered(path, moves) for path in item["fields"]]

    @classmethod
    def record_schema(cls) -> dict[str, Any]:
        """The JSON Schema of the record that to_dict() and to_json() give.

        It describes the record as it is, with every capped list next to its
        `<field>_total`, and allows no other field. The MCP server gives it to
        each MCP tool as the output schema.
        """
        schema = cls.model_json_schema(mode="validation")
        schema["description"] = (
            f"{inspect.cleandoc(cls.__plain_doc__)} {_RECORD_INTEGERS}".strip()
        )
        definitions = schema.get("$defs", {})
        for model in _models_in(cls, {}):
            target = schema if model is cls else definitions[model.__name__]
            for name, field in model.model_fields.items():
                cap = next((m for m in field.metadata if isinstance(m, _Cap)), None)
                if cap is None or cap.each_list:
                    continue
                # The total is null when the list is null, or when the field it
                # is taken from can be null.
                counted = model.model_fields[cap.total] if cap.total else field
                total: dict[str, Any] = {"type": "integer", "minimum": 0}
                if _is_optional(field.annotation) or _is_optional(counted.annotation):
                    total = {"anyOf": [total, {"type": "null"}]}
                target["properties"][f"{name}_total"] = total
                target.setdefault("required", []).append(f"{name}_total")
        return schema


# The docstring names the caps by placeholder, so the markers stay their only copy.
AnalysisResult.to_dict.__doc__ = fill_caps(AnalysisResult.to_dict.__doc__ or "")


class MaskableResult(AnalysisResult):
    """The result of an analysis that can show raw values, and so supports masking.

    Such an analysis takes project= and mask_sensitive=, and its `masking`
    field says whether the values of sensitive columns are placeholders.
    """

    masking: Masking = Field(
        description="Whether the values of sensitive columns that the result "
        "shows are numbered placeholders, decided by mask_sensitive or the "
        "project's data policy, and which columns that concerns. Masking is a "
        "guardrail against accidental exposure, not anonymization or access "
        "control."
    )

    def _masking_facts(self, surface: _text.Surface) -> list[str]:
        return _masking_text(self.masking, surface)

    def _masked(self, column: str) -> bool:
        """Whether placeholders stand for this column's values in the result."""
        return self.masking.applied and column in self.masking.columns

    def _shown(self, value: object, value_type: str, column: str | None) -> str:
        """A value of `column` as the text shows it: text quoted, unless the
        column is masked and the value a placeholder. `value_type` is the type
        name the result records beside the value."""
        masked = column is not None and self._masked(column)
        return _text.shown_value(value, value_type, masked=masked)

    def _row_id_source(self) -> str:
        """Where the row IDs come from, as "values of id_column 'order'" or
        "index labels"."""
        id_column = self.settings.id_column
        if id_column is None:
            return "index labels"
        return f"values of id_column {id_column!r}"

    def _ids_quoted(self) -> bool:
        """Whether the text quotes row IDs: not when they are the placeholders
        of a masked id_column."""
        id_column = self.settings.id_column
        return not (id_column is not None and self._masked(id_column))


def share_text(name: str, share: float | None) -> str:
    """A share as " (missing_share 0.3333)", a fraction of 4 significant digits
    under its field name; "" when it is null."""
    return "" if share is None else f" ({name} {_text.format_number(share)})"
