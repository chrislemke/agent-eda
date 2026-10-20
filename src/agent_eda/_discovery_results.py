"""The records that guide(), list_analyses(), search_analyses() and describe_analysis() return."""

from typing import Any

import pandas as pd
from pydantic import (
    Field,
    JsonValue,
)

from agent_eda import _text
from agent_eda._annotations import python_literal
from agent_eda._result_base import (
    LimitationCode,
    ResultModel,
    Status,
    StatusReasonCode,
    _entries_frame,
    _PrintedRecord,
)
from agent_eda._written_form import record_integers


class _DiscoveryRecord(_PrintedRecord):
    """What a discovery function returns: plain text to print and a JSON record."""

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON-compatible Python objects."""
        return record_integers(self.model_dump(mode="json"))


class AnalysisInfo(ResultModel):
    """What the registry records about one analysis."""

    name: str = Field(description="The analysis name, such as 'describe_dataset'.")
    module: str = Field(description="The module that defines it.")
    summary: str = Field(
        description="The question it answers: its docstring's first line."
    )
    supports_masking: bool = Field(
        description="Whether it can mask the values of sensitive columns."
    )
    tags: list[str] = Field(description="Words that help search find it.")
    parameters: list[str] = Field(description="Its parameter names, data first.")
    methods: list[str] = Field(
        description="The methods it offers; empty when it has none."
    )
    statuses: list[Status] = Field(description="Every status it can return.")
    finding_codes: list[str] = Field(description="Every finding code it can report.")
    work_limit_parameters: list[str] = Field(
        description="Its max_* parameters that bound the work it does."
    )
    result_class: str = Field(
        description="The name of its result class in agent_eda.results."
    )


class AnalysisListing(_DiscoveryRecord):
    """Every analysis agent-eda offers, ordered by module, then name."""

    analyses: list[AnalysisInfo] = Field(description="One entry per analysis.")

    def _render(self, surface: _text.Surface) -> str:
        count = len(self.analyses)
        lines = [f"{_text.plural(count, 'analysis', 'analyses')}, by module:"]
        module = None
        for info in self.analyses:
            if info.module != module:
                module = info.module
                lines.append(module)
            masking = " [supports masking]" if info.supports_masking else ""
            lines.append(f"  {info.name}: {info.summary}{masking}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per analysis, with every recorded field as a column."""
        return _entries_frame(self.analyses, AnalysisInfo)


# The text of a search listing stays within the printed text's 40 lines, however
# long the query is: the query echo and each word are cut, and so are the lists.
_QUERY_ECHO_LENGTH = 40
_WORD_LENGTH = 30
_MATCHED_SHOWN = 6
_SEARCH_LINES = 40


def _cut(text: str, length: int) -> str:
    """The text, or its first `length` characters and an ellipsis."""
    return text if len(text) <= length else text[:length].rstrip() + "…"


class SearchMatch(ResultModel):
    """An analysis that matched a search, with what matched."""

    name: str = Field(description="The analysis name.")
    module: str = Field(description="The module that defines it.")
    summary: str = Field(
        description="The question it answers: its docstring's first line."
    )
    score: int = Field(
        description="How well it matched: the sum over the matched query terms of "
        "the weight of the best field each occurs in (a tag the query holds whole "
        "5, name 4, tags 3, summary 2, When to use: text 1; a generic word such as "
        "'column' or 'rows' at most 1)."
    )
    matched_terms: list[str] = Field(
        description="The query terms that matched, in query order."
    )
    weak: bool = Field(
        description="Whether the match is weak: only generic words such as 'column' "
        "or 'rows' matched, or the words matched only in the When to use: text. "
        "Read the summary before you rely on a weak match."
    )


class SearchTerm(ResultModel):
    """A word of a search query and the other words it matched."""

    term: str = Field(description="The query word, lowercased.")
    synonyms: list[str] = Field(
        description="The package's word it matched as well, from a small curated "
        "list, such as missing for nulls; empty for any other word."
    )


class SearchListing(_DiscoveryRecord):
    """The analyses that match a search query, best first."""

    query: str = Field(description="The query as given.")
    terms: list[SearchTerm] = Field(
        description="The words searched for, in query order, without common words "
        "such as 'the' and 'of' and without one-letter words, each with its "
        "synonyms. Only the first 20 distinct words are searched."
    )
    term_count: int = Field(
        description="How many distinct words the query holds, without common and "
        "one-letter words; more than the length of terms when the query is long."
    )
    first_look: bool = Field(
        description="Whether the query asks where to start, such as 'first look' or "
        "'EDA'. No single analysis answers it, so matches is empty; the guide "
        "(the MCP server's instructions) gives the order for a first look."
    )
    limit: int = Field(description="The most matches the listing keeps.")
    match_count: int = Field(
        description="How many analyses matched, including any beyond the limit."
    )
    matches: list[SearchMatch] = Field(
        description="The best matches, highest score first, ties by name; empty "
        "when nothing matched."
    )

    def _searched(self) -> str:
        """The words searched for, each cut and the list counted when capped."""
        if not self.terms:
            return (
                "no words, since common ones such as 'the' and 'of' and single "
                "letters don't count"
            )
        listed = ", ".join(
            f"{_cut(term.term, _WORD_LENGTH)} (also {', '.join(term.synonyms)})"
            if term.synonyms
            else _cut(term.term, _WORD_LENGTH)
            for term in self.terms
        )
        if self.term_count > len(self.terms):
            return (
                f"the first {len(self.terms):,} of {self.term_count:,} words, {listed}"
            )
        return listed

    def _render(self, surface: _text.Surface) -> str:
        python = surface == "python"
        echo = _cut(self.query, _QUERY_ECHO_LENGTH)
        searched = f"Searched for: {self._searched()}."
        if self.first_look:
            return "\n".join(
                [
                    *_text.wrap(
                        f"No analysis matches {echo!r}: it asks where to start, "
                        "which no single analysis answers.",
                        first="",
                        indent="    ",
                    ),
                    "ae.guide() prints the order for a first look at a new dataset."
                    if python
                    else "The MCP server's instructions give the order for a first "
                    "look at a new dataset.",
                ]
            )
        if not self.matches:
            lines = [
                *_text.wrap(
                    f"No analysis matches {echo!r}. {searched}",
                    first="",
                    indent="    ",
                ),
                "Try other words, or call ae.list_analyses() for every analysis; "
                "ae.guide() says where to start."
                if python
                else "Try other words, or call list_analyses for every analysis; "
                "the MCP server's instructions say where to start.",
            ]
            return "\n".join(lines)
        # Show as many matches as fit within the line cap.
        for shown in range(len(self.matches), 0, -1):
            lines = self._lines(python, echo, searched, shown)
            if len(lines) <= _SEARCH_LINES:
                break
        return "\n".join(lines)

    def _lines(self, python: bool, echo: str, searched: str, shown: int) -> list[str]:
        """The text with the first `shown` matches."""
        lines = _text.wrap(
            f"Analyses matching {echo!r} ({shown:,} of {self.match_count:,} shown, "
            "best first):",
            first="",
            indent="    ",
        )
        for match in self.matches[:shown]:
            matched = [_cut(term, _WORD_LENGTH) for term in match.matched_terms]
            if len(matched) > _MATCHED_SHOWN:
                rest = len(matched) - _MATCHED_SHOWN
                matched = [*matched[:_MATCHED_SHOWN], f"and {rest:,} more"]
            marker = " (weak match)" if match.weak else ""
            lines.extend(
                _text.wrap(
                    f"{match.name}{marker}: {match.summary} Matched: "
                    f"{', '.join(matched)}.",
                    first="  ",
                    indent="    ",
                )
            )
        if all(match.weak for match in self.matches[:shown]):
            lines.extend(
                _text.wrap(
                    "Every match shown is weak: it rests on generic words such as "
                    "'column' or 'rows', or on the When to use: text alone. Try "
                    "more specific words.",
                    first="",
                    indent="    ",
                )
            )
        lines.extend(_text.wrap(searched, first="", indent="    "))
        if shown < len(self.matches):
            lines.extend(
                _text.wrap(
                    f"More: {len(self.matches) - shown:,} more matches are in "
                    + (
                        "to_dict()['matches']." if python else "the structured content."
                    ),
                    first="",
                    indent="    ",
                )
            )
        elif self.match_count > len(self.matches):
            lines.extend(
                _text.wrap(
                    f"More: ae.search_analyses({self.query!r}, "
                    f"limit={self.match_count}) shows all {self.match_count:,} "
                    "matches."
                    if python and echo == self.query
                    else f"More: search_analyses with limit {self.match_count} "
                    f"shows all {self.match_count:,} matches.",
                    first="",
                    indent="    ",
                )
            )
        lines.extend(
            _text.wrap(
                f"Next: print(ae.describe_analysis({self.matches[0].name!r})) shows "
                "how to call it."
                if python
                else f"Next: the {self.matches[0].name} MCP tool's description and "
                "input schema say how to call it.",
                first="",
                indent="    ",
            )
        )
        return lines


class ParameterDescription(ResultModel):
    """One parameter of an analysis, as describe_analysis shows it."""

    name: str = Field(description="The parameter name.")
    type: str = Field(
        description="Its type as an agent reads it, such as 'Sequence[str] | None'."
    )
    required: bool = Field(description="Whether a call must pass it.")
    default: JsonValue = Field(
        description="Its default as a JSON value; null when it is required or "
        "defaults to None."
    )
    allowed_values: list[JsonValue] | None = Field(
        description="The fixed values it takes, or that each of its values takes "
        "when it is a list; null when any value of its type is allowed."
    )
    valid_range: str | None = Field(
        description="The range it must lie in, such as '0 <= trim_fraction < 0.5', "
        "or 'each value: 0 <= value <= 1' for a list; null when it declares none."
    )
    description: str = Field(
        description="What it means and takes, from the docstring's Args: section."
    )


class ReasonCodeDescription(ResultModel):
    """A status reason code an analysis can return, and what it means."""

    code: StatusReasonCode = Field(description="The status reason code.")
    meaning: str = Field(
        description="What it means for every analysis that returns it, from the "
        "StatusReasonCode list; the status_reason message says the rest."
    )


class LimitationCodeDescription(ResultModel):
    """A limitation code that results can carry, and what it means."""

    code: LimitationCode = Field(description="The limitation code.")
    meaning: str = Field(description="What it means, from the LimitationCode list.")


class StatusDescription(ResultModel):
    """A status an analysis can return, and why it returns it."""

    status: Status = Field(description="The status.")
    reason: str = Field(
        description="When the analysis returns it, from the docstring's Statuses: "
        "section."
    )
    reason_codes: list[ReasonCodeDescription] = Field(
        description="The status reason codes it returns the status with, each "
        "with its meaning; empty for completed, which has no status_reason."
    )


class FindingCodeDescription(ResultModel):
    """A finding code an analysis can report, and what it means."""

    code: str = Field(description="The finding code.")
    description: str | None = Field(
        description="What it means, from the docstring's Returns: section; null "
        "when the docstring doesn't define it in a bullet."
    )


class ResultFieldDescription(ResultModel):
    """A field of an analysis result, and what it holds."""

    name: str = Field(description="The field name.")
    type: str = Field(
        description="Its type as an agent reads it, such as 'list[str]' or "
        "'dict[str, SummarizeNumericColumn]'."
    )
    description: str = Field(description="What it holds and how to read it.")
    fields: list["ResultFieldDescription"] = Field(
        description="The fields of each entry when it holds records, such as "
        "one per column; empty otherwise."
    )


class AnalysisDescription(_DiscoveryRecord):
    """Everything an agent needs for a correct first call of one analysis."""

    name: str = Field(description="The analysis name, such as 'summarize_numeric'.")
    module: str = Field(description="The module that defines it.")
    summary: str = Field(
        description="The question it answers: its docstring's first line."
    )
    when_to_use: str = Field(
        description="When to call it and what to call instead, from the "
        "docstring's When to use: section."
    )
    signature: str = Field(
        description="How to call it, with each parameter's type and default, such "
        "as 'describe_dataset(data: pd.DataFrame, *, source: str | None = None, "
        "...)'. A parameter with fixed choices shows the type of the choices."
    )
    parameters: list[ParameterDescription] = Field(
        description="Every parameter in signature order, data first."
    )
    parameters_schema: dict[str, JsonValue] = Field(
        description="A JSON Schema of the parameters other than data, with their "
        "types, defaults, allowed values, ranges and descriptions, as tool "
        "frameworks take it."
    )
    statuses: list[StatusDescription] = Field(
        description="Every status it can return, in the order it declares them."
    )
    finding_codes: list[FindingCodeDescription] = Field(
        description="Every finding code it can report, in the order it declares them."
    )
    limitation_codes: list[LimitationCodeDescription] = Field(
        description="Every limitation code a result can carry, with its meaning. "
        "The list is the same for every analysis."
    )
    result_class: str = Field(
        description="The name of its result class in agent_eda.results."
    )
    result_fields: list[ResultFieldDescription] = Field(
        description="The fields its result adds to the ones every result has."
    )
    shared_result_fields: list[ResultFieldDescription] = Field(
        description="The fields every analysis result has, such as status and findings."
    )
    methods: list[str] = Field(
        description="The methods it offers; empty when it has none."
    )
    shows_raw_values: bool = Field(
        description="Whether its results can show raw values from the data, such "
        "as category labels, example values or row IDs. Every analysis whose "
        "results can show them supports masking. Numeric statistics, such as a "
        "minimum, don't count and are never masked."
    )
    supports_masking: bool = Field(
        description="Whether it can mask the values of sensitive columns, through "
        "its project and mask_sensitive parameters, replacing them with numbered "
        "placeholders. Masking is a guardrail against accidental exposure, not "
        "anonymization or access control."
    )
    work_limit_parameters: list[str] = Field(
        description="Its max_* parameters that bound the work it does; it returns "
        "budget_exceeded when one is hit."
    )
    example: str = Field(
        description="The doctest from its docstring's Examples: section: Python "
        "lines after '>>> ' and '... ' prompts, each followed by what it prints."
    )

    def _render(self, surface: _text.Surface) -> str:
        masking = (
            "Masking: supported (project=, mask_sensitive=). Its results can show "
            "raw values, such as labels or row IDs, and masking replaces those of "
            "sensitive columns with numbered placeholders."
            if self.supports_masking
            else "Masking: not needed. Its results show no raw values such as "
            "labels, example values or row IDs; numeric statistics, such as a "
            "minimum, are never masked."
        )
        limits = ", ".join(self.work_limit_parameters) or "none"
        lines = [
            f"{self.name}: {self.summary}",
            *_text.wrap(self.signature, first="Signature: ", indent="    "),
            *_text.wrap(
                f"Module: {self.module}. Work limits: {limits}.",
                first="",
                indent="    ",
            ),
            *_text.wrap(masking, first="", indent="    "),
        ]
        if self.methods:
            lines.extend(_text.wrap(f"Methods: {', '.join(self.methods)}."))
        lines.append("When to use:")
        lines.extend(_text.wrap(self.when_to_use, indent="  "))
        lines.append("Parameters:")
        for parameter in self.parameters:
            lines.extend(_parameter_lines(parameter))
        lines.append("Statuses:")
        for status in self.statuses:
            lines.extend(_entry_lines(status.status, status.reason))
            for reason_code in status.reason_codes:
                lines.extend(
                    _text.wrap(
                        f"{reason_code.code}: {reason_code.meaning}",
                        first="    ",
                        indent="        ",
                    )
                )
        lines.append("Finding codes:" if self.finding_codes else "Finding codes: none")
        for finding in self.finding_codes:
            lines.extend(_entry_lines(finding.code, finding.description))
        lines.append("Limitation codes:")
        for limitation in self.limitation_codes:
            lines.extend(_entry_lines(limitation.code, limitation.meaning))
        shared = _text.join_items(
            [field.name for field in self.shared_result_fields],
            limit=len(self.shared_result_fields),
        )
        lines.extend(
            _text.wrap(
                f"Result: {self.result_class}. Besides the fields every result has "
                f"({shared}), it has:",
                first="",
                indent="    ",
            )
        )
        for field in self.result_fields:
            lines.extend(
                _text.wrap(f"{field.name}: {field.type}", first="  ", indent="      ")
            )
            lines.extend(_text.wrap(field.description, indent="    "))
            if field.fields:
                names = ", ".join(inner.name for inner in field.fields)
                lines.extend(_text.wrap(f"Fields: {names}.", indent="    "))
        lines.append("Example:")
        lines.extend(f"  {code}".rstrip() for code in _example_code(self.example))
        lines.extend(
            _text.wrap(
                f"More: help(ae.{self.name}) has the full docstring; to_json() adds "
                "the parameters' JSON Schema.",
                first="",
                indent="    ",
            )
        )
        return "\n".join(lines)


def _parameter_lines(parameter: ParameterDescription) -> list[str]:
    head = f"{parameter.name}: {parameter.type}"
    if parameter.required:
        head += " (required)"
    else:
        head += f" = {python_literal(parameter.default)}"
    if parameter.valid_range:
        head += f"; {parameter.valid_range}"
    lines = _text.wrap(head, first="  ", indent="      ")
    if parameter.allowed_values is not None:
        allowed = ", ".join(python_literal(value) for value in parameter.allowed_values)
        lines.extend(_text.wrap(allowed, first="    Allowed: ", indent="    "))
    lines.extend(_text.wrap(parameter.description, indent="    "))
    return lines


def _entry_lines(name: str, text: str | None) -> list[str]:
    """A "name: text" entry under a heading, its text wrapped below it."""
    return _text.wrap(f"{name}: {text}" if text else name, first="  ", indent="    ")


def _example_code(doctest_text: str) -> list[str]:
    """The Python lines of a doctest, without prompts.

    A statement's output follows it as a comment when it is one line, such as
    'res.columns["age"].median  # 38.0', and is left out when it is longer, such
    as a printed result. As doctest reads it, a line starting with '...' right
    after a '>>>' line continues that statement.
    """
    code: list[str] = []
    printed: list[str] = []

    def add_printed() -> None:
        if len(printed) == 1 and code:
            commented = f"{code[-1]}  # {printed[0].strip()}"
            if len(commented) + 2 <= _text.MAX_WIDTH:
                code[-1] = commented
        printed.clear()

    in_statement = False
    for line in doctest_text.splitlines():
        if line.startswith(">>>"):
            add_printed()
            code.append(line[4:])
            in_statement = True
        elif in_statement and line.startswith("..."):
            code.append(line[4:])
        else:
            in_statement = False
            if line.strip():
                printed.append(line)
    add_printed()
    return code
