"""The findings that more than one analysis reports, one builder for each code.

A builder owns the code, the assessment and the claim's wording, so
`constant_columns` reads the same wherever it appears. What an analysis
documents as different is an argument: how the columns are named, which result
fields back the claim and, where the claim says it, the verb or a clause.

A builder returns the finding in a list, empty when no column qualifies, so a
caller spreads it into its findings.
"""

from collections.abc import Sequence
from typing import Literal

from agent_eda._result_base import Assessment, Finding
from agent_eda._text import join_labels, plural

__all__ = [
    "ASSESSMENTS",
    "all_missing_columns",
    "constant_columns",
    "infinite_values",
    "mixed_value_types",
]

# The assessment each shared code carries in every analysis.
ASSESSMENTS: dict[str, Assessment] = {
    "constant_columns": "suspected",
    "all_missing_columns": "suspected",
    "mixed_value_types": "suspected",
    "infinite_values": "suspected",
}


def _finding(
    code: str, claim: str, columns: Sequence[str], fields: Sequence[str]
) -> list[Finding]:
    return [
        Finding(
            code=code,
            claim=claim,
            assessment=ASSESSMENTS[code],
            columns=list(columns),
            fields=list(fields),
        )
    ]


def constant_columns(
    columns: Sequence[str], fields: Sequence[str], *, shown: str | None = None
) -> list[Finding]:
    """Columns that hold a single distinct non-missing value.

    `columns` are the names the finding records and `fields` the paths of the
    values behind it. `shown` is how the claim names the columns, by default as
    `join_labels(columns)`.
    """
    if not columns:
        return []
    hold = plural(len(columns), "column holds", "columns hold")
    named = join_labels(columns) if shown is None else shown
    return _finding(
        "constant_columns",
        f"{hold} a single distinct non-missing value: {named}.",
        columns,
        fields,
    )


def all_missing_columns(
    columns: Sequence[str],
    fields: Sequence[str],
    *,
    verb: Literal["hold", "contain"],
    counting: str = "",
    shown: str | None = None,
) -> list[Finding]:
    """Columns that hold only missing values.

    `verb` is the claim's verb, which analyses word differently: 'hold' or
    'contain'. `counting` is a clause between "missing values" and the names,
    such as ", counting declared sentinel values", for an analysis that counts
    more than the stored missing values.
    """
    if not columns:
        return []
    word = plural(len(columns), f"column {verb}s", f"columns {verb}")
    named = join_labels(columns) if shown is None else shown
    return _finding(
        "all_missing_columns",
        f"{word} only missing values{counting}: {named}.",
        columns,
        fields,
    )


def infinite_values(
    columns: Sequence[str],
    fields: Sequence[str],
    *,
    shown: str | None = None,
    claim: str | None = None,
) -> list[Finding]:
    """Columns that hold infinite values, which the statistics leave out.

    `claim` replaces the sentence for an analysis that counts the values of one
    column rather than naming columns, as profile_target does for its target.
    """
    if not columns:
        return []
    if claim is None:
        hold = plural(len(columns), "column holds", "columns hold")
        named = join_labels(columns) if shown is None else shown
        claim = f"{hold} infinite values, which every statistic leaves out: {named}."
    return _finding("infinite_values", claim, columns, fields)


def mixed_value_types(
    columns: Sequence[str],
    fields: Sequence[str],
    *,
    wording: Literal["python_types", "different_types"],
    shown: str | None = None,
) -> list[Finding]:
    """Columns that hold values of more than one type.

    `wording` is how the claim puts it: 'python_types' says "values of more than
    one Python type", for an analysis that lists the types; 'different_types'
    says "mix values of different types".
    """
    if not columns:
        return []
    named = join_labels(columns) if shown is None else shown
    if wording == "python_types":
        hold = plural(len(columns), "column holds", "columns hold")
        claim = f"{hold} values of more than one Python type: {named}."
    else:
        mix = plural(len(columns), "column mixes", "columns mix")
        claim = f"{mix} values of different types: {named}."
    return _finding("mixed_value_types", claim, columns, fields)
