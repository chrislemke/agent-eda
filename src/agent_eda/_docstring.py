"""The sections of an analysis's Google-style docstring.

Discovery reads each analysis's docstring, so the docstring is the only copy of
its summary, parameter descriptions and the rest. The contract test holds every
analysis's docstring to the format read here.
"""

import re
import textwrap
from dataclasses import dataclass
from typing import Any

# The section headers, in the order every analysis's docstring has them.
SECTIONS = ("When to use:", "Args:", "Returns:", "Statuses:", "Examples:", "See Also:")

# An entry of Args:, Statuses: or See Also:, such as "columns (Sequence[str]): ...",
# whose text continues on the more deeply indented lines below it.
_ENTRY = re.compile(r"(?P<name>[A-Za-z_]\w*)(?: \([^)]*\))?:(?: (?P<text>.*))?")
# The end of a Statuses: entry that names the status reason codes it returns,
# such as "Reason codes: `no_rows`, `no_values`."
_REASON_CODES = re.compile(
    r"\s*Reason codes?: (?P<codes>(?:`[a-z]\w*`(?:, | and |, and )?)+)\.$"
)
_CODE = re.compile(r"`([a-z]\w*)`")
# A bullet that defines a code, such as "- `constant_columns`: a column holds",
# whose text continues on the lines below it indented by two spaces. Any text
# between the backticks reads as a code, so registration can reject a malformed one.
_BULLET = re.compile(r"- `(?P<code>[^`]+)`: (?P<text>.*)")


@dataclass(frozen=True)
class Docstring:
    """An analysis's docstring, read section by section."""

    # The first line: the question the analysis answers.
    summary: str
    # When to use it and what to use instead, as one paragraph.
    when_to_use: str
    # Each parameter's description from Args:, as one paragraph.
    args: dict[str, str]
    # Why the analysis returns each status, from Statuses:, without the reason
    # codes that end an entry.
    statuses: dict[str, str]
    # The status reason codes each status returns, from the "Reason codes:"
    # sentence that ends its Statuses: entry; empty when it names none.
    status_codes: dict[str, tuple[str, ...]]
    # The meaning of each code defined in Returns: by a bullet such as
    # "- `constant_columns`: a column holds ...", finding codes among them.
    codes: dict[str, str]
    # The whole of Returns:, dedented, with its line breaks and bullets.
    returns: str
    # The doctest of Examples:, as it runs.
    examples: str
    # Each related analysis in See Also:, with what it answers.
    see_also: dict[str, str]


def parse(doc: str) -> Docstring:
    """Read a docstring cleaned by inspect.getdoc; missing sections read empty."""
    lines = doc.splitlines()
    sections = _sections(lines)
    statuses: dict[str, str] = {}
    status_codes: dict[str, tuple[str, ...]] = {}
    for status, text in _entries(sections.get("Statuses:", [])).items():
        named = _REASON_CODES.search(text)
        statuses[status] = text[: named.start()] if named else text
        status_codes[status] = tuple(_CODE.findall(named["codes"])) if named else ()
    return Docstring(
        summary=lines[0].strip() if lines else "",
        when_to_use=_paragraph(
            [line.strip() for line in sections.get("When to use:", [])]
        ),
        args=_entries(sections.get("Args:", [])),
        statuses=statuses,
        status_codes=status_codes,
        codes=_bullets(sections.get("Returns:", [])),
        returns="\n".join(sections.get("Returns:", [])),
        examples="\n".join(sections.get("Examples:", [])),
        see_also=_entries(sections.get("See Also:", [])),
    )


def without_examples(schema: dict[str, Any]) -> None:
    """Cut the Examples: section from the description of a pydantic model's JSON
    Schema, which is the model's docstring, since a doctest is Python."""
    description = schema.get("description")
    if description is not None:
        schema["description"] = description.split("\n\nExamples:")[0]


def _sections(lines: list[str]) -> dict[str, list[str]]:
    """Each section's lines, without the section's indentation."""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines[1:]:
        if line in SECTIONS:
            current = found.setdefault(line, [])
        elif current is not None:
            current.append(line)
    return {
        header: textwrap.dedent("\n".join(body)).strip("\n").splitlines()
        for header, body in found.items()
    }


def _entries(lines: list[str]) -> dict[str, str]:
    """Each entry's name and its text, the lines below it joined into one."""
    entries: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        match = _ENTRY.fullmatch(line)
        if match:
            current = entries.setdefault(match["name"], [])
            current.append(match["text"] or "")
        elif current is not None and line.strip():
            current.append(line.strip())
    return {name: _paragraph(parts) for name, parts in entries.items()}


def _bullets(lines: list[str]) -> dict[str, str]:
    """Each code a bullet defines, and its text, continuation lines included."""
    bullets: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        match = _BULLET.fullmatch(line)
        if match:
            current = bullets.setdefault(match["code"], [match["text"]])
        elif current is not None and line.startswith("  ") and line.strip():
            current.append(line.strip())
        else:
            current = None
    return {code: _paragraph(parts) for code, parts in bullets.items()}


def _paragraph(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)
