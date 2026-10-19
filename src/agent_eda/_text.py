"""The fixed plain-text template every result prints with.

A result prints as: a first line with the analysis, status and data size; an
optional status reason; a few fact lines; the key table (at most 10 rows) with
optional detail lines under it; the findings (at most 5); the limitations (at
most 5); and a footer that says where the rest is. The text never exceeds 40
lines or 120 characters per line, and every section counts what it leaves out.

A line that is too long wraps with a hanging indent and is never cut: only a
table cell too wide for its column is shortened, with "...", and the record
keeps the whole value. The detail lines are the only lines that give way to the
others when the 40 lines run short.

The text names what the caller can reach on its surface: a Python caller reads
res.to_frame() and res.to_json(), an MCP caller the full table that table_path
writes and the structured content. Only those phrases differ; the rest of the
text is the same on both surfaces.
"""

import difflib
import reprlib
import textwrap
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agent_eda._written_form import canonical_text

MAX_LINES = 40
MAX_WIDTH = 120
MAX_TABLE_ROWS = 10
MAX_FINDINGS = 5
MAX_LIMITATIONS = 5
# Claims name at most this many columns or labels and count the rest.
MAX_NAMED = 5
# Suggestions for a misspelled name, and the shortest start of a name that
# suggests it.
MAX_SUGGESTIONS = 3
MIN_PREFIX = 3

# A message names at most this many names it didn't find, each cut to
# MAX_NAME_LENGTH characters, so that a column's values passed for names can't
# put more than a glimpse of the data into it.
MAX_UNKNOWN_NAMED = 3
MAX_NAME_LENGTH = 40

ELLIPSIS = "..."
# What a line that wraps continues with.
HANGING = "    "
COLUMN_GAP = "  "
# Text cells never shrink below this width to make a table fit.
MIN_TEXT_WIDTH = 8

type Cell = str | int | float | bool | None

# Where the caller reads the text: print(res) in Python, or the text of an MCP
# tool's response.
type Surface = Literal["python", "mcp"]


@dataclass(frozen=True)
class Table:
    """The key table of a result: every row, of which the text shows 10."""

    title: str
    # What a row is, in the singular and the plural, such as ("column", "columns").
    unit: tuple[str, str]
    headers: Sequence[str]
    rows: Sequence[Sequence[Cell]]
    # How many rows the table has when `rows` holds only the first ones, which
    # saves formatting rows the text never shows; None means len(rows).
    total: int | None = None
    # What res.to_frame() holds when its rows aren't the table's, such as
    # levels for a table of columns: the row count and the unit.
    frame_rows: int | None = None
    frame_unit: tuple[str, str] | None = None
    # What res.to_frame() has, in words, when a count and a unit can't say it,
    # such as "all 5 rows in groups"; it wins over frame_rows and frame_unit.
    # None means the count of the table's rows or of frame_rows.
    frame: str | None = None
    # Lines under the table, one per shown row, such as a column's top levels,
    # which the text shows as far as the line cap leaves room, under the title
    # `details_title`.
    details: Sequence[str] = ()
    details_title: str = ""


@dataclass(frozen=True)
class Item:
    """A finding or limitation line: '- <label>: <text>'."""

    label: str
    text: str


def render(
    *,
    heading: str,
    facts: Sequence[str],
    table: Table | None,
    findings: Sequence[Item],
    limitations: Sequence[Item],
    other_tables: Sequence[str] = (),
    surface: Surface = "python",
    reserve: int = 0,
) -> str:
    """Render a result with the fixed template, within the line and width caps.

    `reserve` lines are kept free below the cap, for lines the caller adds.

    `other_tables` names the tables besides the key table that to_frame()
    returns, each as a clause for the footer, such as "res.to_frame(table='x')
    has ...". Only Python can ask for them, so the MCP text leaves them out.
    """
    lines = [
        line
        for text in [heading, *facts]
        for line in _wrapped(_printable(text), indent=HANGING)
    ]
    if table is not None:
        lines.extend(_fit(_printable(line)) for line in _table_lines(table))
    after_table = [
        *_item_lines("Findings", findings, MAX_FINDINGS),
        *_item_lines("Limitations", limitations, MAX_LIMITATIONS),
        *_footer(table, other_tables if surface == "python" else (), surface),
    ]
    if table is not None and table.details:
        room = MAX_LINES - reserve - len(lines) - len(after_table)
        lines.extend(_detail_lines(table, room))
    lines.extend(after_table)
    limit = MAX_LINES - reserve
    if len(lines) > limit:
        hidden = len(lines) - limit + 1
        lines = [
            *lines[: limit - 1],
            f"... {hidden:,} more lines; see {_FOOTER_RECORD[surface]}.",
        ]
    return "\n".join(lines)


# Where the whole record is, on each surface.
_FOOTER_RECORD: dict[Surface, str] = {
    "python": "res.to_json()",
    "mcp": "the structured content",
}


def also_in_table(what: str, surface: Surface) -> str:
    """The fact line saying the full table has more than the text shows."""
    if surface == "python":
        return f"res.to_frame() also has {what}."
    return f"The full table (table_path) also has {what}."


def field_name(name: str, surface: Surface) -> str:
    """A field of the result as the caller reaches it: res.name in Python, the
    record's key over MCP."""
    return f"res.{name}" if surface == "python" else name


def format_cell(value: Cell) -> str:
    """Counts get thousands separators, other numbers 4 significant digits."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return format_number(value)
    return value


def format_number(value: float) -> str:
    """A number with 4 significant digits, such as 0.1235 or 1.235e+06."""
    return f"{value:.4g}"


def format_bytes(count: int) -> str:
    """A byte count, with a binary unit at 4 significant digits from 1 KiB up."""
    if count < 1024:
        return f"{count:,} bytes"
    size = float(count)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        size /= 1024
        if size < 1024 or unit == "TiB":
            break
    return f"{format_number(size)} {unit} ({count:,} bytes)"


def join_items(
    items: Sequence[str], *, limit: int = MAX_NAMED, total: int | None = None
) -> str:
    """Join items as "a, b and c", naming at most `limit` and counting the rest.

    `total` is how many items there are when `items` holds only the first ones.
    """
    count = len(items) if total is None else total
    if count > limit:
        shown = list(items[:limit])
        return ", ".join(shown) + f" and {count - limit:,} more"
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def join_alternatives(items: Sequence[str]) -> str:
    """Join items as "a, b or c"."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def wrap(text: str, *, indent: str = "", first: str | None = None) -> list[str]:
    """Wrap text into lines of at most MAX_WIDTH characters.

    The first line starts with `first`, by default the indent, and the others
    with `indent`. Words longer than a line are split, and no line ends in a
    space, which textwrap leaves when a line fills up right before a long word.
    """
    lead = indent if first is None else first
    lines = textwrap.wrap(
        text,
        width=MAX_WIDTH,
        initial_indent=lead,
        subsequent_indent=indent,
        break_on_hyphens=False,
    )
    return [line.rstrip() for line in lines] or [lead.rstrip()]


def plural(count: int, singular: str, plural_form: str) -> str:
    """Return "1 column" or "3 columns" with thousands separators."""
    return f"{count:,} {singular if count == 1 else plural_form}"


def closest_names(name: str, names: Sequence[str]) -> list[str]:
    """The names closest to `name`, at most MAX_SUGGESTIONS of them.

    Names equal to it apart from case and surrounding spaces come first, then
    names that start with it, such as 'binary_classification' for 'binary',
    then close spellings, ranked by difflib's similarity.
    """
    by_key: dict[str, list[str]] = {}
    for candidate in names:
        by_key.setdefault(_fold(candidate), []).append(candidate)
    key = _fold(name)
    starting = (
        [k for k in by_key if k != key and k.startswith(key)]
        if len(key) >= MIN_PREFIX
        else []
    )
    similar = difflib.get_close_matches(key, list(by_key), n=MAX_SUGGESTIONS)
    found = [
        *by_key.get(key, []),
        *(c for k in starting for c in by_key[k]),
        *(c for k in similar for c in by_key[k]),
    ]
    return list(dict.fromkeys(found))[:MAX_SUGGESTIONS]


def _fold(name: str) -> str:
    return name.strip().casefold()


def a_or_an(noun: str) -> str:
    """The noun with its article: "an object", "an Interval", "a str"."""
    return f"{'an' if noun[:1] in 'AEIOUaeiou' else 'a'} {noun}"


def placeholder(number: int) -> str:
    """What masking shows in place of the value numbered `number`, from 1.

    Such as '<masked 1>': it reveals neither the value's content nor its length.
    """
    return f"<masked {number}>"


_SHORT = reprlib.Repr(maxstring=MAX_NAME_LENGTH, maxother=MAX_NAME_LENGTH)


def short_repr(value: object) -> str:
    """A scalar as repr() shows it, cut to MAX_NAME_LENGTH characters.

    A long string keeps its start and end around '...'.
    """
    return _SHORT.repr(value)


def label_text(label: object) -> str:
    """Show a column or index label: strings quoted, everything else as str().

    Sets list their items in a fixed order and objects show no memory address,
    so the text is the same in every process.
    """
    return repr(label) if isinstance(label, str) else canonical_text(label)


def join_labels(labels: Sequence[object], *, quote: bool = True) -> str:
    """Show labels as "'a', 'b' and 3", naming at most MAX_NAMED of them.

    With quote=False, strings show without quotes, as masking's placeholders
    such as <masked 1> do.
    """
    show = label_text if quote else str
    shown = [show(label) for label in labels[:MAX_NAMED]]
    if len(labels) > MAX_NAMED:
        return ", ".join(shown) + f" and {len(labels) - MAX_NAMED:,} more"
    return join_items(shown)


def _table_lines(table: Table) -> list[str]:
    total = len(table.rows) if table.total is None else table.total
    if total == 0:
        return [f"{table.title}: none"]
    shown = table.rows[:MAX_TABLE_ROWS]
    cells = [[_printable(format_cell(value)) for value in row] for row in shown]
    numeric = [
        all(isinstance(row[i], int | float) for row in shown if row[i] is not None)
        for i in range(len(table.headers))
    ]
    widths = _column_widths(table.headers, cells, numeric)
    lines = [f"{table.title} ({len(shown):,} of {total:,} shown):"]
    for row in [list(table.headers), *cells]:
        lines.append(
            COLUMN_GAP.join(
                _pad(cell, width, right=is_numeric)
                for cell, width, is_numeric in zip(row, widths, numeric, strict=True)
            ).rstrip()
        )
    return lines


def _column_widths(
    headers: Sequence[str], cells: list[list[str]], numeric: list[bool]
) -> list[int]:
    widths = [
        max([len(header), *(len(row[i]) for row in cells)])
        for i, header in enumerate(headers)
    ]
    floor = [
        len(header) if is_numeric else max(min(len(header), widths[i]), MIN_TEXT_WIDTH)
        for i, (header, is_numeric) in enumerate(zip(headers, numeric, strict=True))
    ]
    # Shrink the widest text column until the table fits the line width.
    while sum(widths) + len(COLUMN_GAP) * (len(widths) - 1) > MAX_WIDTH:
        candidates = [i for i in range(len(widths)) if widths[i] > floor[i]]
        if not candidates:
            break
        widest = max(candidates, key=lambda i: (widths[i], -i))
        widths[widest] -= 1
    return widths


def _pad(cell: str, width: int, *, right: bool) -> str:
    if len(cell) > width:
        cell = cell[: max(width - len(ELLIPSIS), 0)] + ELLIPSIS
    return cell.rjust(width) if right else cell.ljust(width)


def _item_lines(title: str, items: Sequence[Item], limit: int) -> list[str]:
    if not items:
        return [f"{title}: none"]
    shown = items[:limit]
    lines = [f"{title} ({len(shown):,} of {len(items):,} shown):"]
    for item in shown:
        lines.extend(
            wrap(_printable(f"- {item.label}: {item.text}"), first="", indent="  ")
        )
    return lines


def _detail_lines(table: Table, room: int) -> list[str]:
    """The table's detail lines that fit in `room` lines, and a count of the rest."""
    entries = [
        wrap(_printable(text), first="  ", indent="    ")
        for text in table.details[:MAX_TABLE_ROWS]
    ]
    if room < 1:
        return []
    if room < 3:
        return [f"{table.details_title}: {len(entries):,} not shown for lack of room."]
    budget = room - 1
    if sum(len(entry) for entry in entries) <= budget:
        shown = len(entries)
    else:
        used = 0
        shown = 0
        for entry in entries:
            if used + len(entry) > budget - 1:
                break
            used += len(entry)
            shown += 1
    lines = [f"{table.details_title}:"]
    for entry in entries[:shown]:
        lines.extend(entry)
    if shown < len(entries):
        lines.append(f"  and {len(entries) - shown:,} more not shown.")
    return lines


def _footer(
    table: Table | None, other_tables: Sequence[str], surface: Surface
) -> list[str]:
    if surface == "python":
        rest = "res.findings, res.limitations and res.to_json() have the rest."
        full_table = "res.to_frame()"
    else:
        rest = "the structured content has the rest."
        full_table = "the full table, which table_path writes to a file,"
    if table is None:
        return [f"More: {rest}"]
    if table.frame is not None:
        frame = f"{full_table} has {table.frame}"
    else:
        if table.frame_rows is not None:
            count = table.frame_rows
        elif table.total is not None:
            count = table.total
        else:
            count = len(table.rows)
        rows = plural(count, *(table.frame_unit or table.unit))
        frame = f"{full_table} has " + (
            "no rows" if count == 0 else f"{'the' if count == 1 else 'all'} {rows}"
        )
    return wrap(
        f"More: {'; '.join([frame, *other_tables])}; {rest}", first="", indent="    "
    )


# The categories of characters the text writes as escapes: controls such as
# newlines, line and paragraph separators, lone surrogates, which no output can
# encode, and format characters such as the right-to-left override, which
# reorder or hide what is around them.
_ESCAPED_CATEGORIES = ("Cc", "Zl", "Zp", "Cs", "Cf")


def _printable(text: str) -> str:
    """Escape characters that would break, hide or reorder text, such as newlines
    and bidirectional controls, and lone surrogates, which can't be encoded."""
    return "".join(
        ch.encode("unicode_escape").decode("ascii")
        if unicodedata.category(ch) in _ESCAPED_CATEGORIES
        else ch
        for ch in text
    )


def _wrapped(text: str, *, indent: str) -> list[str]:
    """The text as lines within MAX_WIDTH. A line that is too long wraps, with
    the indent it starts with, or `indent` when it starts at the margin."""
    if len(text) <= MAX_WIDTH:
        return [text]
    lead = text[: len(text) - len(text.lstrip())]
    return wrap(text.strip(), first=lead, indent=lead or indent)


def _fit(line: str) -> str:
    """A table line within MAX_WIDTH; only a table too wide to shrink is cut."""
    if len(line) <= MAX_WIDTH:
        return line
    return line[: MAX_WIDTH - len(ELLIPSIS)] + ELLIPSIS
