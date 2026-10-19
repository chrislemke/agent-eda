"""The fixed plain-text template every result prints with.

A result prints as: a first line with the analysis, status and data size; an
optional status reason; a few fact lines; the key table (at most 10 rows); the
findings (at most 5); the limitations (at most 5); and a footer that says where
the rest is. The text never exceeds 40 lines or 120 characters per line, and
every section counts what it leaves out.

The text names what the caller can reach on its surface: a Python caller reads
res.to_frame() and res.to_json(), an MCP caller the full table that table_path
writes and the structured content. Only those phrases differ; the rest of the
text is the same on both surfaces.
"""

import textwrap
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

MAX_LINES = 40
MAX_WIDTH = 120
MAX_TABLE_ROWS = 10
MAX_FINDINGS = 5
MAX_LIMITATIONS = 5
# Claims name at most this many columns or labels and count the rest.
MAX_NAMED = 5

ELLIPSIS = "..."
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
    lines = [heading, *facts]
    if table is not None:
        lines.extend(_table_lines(table))
    lines.extend(_item_lines("Findings", findings, MAX_FINDINGS))
    lines.extend(_item_lines("Limitations", limitations, MAX_LIMITATIONS))
    lines.extend(_footer(table, other_tables if surface == "python" else (), surface))
    lines = [_fit(_printable(line)) for line in lines]
    limit = MAX_LINES - reserve
    if len(lines) > limit:
        hidden = len(lines) - limit + 1
        lines = [
            *lines[: limit - 1],
            f"... {hidden:,} more lines; see {_RECORD[surface]}.",
        ]
    return "\n".join(lines)


# Where the whole record is, on each surface.
_RECORD: dict[Surface, str] = {
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


def join_items(items: Sequence[str], *, limit: int = MAX_NAMED) -> str:
    """Join items as "a, b and c", naming at most `limit` and counting the rest."""
    if len(items) > limit:
        shown = list(items[:limit])
        return ", ".join(shown) + f" and {len(items) - limit:,} more"
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


def placeholder(number: int) -> str:
    """What masking shows in place of the value numbered `number`, from 1.

    Such as '<masked 1>': it reveals neither the value's content nor its length.
    """
    return f"<masked {number}>"


def label_text(label: object) -> str:
    """Show a column or index label: strings quoted, everything else as str()."""
    return repr(label) if isinstance(label, str) else str(label)


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
    lines.extend(f"- {item.label}: {item.text}" for item in shown)
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
        frame = f"{full_table} has {'the' if count == 1 else 'all'} {rows}"
    return wrap(
        f"More: {'; '.join([frame, *other_tables])}; {rest}", first="", indent="    "
    )


def _printable(text: str) -> str:
    """Escape characters that would break or hide lines, such as newlines."""
    return "".join(
        ch.encode("unicode_escape").decode("ascii")
        if unicodedata.category(ch) in ("Cc", "Zl", "Zp")
        else ch
        for ch in text
    )


def _fit(line: str) -> str:
    if len(line) <= MAX_WIDTH:
        return line
    return line[: MAX_WIDTH - len(ELLIPSIS)] + ELLIPSIS
