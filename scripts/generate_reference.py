"""Write the README's Reference from the registry, the docstrings and `ae.errors`.

The docstring is the only hand-written copy of what an analysis does. This
development tool, which the wheel doesn't ship, copies it into the README between
`<!-- reference:start -->` and `<!-- reference:end -->`, so that the Reference can't
disagree with it. `tests/test_readme.py` fails when the README is stale.

    uv run python scripts/generate_reference.py           # rewrite the Reference
    uv run python scripts/generate_reference.py --check   # exit 1 when it is stale

The output holds no timestamp and follows no hash order, so running the tool twice
in a row changes nothing.
"""

import argparse
import inspect
import re
import sys
import typing
from collections.abc import Callable, Sequence
from pathlib import Path

import agent_eda as ae
from agent_eda import mcp
from agent_eda._codes import limitation_meanings
from agent_eda._docstring import parse as parse_docstring
from agent_eda.discovery import FIRST_LOOK
from agent_eda.results import AnalysisDescription, Finding, StatusDescription

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
COMMAND = "uv run python scripts/generate_reference.py"
START = "<!-- reference:start -->"
END = "<!-- reference:end -->"
ASSESSMENTS = ("descriptive", "suspected", "violation", "passed", "unknown")

# The first sentence ends at a full stop followed by a capital letter or a quote,
# so "such as 0, or as" and "df.columns[:3]" stay inside it.
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z`'])")
_MODULE_PREFIX = re.compile(r"\b(?:[a-z_]+\.)+(?=[A-Za-z_])")
_LITERAL = re.compile(r"Literal\[[^\]]*\]")


def render_reference() -> str:
    """The Reference as Markdown, from the registry and the docstrings."""
    descriptions = {
        info.name: ae.describe_analysis(info.name)
        for info in ae.list_analyses().analyses
    }
    parts = [
        "Generated from the docstrings by `scripts/generate_reference.py`: edit a "
        "docstring and rerun it, not this section. Every public name in `agent_eda`, "
        "with the first sentence of each argument and field. `help(ae.<name>)` shows "
        'the whole docstring and `ae.describe_analysis("<name>")` gives an analysis '
        "as data.",
        _analyses(descriptions),
        _discovery(),
        _projects(),
        _results(),
        _errors(),
        _mcp_command(),
    ]
    return "\n\n".join(parts) + "\n"


def update(readme: str) -> str:
    """The README text with the Reference between the markers regenerated."""
    start = readme.find(START)
    end = readme.find(END)
    if start < 0 or end < start:
        raise SystemExit(f"README.md needs {START} above {END}.")
    return f"{readme[: start + len(START)]}\n\n{render_reference()}\n{readme[end:]}"


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the Reference, or with --check only compare it; the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when stale")
    parser.add_argument("--readme", type=Path, default=README, help="file to update")
    options = parser.parse_args(argv)
    text = options.readme.read_text(encoding="utf-8")
    current = update(text)
    if current == text:
        return 0
    if options.check:
        print(
            f"The Reference in {options.readme} is stale. Run `{COMMAND}` and commit "
            "the result.",
            file=sys.stderr,
        )
        return 1
    options.readme.write_text(current, encoding="utf-8")
    print(f"Wrote the Reference to {options.readme}.")
    return 0


def _analyses(descriptions: dict[str, AnalysisDescription]) -> str:
    first_look = [name for _, names in FIRST_LOOK for name in names]
    names = first_look + sorted(set(descriptions) - set(first_look))
    assessments = Finding.model_fields["assessment"].description or ""
    intro = (
        "### Analysis functions\n\n"
        "Every analysis takes the DataFrame first, as `data`, and never changes it. "
        "All other arguments are keyword-only. Each returns its own result class, "
        "which has the fields under [Results](#results) next to the ones listed "
        f"here. A finding's assessment is in brackets: {assessments}"
    )
    return "\n\n".join([intro, *(_analysis(descriptions[name]) for name in names)])


def _analysis(description: AnalysisDescription) -> str:
    function = getattr(ae, description.name)
    call = _call(description.name, function, f" -> {description.result_class}")
    lines = [f"#### `{description.name}`", f"`{call}`", description.summary]
    rows = [
        (
            parameter.name,
            parameter.type,
            _default(function, parameter.name),
            _sentence(parameter.description),
        )
        for parameter in description.parameters[1:]
    ]
    lines.append(_table(("Argument", "Type", "Default", "Meaning"), rows))
    lines.append(f"**Fields:** {_fields(description)}")
    findings = []
    for code in description.finding_codes:
        assessment, _, claim = (code.description or "").partition("; ")
        if assessment not in ASSESSMENTS or not claim:
            raise SystemExit(
                f"{description.name}: start the Returns: bullet of `{code.code}` "
                f"with its assessment, one of {ASSESSMENTS}, a semicolon and the claim."
            )
        findings.append(f"- `{code.code}` ({assessment}): {_sentence(claim)}")
    lines.extend(["\n".join(findings)] if findings else [])
    statuses = ", ".join(_status(status) for status in description.statuses)
    lines.append(f"**Statuses:** {statuses}")
    return "\n\n".join(lines)


def _status(status: StatusDescription) -> str:
    """A status with the reason codes it returns, such as `not_applicable` (`no_columns`)."""
    codes = ", ".join(f"`{entry.code}`" for entry in status.reason_codes)
    return f"`{status.status}`" + (f" ({codes})" if codes else "")


def _fields(description: AnalysisDescription) -> str:
    """The result fields that the analysis adds, each with its entry's fields."""
    fields = []
    for field in description.result_fields:
        inner = ", ".join(f"`{sub.name}`" for sub in field.fields)
        fields.append(f"`{field.name}`" + (f" ({inner})" if inner else ""))
    return ", ".join(fields)


def _discovery() -> str:
    lines = ["### Discovery functions"]
    for name in ("guide", "search_analyses", "describe_analysis", "list_analyses"):
        function = getattr(ae, name)
        doc = parse_docstring(inspect.getdoc(function) or "")
        returns = _type(inspect.signature(function).return_annotation)
        lines.append(f"#### `{name}`")
        lines.append(f"`{_call(name, function, f' -> {returns}')}`")
        lines.append(doc.summary)
        rows = [
            (arg, _type(parameter.annotation), _default(function, arg), _sentence(text))
            for (arg, text), parameter in zip(
                doc.args.items(),
                inspect.signature(function).parameters.values(),
                strict=True,
            )
        ]
        if rows:
            lines.append(_table(("Argument", "Type", "Default", "Meaning"), rows))
        lines.append("**Returns:** " + _sentence(doc.returns.split(": ", 1)[-1]))
    return "\n\n".join(lines)


def _projects() -> str:
    lines = ["### Projects and schemas"]
    for name in ("Project", "DataPolicy", "ColumnSpec"):
        cls = getattr(ae, name)
        lines.append(f"#### `{name}`")
        lines.append(f"`{_call(name, cls)}`")
        lines.append((inspect.getdoc(cls) or "").split("\n")[0])
        rows = [
            (
                field,
                _type(parameter.annotation),
                _default(cls, field),
                _sentence(cls.model_fields[field].description or ""),
            )
            for field, parameter in inspect.signature(cls).parameters.items()
        ]
        lines.append(_table(("Argument", "Type", "Default", "Meaning"), rows))
    methods = []
    for method in (ae.Project.save, ae.Project.load):
        parameters = list(inspect.signature(method).parameters.values())
        if parameters[0].name == "self":
            call = f"project.{method.__name__}{_bare(method, skip=1)}"
        else:
            call = f"ae.Project.{method.__name__}{_bare(method)}"
        summary = (inspect.getdoc(method) or "").split("\n")[0]
        returns = _type(inspect.signature(method).return_annotation)
        returns = "Project" if returns == "Self" else returns
        methods.append(f"- `{call} -> {returns}`: {summary}")
    lines.append("\n".join(methods))
    kinds = ", ".join(f"`'{kind}'`" for kind in typing.get_args(ae.LogicalType))
    lines.append(
        f"#### `LogicalType`\n\nThe logical type names as a `Literal`: {kinds}."
    )
    return "\n\n".join(lines)


def _results() -> str:
    limitations = limitation_meanings().items()
    rows = [
        (f"`{name}`", _sentence(field.description or ""))
        for name, field in ae.AnalysisResult.model_fields.items()
    ]
    methods = []
    for name in ("to_frame", "to_dict", "to_json", "record_schema"):
        method = getattr(ae.AnalysisResult, name)
        summary = (inspect.getdoc(method) or "").split("\n")[0]
        owner = "ResultClass" if name == "record_schema" else "res"
        methods.append(f"- `{owner}.{name}()`: {summary}")
    return "\n\n".join(
        [
            "### Results",
            "#### `AnalysisResult`",
            " ".join(inspect.cleandoc(ae.AnalysisResult.__plain_doc__).split())
            + " `print(res)` gives plain text from a fixed template. The shared fields:",
            "| Field | What it holds |\n| --- | --- |\n"
            + "\n".join(f"| {name} | {text} |" for name, text in rows),
            "The shared methods:",
            "\n".join(methods),
            "Each limitation's `code` is one of these:",
            "\n".join(
                f"- `{code}`: {_sentence(meaning)}" for code, meaning in limitations
            ),
        ]
    )


def _errors() -> str:
    # Entries such as "column_not_found (ColumnNotFoundError): a column ...", whose
    # text goes on in the lines indented further.
    entry = re.compile(r"^    (\w+) \((\w+)[^)]*\):(.*(?:\n {8}.*)*)", flags=re.M)
    rows = []
    for code, cls, text in entry.findall(ae.errors.__doc__ or ""):
        text = _sentence(" ".join(text.split())).split("; ")[0].rstrip(".")
        if not re.match(r"\w*[_.]", text):  # not a name such as Project.load
            text = text[:1].upper() + text[1:]
        rows.append((f"`{code}`", f"`{cls}`", text + "."))
    intro = (ae.errors.__doc__ or "").split("\n\nCodes,")[0].split("\n\n")[0]
    table = "| Code | Exception | Raised when |\n| --- | --- | --- |\n" + "\n".join(
        f"| {a} | {b} | {c} |" for a, b, c in rows
    )
    return "\n\n".join(["### Errors", " ".join(intro.split()), table])


def _mcp_command() -> str:
    parser = mcp._parser()  # the one place the option text lives
    usage = parser.format_usage().removeprefix("usage: ").replace(" [-h]", "").strip()
    [option] = [a for a in parser._actions if a.dest == "project"]
    return "\n\n".join(
        [
            "### MCP server command",
            f"`{usage}`",
            f"{_sentence(parser.description or '')} `{option.option_strings[0]} "
            f"{option.metavar}`: {_sentence(option.help or '')} "
            "[Use over MCP](#use-over-mcp) has the details.",
        ]
    )


def _call(name: str, function: Callable[..., object], returns: str = "") -> str:
    prefix = f"ae.{name}"
    return f"{prefix}{_bare(function)}{returns}"


def _bare(function: Callable[..., object], skip: int = 0) -> str:
    """The parameters with their defaults, the keyword-only star in place."""
    parts: list[str] = []
    parameters = list(inspect.signature(function).parameters.values())[skip:]
    for parameter in parameters:
        if parameter.kind is parameter.KEYWORD_ONLY and "*" not in parts:
            parts.append("*")
        if parameter.default is parameter.empty:
            parts.append(parameter.name)
        else:
            parts.append(f"{parameter.name}={parameter.default!r}")
    return f"({', '.join(parts)})"


def _default(function: Callable[..., object], name: str) -> str:
    default = inspect.signature(function).parameters[name].default
    return "required" if default is inspect.Parameter.empty else f"`{default!r}`"


def _type(annotation: object) -> str:
    """An annotation as a short name: modules dropped, a Literal as its str."""
    if annotation is None or annotation is type(None):
        return "None"
    if annotation == ae.LogicalType:
        return "LogicalType"
    text = annotation if isinstance(annotation, str) else _name(annotation)
    text = _MODULE_PREFIX.sub("", text)
    return _LITERAL.sub("str", text)


def _name(annotation: object) -> str:
    if isinstance(annotation, type) and not typing.get_args(annotation):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _sentence(text: str) -> str:
    """The first sentence of a description, which the docstring keeps whole."""
    return _SENTENCE_END.split(" ".join(text.split()), maxsplit=1)[0]


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def cell(index: int, value: str) -> str:
        # A pipe in the type, such as `str | None`, must be escaped inside a table.
        return f"`{value}`".replace("|", "\\|") if index == 1 else value

    if not rows:
        return ""
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    for row in rows:
        formatted = [f"`{row[0]}`"] + [cell(i, v) for i, v in enumerate(row[1:], 1)]
        lines.append(f"| {' | '.join(formatted)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
