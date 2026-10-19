"""What an MCP caller can't read: Python syntax, found in text written for it.

An MCP caller passes JSON arguments and file paths, so the text it reads (a tool
description, an input schema, an error message) never holds a Python call, a
keyword argument, a pandas or numpy name or a Python constant used as a value.
"""

import re
from typing import Any

# A call, such as to_frame() or ae.guide(): a name directly before a bracket.
CALL = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\(")
# A keyword argument, such as task='x' or columns=None; TOML's `key = value`
# with spaces is how an MCP caller writes a project file, so it isn't one.
KEYWORD = re.compile(r"\b[A-Za-z_]\w*=[^\s=]")
# A Python module or variable, such as pd.NA, df.loc, res.columns or ae.guide.
NAME = re.compile(r"\b(?:pd|np|df|res|ae)\.")
# None, True or False as a value; a quoted one, such as the text 'None' in a
# list of stand-ins for missing values, is data, not a constant.
CONSTANT = re.compile(r"(?<![\w'\"`])(?:None|True|False)(?![\w'\"`])")

PYTHON = re.compile(
    "|".join(pattern.pattern for pattern in (CALL, KEYWORD, NAME, CONSTANT))
)


def python_in(text: str) -> list[str]:
    """The Python syntax in the text, each with the words around it."""
    return [
        text[max(0, m.start() - 30) : m.end() + 30].replace("\n", " ")
        for m in PYTHON.finditer(text)
    ]


def strings_in(schema: object, path: str = "") -> list[tuple[str, str]]:
    """Every text a JSON Schema holds for a reader, with its path: descriptions,
    titles and examples, not the keywords that give types and patterns."""
    found: list[tuple[str, str]] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in ("$ref", "$schema", "type", "format", "pattern", "enum", "const"):
                continue
            found += strings_in(value, f"{path}/{key}")
    elif isinstance(schema, list):
        for index, value in enumerate(schema):
            found += strings_in(value, f"{path}/{index}")
    elif isinstance(schema, str):
        found.append((path, schema))
    return found


_WORDS = {"None": "null", "True": "true", "False": "false"}


# No file an MCP tool reads has a label that isn't a string.
NON_STRING_LABELS = re.compile(
    r"\s?A label that isn't a string can be given as itself(?:, such as 0,)? "
    r"or as its string form\."
)


def json_words(text: str) -> str:
    """The text with None, True and False written as JSON writes them, and
    without the sentence on labels that aren't strings, which is how an MCP
    caller reads a docstring's Args."""
    return CONSTANT.sub(lambda match: _WORDS[match[0]], NON_STRING_LABELS.sub("", text))


def titles_in(schema: Any, path: str = "") -> list[str]:
    """The paths of the pydantic titles in a JSON Schema, which an MCP tool's
    schemas leave out. A property that is itself named title isn't one."""
    found = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "title" and isinstance(value, str):
                found.append(path)
            else:
                found += titles_in(value, f"{path}/{key}")
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            found += titles_in(item, f"{path}/{index}")
    return found


def without_titles(schema: Any) -> Any:
    """A JSON Schema without its pydantic titles."""
    if isinstance(schema, dict):
        return {
            key: without_titles(value)
            for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(schema, list):
        return [without_titles(item) for item in schema]
    return schema


def json_words_in(schema: Any) -> Any:
    """A JSON Schema with json_words applied to each description in it."""
    if isinstance(schema, dict):
        return {
            key: json_words(value)
            if key == "description" and isinstance(value, str)
            else json_words_in(value)
            for key, value in schema.items()
        }
    if isinstance(schema, list):
        return [json_words_in(item) for item in schema]
    return schema
