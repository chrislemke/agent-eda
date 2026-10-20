"""The MCP server's instructions: the guide for a caller that only calls MCP tools.

They share the guide's overview, the areas with their analysis counts, the
first-look order and the pointers to the rest, with MCP text in place of the
guide's Python: data paths in place of DataFrames, the response in place of the
result object, and project_path in place of a Project. Like the guide, they stay
within 60 lines of at most 120 characters however many analyses exist, and hold
no module headings. Nothing here imports mcp, so agent-eda-mcp --help reuses the
sentences on masking that the instructions and the MCP tools' descriptions
share.
"""

import textwrap

from agent_eda._mcp.local_files import (
    CSV_RULES,
    PARQUET_RULES,
    READABLE,
    WRITABLE,
    extensions,
)
from agent_eda._result_base import fill_caps
from agent_eda._version import __version__
from agent_eda.discovery import overview, status_lines

# Which project a call uses, in every MCP text that names project_path.
PRECEDENCE = (
    "A call's project_path replaces the MCP server's project (agent-eda-mcp "
    "--project) for that call, and a call's mask_sensitive wins over both."
)
# What masking is, in every MCP text on masking (ADR 0001).
GUARDRAIL = (
    "Masking is a guardrail against accidental exposure, not anonymization or "
    "access control."
)

_INTRO = """\
agent-eda {version}: deterministic exploratory data analysis (EDA) of data files. Each analysis is an MCP tool that
answers one EDA question; the same call on the same file always gives the same result."""

# Wrapped when the instructions are built, since the extensions come from the
# reader. The tool definitions' data_path, project_path and table_path say one
# sentence each; their rules are here.
_DATA = (
    "Pass the data as data_path: a {extensions} file, absolute or relative to the "
    "MCP server's working directory (a leading ~ is the home directory); the "
    "extension, in any case, names the format. In a CSV or TSV only an empty field "
    "counts as missing, so N/A or -999 are values, and row IDs are 0-based data-row "
    "numbers; in a Parquet file they follow the stored index. id_column names rows "
    "by a column's values. " + CSV_RULES + " " + PARQUET_RULES
)

_RESULTS = """\
Read the response: an MCP client may show you its text, its structured content or both.
- The text is the printed result: status, key numbers, findings and limitations, in at most 40 lines.
- The structured content is the whole record as strict JSON. Lists that grow with the data are capped \
({row_ids_kept} row IDs,
  {examples_kept} example values, {entries_kept} other entries) and <field>_total gives the full count; \
a value that doesn't exist is null.
- table_path, a new {tables} file in a directory that exists and is writable, gets the main table in full
  (not an analysis's other tables), with lists and mappings as JSON text and the index as first column only when it
  is more than row numbers. The text then ends with "Full table written to <path> (<n> rows).", and the record's
  table_file gives the path and row_count. It never overwrites or writes through a symbolic link: an existing path
  gives the error table_file_exists before the analysis runs, and a failed write leaves no file.
{status}
- findings are claims about the data, each with a code and an assessment: descriptive, suspected, violation, passed
  or unknown. limitations say what limits how far the numbers can be read.
- A mistake in the call comes back with isError true, a stable code and a message that says how to fix the call."""

# Wrapped when the instructions are built, around the shared sentences.
_MASKING = (
    "Masking: MCP tools that list_analyses marks [supports masking] can show raw "
    "values, such as labels or row IDs. Their project_path takes a project TOML file "
    "(absolute or relative to the MCP server's working directory, a leading ~ "
    "allowed) with an optional top-level name and a "
    '[data_policy] table that has sensitive_columns = ["email"] and mask_sensitive = '
    "true or false; masked values show as <masked 1> and so on. null, the default, "
    f"uses no project unless agent-eda-mcp --project loaded one. {PRECEDENCE} Numeric "
    "statistics and index labels are never masked; other tools ignore the "
    f"project. {GUARDRAIL}"
)


def instructions() -> str:
    """The MCP server's instructions, with the overview of the analyses."""
    return "\n".join(
        [
            _INTRO.format(version=__version__),
            "",
            _fill(_DATA.format(extensions=extensions(READABLE, either=True))),
            "",
            fill_caps(_RESULTS).format(
                status="\n".join(status_lines("", " That is not an error.")),
                tables=extensions(WRITABLE, either=True),
            ),
            "",
            _fill(_MASKING),
            "",
            *overview("mcp"),
        ]
    )


def _fill(text: str) -> str:
    """A paragraph wrapped at 120 characters, never inside a hyphenated word."""
    return textwrap.fill(text, width=120, break_on_hyphens=False)
