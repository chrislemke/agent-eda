"""The MCP server's instructions: the guide for a caller that only calls MCP tools.

They are built from the guide's parts, the listing and the first-look order,
with MCP text in place of the guide's Python: data paths in place of
DataFrames, the response in place of the result object, and project_path in
place of a Project. Like the guide, they stay within 60 lines of at most 120
characters. Nothing here imports mcp.
"""

import textwrap

from agent_eda._data_files import supported_extensions
from agent_eda._version import __version__
from agent_eda.discovery import _first_look_lines, list_analyses

_INTRO = """\
agent-eda {version}: deterministic exploratory data analysis (EDA) of data files. Each MCP tool runs one analysis,
which answers one EDA question; the same call on the same file always gives the same result."""

# Wrapped when the instructions are built, since the extensions come from the
# reader.
_DATA = (
    "Pass the data as data_path: a {extensions} file, absolute or relative to the "
    "MCP server's working directory. In a CSV or TSV only an empty field counts as "
    "missing, so values such as N/A or -999 reach the analysis as values. Row IDs in "
    "a result are 0-based data-row numbers for a CSV or TSV and follow the stored "
    "index for a Parquet file; id_column names rows by a column's values instead. "
    "Each MCP tool's input schema describes the rest."
)

_RESULTS = """\
Read the response:
- The text is the printed result: status, key numbers, findings and limitations, in at most 40 lines.
- The structured content is the whole record as strict JSON. Lists that grow with the data are capped (10 row IDs,
  5 example values, 100 other entries) and <field>_total gives the full count; a value that doesn't exist is null.
- table_path, a new .csv, .parquet or .pq file, gets the complete table past those caps, lists written as JSON text.
  It never overwrites: an existing file gives the error table_file_exists, and nothing is written.
- status says how the run ended: completed, partial, insufficient_data, not_applicable, missing_prerequisites,
  budget_exceeded, blocked_by_policy or failed; status_reason says why it didn't complete. That is not an error.
- findings are claims about the data, each with a code and an assessment: descriptive, suspected, violation, passed
  or unknown. limitations say what limits how far the numbers can be read.
- A mistake in the call comes back with isError true, a stable code and a message that says how to fix the call."""

_MASKING = """\
Masking: MCP tools marked [supports masking] can show raw values, such as labels or row IDs. Their project_path takes
a project TOML file whose [data_policy] table has sensitive_columns = ["email"] and mask_sensitive = true, and the
values show as <masked 1> and so on. agent-eda-mcp --project PATH sets the project for every call; a project_path
replaces it for one call, and mask_sensitive wins over both. Masking is a guardrail against accidental exposure, not
anonymization or access control."""

_MORE = """\
Find more: search_analyses finds MCP tools for a question in your own words, such as "nulls per column", and
list_analyses lists every one with what it declares. Each MCP tool's description and input schema hold its docs."""


def instructions() -> str:
    """The server's instructions, naming every registered analysis."""
    listing = list_analyses()
    return "\n".join(
        [
            _INTRO.format(version=__version__),
            "",
            textwrap.fill(
                _DATA.format(
                    extensions=supported_extensions().replace(" and ", " or ")
                ),
                width=120,
                break_on_hyphens=False,
            ),
            "",
            _RESULTS,
            "",
            str(listing),
            "",
            _MASKING,
            "",
            "A first look at a new dataset, in this order:",
            *_first_look_lines(listing),
            "",
            _MORE,
        ]
    )
