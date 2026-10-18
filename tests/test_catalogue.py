"""The catalogue marks every analysis that shipped, so its entry isn't read as a plan."""

import re
from pathlib import Path

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "docs" / "eda-python-implementation-catalogue.md"


def test_marks_every_analysis_as_implemented_with_its_differences():
    rows = CATALOGUE.read_text().splitlines()

    unmarked = []
    for info in ae.list_analyses().analyses:
        entries = [row for row in rows if row.startswith(f"| `{info.name}`")]
        shipped = rf"\*\*Implemented in agent-eda \d+\.\d+\*\* as `{info.module}\."
        if not (
            len(entries) == 1
            and re.search(shipped + rf"{info.name}`", entries[0])
            and "**Implemented differently:**" in entries[0]
        ):
            unmarked.append(info.name)
    assert unmarked == []
