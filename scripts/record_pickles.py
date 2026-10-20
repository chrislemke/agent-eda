"""Write one pickle of every result class, and the record it must give back.

tests/fixtures/pickles/numpy<major>/<ClassName>.pkl is a pickle of a result, written with
agent-eda 0.1.0, next to <ClassName>.json, the result's `to_json()` at the time.
tests/test_result_pickles.py loads each pickle and compares its record, so a
result that a user pickled with 0.1.0 still loads after the classes move.

Run it only to add a class, never to refresh a pickle: a fixture written with the
current code stops proving that an old pickle loads.

    uv run python scripts/record_pickles.py
    uv run --isolated --resolution lowest-direct --extra mcp --python 3.12 \
        python scripts/record_pickles.py     # the NumPy 1 pickles
"""

import importlib
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "pickles" / f"numpy{np.__version__[0]}"
PROTOCOL = 4


def results() -> dict[str, Any]:
    """One result of every class: each analysis's, and the discovery records."""
    sys.path.insert(0, str(ROOT / "tests"))
    strategies = importlib.import_module("strategies")
    recipes = importlib.import_module("recipes")
    frame = strategies.EDGE_FRAMES["many rows"]()
    found: dict[str, Any] = {}
    for info in ae.list_analyses().analyses:
        arguments = recipes.recipe_for(info.name).required(frame)
        res = getattr(ae, info.name)(frame, **(arguments or {}))
        found[type(res).__name__] = res
    for res in (
        ae.list_analyses(),
        ae.search_analyses("missing values"),
        ae.describe_analysis("describe_dataset"),
    ):
        found[type(res).__name__] = res
    return found


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, res in results().items():
        (FIXTURES / f"{name}.pkl").write_bytes(pickle.dumps(res, protocol=PROTOCOL))
        (FIXTURES / f"{name}.json").write_text(res.to_json() + "\n", encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
