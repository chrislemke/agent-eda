"""The process that tests/test_determinism.py starts once per PYTHONHASHSEED.

Usage: python determinism_worker.py FRAMES.pickle

It runs every analysis with its recipe's required arguments on every fixed
frame the recipe takes and on the pickled frames, and prints one JSON object:
for each "analysis|frame" the record, the printed text and every table as CSV,
or the error the call raised. The caller compares what two processes print.
"""

import json
import pickle
import sys
import typing
from pathlib import Path

import pandas as pd
from recipes import RECIPES
from strategies import EDGE_FRAMES, LABEL_EDGE_FRAMES

import agent_eda as ae
from agent_eda.results import row_id_json


def table_text(table: pd.DataFrame) -> str:
    """A table as JSON. Cells that hold objects, such as the row IDs of a group
    or a label that is a frozenset, are written with row_id_json, since the
    table holds the objects themselves and pandas would write a set in hash order.
    """
    return json.dumps(
        {
            "columns": [row_id_json(label) for label in table.columns],
            "dtypes": [str(dtype) for dtype in table.dtypes],
            "index": [row_id_json(label) for label in table.index],
            "rows": [
                [row_id_json(cell) for cell in row] for row in table.to_numpy(object)
            ],
        },
        sort_keys=True,
        allow_nan=False,
    )


def tables(res: typing.Any) -> dict[str, str]:
    """Every table to_frame() returns, by how to get it."""
    hints = typing.get_type_hints(res.to_frame)
    names = typing.get_args(hints["table"]) if "table" in hints else (None,)
    return {
        str(name): table_text(
            res.to_frame() if name is None else res.to_frame(table=name)
        )
        for name in names
    }


def record(name: str, df: pd.DataFrame) -> dict[str, typing.Any] | None:
    arguments = RECIPES[name].required(df)
    if arguments is None:
        return None
    try:
        res = getattr(ae, name)(df, **arguments)
        return {"to_json": res.to_json(), "text": str(res), "tables": tables(res)}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def main(path: str) -> None:
    # The pickle is the one the test just wrote from its own frames: trusted.
    generated: list[tuple[str, pd.DataFrame]] = pickle.loads(Path(path).read_bytes())
    records: dict[str, typing.Any] = {}
    for info in ae.list_analyses().analyses:
        name = info.name
        fixed = dict(EDGE_FRAMES)
        if RECIPES[name].any_column_labels:
            fixed |= LABEL_EDGE_FRAMES
        cases = [(label, build()) for label, build in fixed.items()] + generated
        for label, df in cases:
            result = record(name, df)
            if result is not None:
                records[f"{name}|{label}"] = result
    json.dump(records, sys.stdout, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
