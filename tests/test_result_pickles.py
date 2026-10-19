"""A result that a user pickled with agent-eda 0.1.0 still loads and gives its record.

tests/fixtures/pickles holds, for every result class, a pickle written with 0.1.0
and the record it gave. The pickles name their classes by module, so they keep
loading only while ae.results gives every class by name. scripts/record_pickles.py
says how a class is added.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

import agent_eda as ae

# A result holds NumPy arrays and DataFrames, which a NumPy 2 pickle can't give
# back on NumPy 1: each major version has the pickles it wrote.
FIXTURES = Path(__file__).parent / "fixtures" / "pickles" / f"numpy{np.__version__[0]}"
CLASSES = sorted(path.stem for path in FIXTURES.glob("*.pkl"))


def test_every_result_class_has_a_pickle_fixture():
    expected = {info.result_class for info in ae.list_analyses().analyses}
    expected |= {"AnalysisListing", "SearchListing", "AnalysisDescription"}

    assert set(CLASSES) == expected


@pytest.mark.parametrize("name", CLASSES)
def test_a_result_pickled_with_0_1_0_loads_and_gives_the_same_record(name):
    # The fixtures are files of this repository, written by scripts/record_pickles.py.
    loaded = pickle.loads(
        (FIXTURES / f"{name}.pkl").read_bytes()
    )  # a fixture of this repository, never untrusted data

    assert type(loaded) is getattr(ae.results, name)
    expected = (FIXTURES / f"{name}.json").read_text(encoding="utf-8")
    assert loaded.to_json() + "\n" == expected
    assert (
        pickle.loads(pickle.dumps(loaded)) == loaded
    )  # a round trip of a value made here
