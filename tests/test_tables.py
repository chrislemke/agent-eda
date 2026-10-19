"""Each result class declares the tables to_frame returns, in one place.

The class's TABLES names them: the first is the default, and the names are what
to_frame accepts, what its type annotation allows and what the printed text's
footer offers. These tests read all three from the public API and compare them.
"""

import inspect
import typing

import pandas as pd
import pytest
from recipes import recipe_for

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError

DATA = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
        "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        "c": ["x", "x", "y", "y", "z", "z"],
        "d": [1, 1, 1, 1, 1, 1],
    }
)


def result_classes():
    """The result class of every analysis, found by running each on DATA."""
    classes = {}
    for spec in ae.list_analyses().analyses:
        arguments = recipe_for(spec.name).required(DATA)
        if arguments is not None:
            res = getattr(ae, spec.name)(DATA, **arguments)
            classes[spec.name] = res
    return classes


RESULTS = result_classes()


def test_every_analysis_ran_on_the_sample_data():
    assert len(RESULTS) == len(ae.list_analyses().analyses)


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_tables_are_the_names_to_frame_allows_and_its_first_is_the_default(name):
    res = RESULTS[name]
    allowed = typing.get_args(typing.get_type_hints(res.to_frame)["table"])
    default = inspect.signature(res.to_frame).parameters["table"].default
    assert tuple(res.TABLES) == allowed
    assert default == next(iter(res.TABLES))
    assert res.TABLES[default] is None


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_to_frame_returns_every_declared_table_and_names_them_for_a_wrong_one(name):
    res = RESULTS[name]
    for table in res.TABLES:
        assert isinstance(res.to_frame(table=table), pd.DataFrame)
    with pytest.raises(InvalidArgumentError) as caught:
        res.to_frame(table="nothing")
    for table in res.TABLES:
        assert repr(table) in str(caught.value)


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_the_printed_footer_offers_each_other_table(name):
    res = RESULTS[name]
    for table, clause in list(res.TABLES.items())[1:]:
        assert clause is not None
        assert f"res.to_frame(table={table!r})" in str(res)
