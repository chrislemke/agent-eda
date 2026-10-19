"""The registry builds the result envelope: analysis, method, parameters, execution."""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
from recipes import recipe_for
from registering import CountedResult, stand_in_analyses
from strategies import EDGE_FRAMES

import agent_eda as ae
from agent_eda.results import ColumnSupport

ANALYSES = [info.name for info in ae.list_analyses().analyses]
PACKAGE = Path(ae.__file__).parent


def test_the_registry_records_what_the_analysis_didnt_restate():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with stand_in_analyses() as analyses:
        res = analyses["counted"](df, limit=2)
        default = analyses["counted"](df)

    assert res.analysis == "stand_in_counted"
    assert res.method is None
    assert res.parameters == {"data": {"row_count": 3, "column_count": 1}, "limit": 2}
    assert default.parameters["limit"] == 5, "a default is recorded too"
    assert res.seen == 2
    assert list(res.execution.versions)[:4] == [
        "agent_eda",
        "python",
        "numpy",
        "pandas",
    ]


def test_a_result_is_built_for_a_call_only_inside_one():
    with pytest.raises(RuntimeError, match="registered analysis"):
        CountedResult.for_call(
            status="completed",
            status_reason=None,
            support=ae.results.Support(row_count=0, column_count=0, columns=[]),
            findings=[],
            limitations=[],
            seen=0,
        )


def test_an_override_replaces_a_parameter_the_analysis_resolved():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with stand_in_analyses() as analyses:
        res = analyses["sentinels"](df, sentinel_values={"a": ["secret"]})

    assert res.parameters["sentinel_values"] == {"a": ["<masked 1>"]}
    assert "secret" not in res.to_json()


def test_a_forgotten_override_of_data_values_raises_instead_of_leaking():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with stand_in_analyses(forget_override=True) as analyses:
        with pytest.raises(RuntimeError, match="sentinel_values") as raised:
            analyses["sentinels"](df, sentinel_values={"a": ["secret"]})
        # Declaring nothing leaves nothing to leak.
        res = analyses["sentinels"](df)

    assert "secret" not in str(raised.value)
    assert res.parameters["sentinel_values"] is None


def test_an_override_of_an_unknown_parameter_is_a_bug_that_raises():
    df = pd.DataFrame({"a": [1]})
    with (
        stand_in_analyses(unknown_override=True) as analyses,
        pytest.raises(RuntimeError, match="no_such_parameter"),
    ):
        analyses["counted"](df)


def test_column_support_derives_the_counts_from_the_exclusions():
    support = ColumnSupport.of("age", 10, {"missing": 3, "infinite": 2})

    assert support.model_dump() == {
        "column": "age",
        "eligible_count": 10,
        "evaluated_count": 5,
        "excluded_count": 5,
        "excluded_by_reason": {"missing": 3, "infinite": 2},
    }
    assert ColumnSupport.of("age", 4).excluded_by_reason == {}
    assert ColumnSupport.of("age", 4).evaluated_count == 4


def test_column_support_refuses_more_exclusions_than_values():
    with pytest.raises(ValueError, match="age"):
        ColumnSupport.of("age", 2, {"missing": 3})


@pytest.mark.parametrize("name", ANALYSES)
def test_every_support_adds_up_on_the_edge_frames(name):
    for frame in EDGE_FRAMES.values():
        df = frame()
        arguments = recipe_for(name).required(df)
        if arguments is None:
            continue
        res = getattr(ae, name)(df, **arguments)
        record = json.loads(res.to_json())
        assert record["support"]["column_count"] >= len(record["support"]["columns"])
        for column in res.support.columns:
            reasons = sum(column.excluded_by_reason.values())
            assert column.excluded_count == reasons
            assert (
                column.excluded_count == column.eligible_count - column.evaluated_count
            )


def analysis_modules():
    """The source files of the analyses: the public modules of data and statistics."""
    return sorted(
        path
        for area in ("data", "statistics")
        for path in (PACKAGE / area).glob("[!_]*.py")
    )


def test_no_analysis_builds_its_own_envelope():
    built = []
    for path in analysis_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and node.keywords):
                continue
            name = ast.unparse(node.func)
            for keyword in node.keywords:
                own = keyword.arg in {"analysis", "method", "execution"}
                if own and name.endswith("Result"):
                    built.append(f"{path.name}:{node.lineno} {keyword.arg}=")
                if keyword.arg == "parameters" and not name.endswith(".for_call"):
                    built.append(f"{path.name}:{node.lineno} parameters=")

    assert built == []


def test_no_analysis_builds_a_column_support_by_hand():
    built = [
        f"{path.name}:{node.lineno}"
        for path in [*analysis_modules(), PACKAGE / "_numeric.py"]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ColumnSupport"
    ]

    assert built == []


def test_result_text_reads_its_settings_through_the_typed_view():
    found = [
        f"{path.name}:{number}"
        for path in sorted(PACKAGE.rglob("_*results.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "self.parameters.get(" in line
    ]

    assert found == []
