import warnings

import numpy as np
import pandas as pd
import pytest

import agent_eda as ae


@pytest.mark.parametrize("category", [DeprecationWarning, FutureWarning])
def test_deprecation_and_future_warnings_from_package_code_fail_the_suite(category):
    # A warning attributed to an agent_eda module, as when pandas warns about a
    # call inside an analysis, raises instead of passing quietly.
    with pytest.raises(category):
        warnings.warn_explicit(
            "deprecated call inside an analysis",
            category,
            filename="provenance.py",
            lineno=1,
            module="agent_eda.data.provenance",
        )


class WarnsWhenHashed:
    """A value whose hashing warns, as a library call inside an analysis might."""

    def __init__(self, category, message):
        self.category = category
        self.message = message

    def __hash__(self):
        warnings.warn(self.message, self.category, stacklevel=2)
        return 1

    def __eq__(self, other):
        return isinstance(other, WarnsWhenHashed)


def warning_frame(category, message="this call is deprecated"):
    return pd.DataFrame({"x": [WarnsWhenHashed(category, message) for _ in range(3)]})


@pytest.mark.parametrize("category", [DeprecationWarning, FutureWarning])
def test_deprecation_and_future_warnings_inside_an_analysis_never_reach_callers(
    category,
):
    df = warning_frame(category)

    # A caller that asked to see every warning still sees none.
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        res = ae.describe_dataset(df)

    assert seen == []
    assert res.status == "completed"
    assert res.columns[0].distinct_count == 1


@pytest.mark.parametrize("category", [DeprecationWarning, FutureWarning])
def test_deprecation_and_future_warnings_inside_an_analysis_fail_the_suite(category):
    df = warning_frame(category)

    # The suite's filterwarnings make these warnings errors, so they surface.
    with pytest.raises(category, match="this call is deprecated"):
        ae.describe_dataset(df)


def test_a_numerical_warning_anywhere_in_an_analysis_becomes_a_limitation():
    df = warning_frame(RuntimeWarning, "overflow encountered in add")

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        res = ae.describe_dataset(df)

    assert seen == []
    [limitation] = [lim for lim in res.limitations if lim.code == "numerical_warning"]
    assert limitation.fields == []
    assert limitation.reason == (
        "Numerical warnings were raised that can't be tied to particular values: "
        "overflow encountered in add."
    )


def test_the_callers_numpy_error_settings_dont_change_a_result():
    df = pd.DataFrame({"big": [-1.7e308, 1.7e308, 1.7e308, -1.7e308]})
    expected = ae.summarize_numeric(df).to_json()

    with np.errstate(all="raise"):
        res = ae.summarize_numeric(df)

    assert res.to_json() == expected
    assert res.columns["big"].std is None
