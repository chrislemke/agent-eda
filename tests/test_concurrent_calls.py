import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import agent_eda as ae

CALLS = 400
THREADS = 8

ANALYSES = [
    ae.summarize_numeric,
    ae.describe_dataset,
    ae.summarize_categorical,
    ae.compute_association_matrix,
]


class WarnsWhenHashed:
    """A value whose hashing raises a RuntimeWarning, as a NumPy call might."""

    def __hash__(self):
        warnings.warn("overflow encountered in add", RuntimeWarning, stacklevel=2)
        return 1

    def __eq__(self, other):
        return isinstance(other, WarnsWhenHashed)


def clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "c": ["x", "y", "x", "y", "x", "z"],
        }
    )


def noisy_frame() -> pd.DataFrame:
    frame = clean_frame()
    frame["c"] = pd.Series([WarnsWhenHashed() for _ in range(6)], dtype=object)
    return frame


def overflowing_frame() -> pd.DataFrame:
    huge = np.finfo("float64").max
    return pd.DataFrame({"big": [-huge, huge, huge, -huge], "small": [1.0, 2, 3, 4]})


def mixed_calls():
    """Every third call is noisy, and for summarize_numeric half of those overflow."""
    calls = []
    for i in range(CALLS):
        analysis = ANALYSES[i % len(ANALYSES)]
        if i % 3 != 0:
            calls.append((analysis, clean_frame(), False))
        elif analysis is ae.summarize_numeric:
            calls.append((analysis, overflowing_frame(), True))
        else:
            calls.append((analysis, noisy_frame(), True))
    return calls


def run_in_threads(calls):
    with ThreadPoolExecutor(THREADS) as pool:
        return list(pool.map(lambda call: call[0](call[1]), calls))


def test_concurrent_calls_leave_the_callers_warning_settings_untouched(capfd):
    filters = list(warnings.filters)
    showwarning = warnings.showwarning
    # The hook that a leaked catch_warnings(record=True) replaces.
    showwarnmsg = warnings._showwarnmsg_impl  # ty: ignore[unresolved-attribute]  # private hook, read only to prove it is unchanged

    results = run_in_threads(mixed_calls())

    assert len(results) == CALLS
    assert warnings.filters == filters
    assert warnings.showwarning is showwarning
    assert warnings._showwarnmsg_impl is showwarnmsg  # ty: ignore[unresolved-attribute]  # the same private hook as above
    assert capfd.readouterr().err == ""
    # A warning raised afterwards in the caller is still shown.
    with warnings.catch_warnings(record=True) as seen:
        warnings.warn("after the pool", UserWarning, stacklevel=1)
    assert [str(w.message) for w in seen] == ["after the pool"]


def test_concurrent_calls_give_numerical_warning_to_noisy_results_only(capfd):
    calls = mixed_calls()

    results = run_in_threads(calls)

    for (analysis, _, noisy), res in zip(calls, results, strict=True):
        codes = [lim.code for lim in res.limitations]
        assert ("numerical_warning" in codes) == noisy, (analysis.__name__, codes)
    assert capfd.readouterr().err == ""
