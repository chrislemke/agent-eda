"""The opt-in benchmark: every analysis on 1,000,000 rows x 100 mixed columns.

Run it with `uv run pytest -m benchmark`; the default run skips it. Each analysis
gets the arguments its recipe requires, so every option keeps its default, and
must finish in under 5 seconds, the target on a 2-vCPU, 8 GB AgentCore session.
NumPy's BLAS and pyarrow use 2 threads here to match, and CI also keeps the
process on 2 CPUs with taskset. A call on the first 1,000 rows loads what the
analysis imports before the timed call. The timings print at the end of the run.
"""

import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from recipes import recipe_for
from threadpoolctl import threadpool_limits

import agent_eda as ae

pytestmark = pytest.mark.benchmark

ANALYSES = [info.name for info in ae.list_analyses().analyses]
ROWS = 1_000_000
LIMIT_SECONDS = 5.0
THREADS = 2
MISSING_SHARE = 0.055  # of the columns that can be missing: about 5% overall

# Distinct values per column; None means every value is distinct.
FLOAT_COLUMNS = 25
INTEGER_CARDINALITIES: list[int | None] = [
    *(2, 3, 5, 10, 50, 100, 1_000, 5_000, 10_000, 50_000),
    *(100_000, 300_000, 500_000, 900_000, None),
]
TEXT_CARDINALITIES: list[int | None] = [
    *(2, 3, 5, 7, 10, 20, 50, 100, 200, 1_000, 2_000, 10_000),
    *(50_000, 100_000, 200_000, 500_000, 800_000, None, None, None),
]
CATEGORY_CARDINALITIES: list[int | None] = [
    *(2, 2, 3, 4, 5, 8, 10, 12, 20, 50, 100, 1_000, 10_000, 100_000, None),
]
BOOLEAN_COLUMNS, NULLABLE_BOOLEAN_COLUMNS = 7, 3
DATETIME_CARDINALITIES: list[int | None] = [
    *(365, 1_000, 24 * 365, 10_000, 50_000, 100_000, 500_000, None),
    *(1_000, 10_000, 100_000, 500_000, None, 2_000, None),
]
TIME_ZONES = {13: "UTC", 14: "Europe/Berlin"}  # by datetime column number


def benchmark_frame(rows: int = ROWS, seed: int = 0) -> pd.DataFrame:
    """A deterministic DataFrame of 100 mixed columns.

    25 float64 columns hold missing values, every 3rd is rounded to one
    decimal (ties) and every 5th holds 0.1% inf and 0.1% -inf. 15 Int64, 20
    text and 15 categorical columns range from 2 distinct values to all
    distinct. Text is object on pandas 2 and str on pandas 3, as read_csv
    gives it; the 3 all-distinct text columns have no missing values, like an
    ID. 10 booleans (3 nullable) and 15 datetimes (2 time-zone aware) follow.
    About 5% of all values are missing.
    """
    rng = np.random.default_rng(seed)
    columns: dict[str, object] = {}

    def missing() -> np.ndarray:
        return rng.random(rows) < MISSING_SHARE

    def codes(cardinality: int | None) -> np.ndarray:
        if cardinality is None or cardinality >= rows:
            return rng.permutation(rows)
        return rng.integers(0, cardinality, rows)

    for i in range(FLOAT_COLUMNS):
        x = rng.normal(10.0 * i, 1.0 + i, rows)
        if i % 3 == 0:
            x = np.round(x, 1)
        x[missing()] = np.nan
        if i % 5 == 0:
            x[rng.random(rows) < 0.001] = np.inf
            x[rng.random(rows) < 0.001] = -np.inf
        columns[f"f{i:02d}"] = x
    for i, cardinality in enumerate(INTEGER_CARDINALITIES):
        integers = pd.array(codes(cardinality), dtype="Int64")
        integers[missing()] = pd.NA
        columns[f"i{i:02d}"] = integers
    for i, cardinality in enumerate(TEXT_CARDINALITIES):
        text = _text(codes(cardinality), cardinality)
        if cardinality is not None:  # all-distinct text is an ID: never missing
            text[missing()] = None
        columns[f"t{i:02d}"] = pd.Series(text)
    for i, cardinality in enumerate(CATEGORY_CARDINALITIES):
        category_codes = codes(cardinality)
        category_codes[missing()] = -1
        levels = int(cardinality or rows)
        columns[f"c{i:02d}"] = pd.Categorical.from_codes(
            category_codes, categories=[f"k{j}" for j in range(levels)]
        )
    for i in range(BOOLEAN_COLUMNS):
        columns[f"b{i:02d}"] = rng.random(rows) < 0.05 + 0.13 * i
    for i in range(NULLABLE_BOOLEAN_COLUMNS):
        booleans = pd.array(rng.random(rows) < 0.5, dtype="boolean")
        booleans[missing()] = pd.NA
        columns[f"b{BOOLEAN_COLUMNS + i:02d}"] = booleans
    start = np.datetime64("2020-01-01T00:00:00", "ns")
    for i, cardinality in enumerate(DATETIME_CARDINALITIES):
        hours = codes(cardinality).astype("timedelta64[h]")
        times = start + hours.astype("timedelta64[ns]")
        times[missing()] = np.datetime64("NaT", "ns")
        if i in TIME_ZONES:
            utc = pd.Series(times).dt.tz_localize("UTC")
            columns[f"d{i:02d}"] = utc.dt.tz_convert(TIME_ZONES[i])
        else:
            columns[f"d{i:02d}"] = times
    return pd.DataFrame(columns)


def _text(codes: np.ndarray, cardinality: int | None) -> np.ndarray:
    """Text labels in row order, as read_csv allocates them.

    Repeated labels share one object each, as pandas' parser makes them, while
    each value of a column with many distinct values is its own object.
    """
    if cardinality is None:
        return np.array([f"id{code:07d}" for code in codes], dtype=object)
    if cardinality >= 10_000:
        return np.array([f"v{code}" for code in codes], dtype=object)
    labels = np.array([f"v{j}" for j in range(cardinality)], dtype=object)
    return labels[codes]


@pytest.fixture(scope="module")
def frame(benchmark_report):
    start = time.perf_counter()
    data = benchmark_frame()
    built = time.perf_counter() - start
    missing = float(data.isna().to_numpy().mean())
    benchmark_report.append(
        f"{data.shape[0]:,} rows x {data.shape[1]} columns, {missing:.2%} missing, "
        f"built in {built:.1f} s; Python {platform.python_version()}, numpy "
        f"{np.__version__}, pandas {pd.__version__}, pyarrow {pa.__version__}; "
        f"{THREADS} threads; {platform.system()} {platform.machine()}, "
        f"{_usable_cpus()} of {os.cpu_count()} CPUs"
    )
    yield data
    benchmark_report.append(f"Peak memory: {_peak_memory_gib():.2f} GiB")


@pytest.fixture(scope="module", autouse=True)
def two_threads():
    """Limit NumPy's BLAS and pyarrow to the target's 2 vCPUs."""
    cpus = pa.cpu_count()
    pa.set_cpu_count(THREADS)
    with threadpool_limits(limits=THREADS):
        yield
    pa.set_cpu_count(cpus)


def test_the_benchmark_frame_has_the_specified_columns(frame):
    text = str(pd.Series(["text"]).dtype)  # object on pandas 2, str on pandas 3
    floats = frame.select_dtypes("float64").to_numpy()

    assert frame.shape == (1_000_000, 100)
    assert frame.dtypes.astype(str).value_counts().to_dict() == {
        "float64": 25,
        "Int64": 15,
        text: 20,
        "category": 15,
        "bool": 7,
        "boolean": 3,
        "datetime64[ns]": 13,
        "datetime64[ns, UTC]": 1,
        "datetime64[ns, Europe/Berlin]": 1,
    }
    assert np.isnan(floats).any()
    assert np.isposinf(floats).any()
    assert np.isneginf(floats).any()
    # Text and categories from 2 distinct values up to all distinct.
    assert frame["t00"].nunique() == frame["c00"].nunique() == 2
    assert frame["t19"].nunique() == 1_000_000
    assert frame["c14"].nunique() == frame["c14"].count()
    assert 0.045 <= frame.isna().to_numpy().mean() <= 0.055


@pytest.mark.parametrize("name", ANALYSES)
def test_every_analysis_finishes_in_under_5_seconds(name, frame, benchmark_report):
    arguments = recipe_for(name).required(frame)
    assert arguments is not None, f"the recipe for {name} gives no call"
    analysis = getattr(ae, name)
    analysis(frame.head(1_000), **arguments)  # loads what the analysis imports

    start = time.perf_counter()
    res = analysis(frame, **arguments)
    seconds = time.perf_counter() - start

    benchmark_report.append(f"{name}: {seconds:.2f} s, {res.status}")
    assert seconds < LIMIT_SECONDS


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_a_rank_method_on_100_numeric_columns_stops_at_the_budget(
    method, benchmark_report
):
    # Ranking or sorting each of the 4,950 pairs' 1,000,000 rows would take
    # about 0.3 seconds a pair: the pair budget counts them by their cost.
    rng = np.random.default_rng(0)
    numeric = pd.DataFrame(rng.random((ROWS, 100), dtype=np.float32))

    start = time.perf_counter()
    res = ae.compute_association_matrix(numeric, numeric_method=method)
    seconds = time.perf_counter() - start

    benchmark_report.append(
        f"compute_association_matrix {method}, 100 numeric columns: {seconds:.2f} s, "
        f"{res.status}"
    )
    assert res.status == "budget_exceeded"
    assert res.status_reason is not None
    assert "counts as 100 pairs" in res.status_reason.message
    assert seconds < LIMIT_SECONDS


def _usable_cpus() -> int | None:
    """The CPUs this process may run on, which CI limits to 2 with taskset."""
    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _peak_memory_gib() -> float:
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20
