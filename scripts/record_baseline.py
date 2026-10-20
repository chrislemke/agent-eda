"""Record, or check, the byte baseline that proves a restructuring changed nothing.

The baseline holds, for every analysis on every fixed and generated DataFrame:
the SHA-256 of `to_json()`, of `str(res)` and of every `to_frame()` table, and
the same for the discovery records, plus the MCP tool definitions and every
message of the message tables (tests/test_argument_messages.py,
tests/test_mcp_argument_messages.py and tests/test_mcp_fixes.py) in full. It
lives in tests/baseline/pandas<major>.json, one file per pandas major version,
since a record names the storage types pandas gives.

The baseline must not depend on where it runs: the digests are the same on every
operating system, CPU and patch release of Python, numpy and pandas that the
major versions allow. Every record names the versions it ran on
(execution.versions), so `without_environment` blanks their values before a
digest is taken; a table is written with "\n" line ends, which the operating
system doesn't choose; and a message names the temporary directory in a form
that `_stable` replaces on every operating system. One analysis needs more: the
numbers of compute_association_matrix's records are rounded to 10 significant
digits before a digest is taken (`_rounded`), since the last bits of what BLAS
sums depend on the CPU.

Refresh it, only when a change is meant to alter a record or a message:

    uv run python scripts/record_baseline.py            # rewrite this pandas's file
    uv run python scripts/record_baseline.py --check    # exit 1 where it differs
    uv run python scripts/record_baseline.py --dump DIR # write every full text to DIR

Refresh the pandas 2 file on the minimum versions:

    uv run --isolated --resolution lowest-direct --extra mcp --python 3.12 \
        python scripts/record_baseline.py

tests/test_baseline.py compares the baseline with a fresh run.
"""

import argparse
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import re
import sys
import tempfile
import typing
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
BASELINES = TESTS / "baseline"

Entry = dict[str, str]


def baseline_path() -> Path:
    """The baseline file for the installed pandas major version."""
    major = pd.__version__.split(".")[0]
    return BASELINES / f"pandas{major}.json"


_VERSIONS = re.compile(r'("versions"\s*:\s*\{)([^{}]*)(\})')
_STRING_VALUE = re.compile(r'(:\s*)"(?:[^"\\]|\\.)*"')


def without_environment(text: str) -> str:
    """The text with the version of each library that a record names blanked.

    A record names, in execution.versions, the versions of agent_eda, Python,
    numpy and pandas it ran on. Those differ between machines, the patch release
    of Python among them, so they stay out of a digest; the names stay in.
    """

    def blank(found: re.Match[str]) -> str:
        values = _STRING_VALUE.sub(r'\1"<version>"', found[2])
        return f"{found[1]}{values}{found[3]}"

    return _VERSIONS.sub(blank, text)


def _digest(text: str) -> str:
    return hashlib.sha256(
        without_environment(text).encode("utf-8", "surrogatepass")
    ).hexdigest()


# The records whose numbers hold only so many significant digits: they come from
# sums that run through the machine's BLAS, which adds in a different order on
# each CPU and so differs in the last bits, as 0.9999999999999996 from 1.0.
_BLAS_RECORDS = ("compute_association_matrix | ",)
_SIGNIFICANT_DIGITS = 10
_NUMBER = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?")


def _rounded(key: str, text: str) -> str:
    """The text of a record, with each decimal number of a BLAS record rounded
    to _SIGNIFICANT_DIGITS digits; any other record whole."""
    if not key.startswith(_BLAS_RECORDS):
        return text
    return _NUMBER.sub(lambda found: f"{float(found[0]):.{_SIGNIFICANT_DIGITS}g}", text)


def _table(frame: pd.DataFrame) -> str:
    """A table as text: its CSV, the dtypes of its columns and its index type.

    The CSV ends its lines with "\\n" on every operating system; pandas' default
    is os.linesep, which is "\\r\\n" on Windows.
    """
    dtypes = ", ".join(str(dtype) for dtype in frame.dtypes)
    csv = frame.to_csv(lineterminator="\n")
    return f"{csv}\n{dtypes}\n{type(frame.index).__name__}"


def _error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _record(res: Any) -> dict[str, str]:
    """Every output of a result as text: the JSON, the print, each table."""
    out: dict[str, str] = {}
    for key, make in (("json", res.to_json), ("text", lambda: str(res))):
        try:
            out[key] = make()
        except Exception as error:  # the baseline records any failure
            out[key] = _error(error)
    if not hasattr(res, "to_frame"):
        return out
    parameter = inspect.signature(res.to_frame).parameters.get("table")
    tables: list[str | None] = [None]
    if parameter is not None:
        tables = list(typing.get_args(parameter.annotation))
    for table in tables:
        try:
            frame = res.to_frame() if table is None else res.to_frame(table=table)
            out[f"table:{table}"] = _table(frame)
        except Exception as error:  # the baseline records any failure
            out[f"table:{table}"] = _error(error)
    return out


def _generated_frames() -> dict[str, pd.DataFrame]:
    """DataFrames from a seeded generator, so that they never change."""
    rng = np.random.default_rng(20261019)
    frames: dict[str, pd.DataFrame] = {}
    for number, rows in enumerate([5, 17, 60, 200]):
        numbers = rng.normal(size=rows)
        numbers[rng.random(rows) < 0.2] = np.nan
        if rows > 10:
            numbers[3] = np.inf
        labels = rng.choice(["a", "b", "c", None, "N/A", "-999"], size=rows)
        mixed = np.array(
            [rng.choice([1, "x", 2.5, None], p=[0.4, 0.3, 0.2, 0.1]) for _ in labels],
            dtype=object,
        )
        frames[f"generated {number}"] = pd.DataFrame(
            {
                "id": np.arange(rows),
                "count": rng.integers(-5, 5, size=rows),
                "amount": numbers,
                "label": labels,
                "flag": rng.random(rows) < 0.5,
                "when": pd.Timestamp("2024-01-01")
                + pd.to_timedelta(rng.integers(0, 400, size=rows), unit="D"),
                "mixed": mixed,
                "code": pd.Categorical(rng.choice(["p", "q", "r"], size=rows)),
                "text": pd.Series(
                    [f"v{int(i)}" for i in rng.integers(0, 6, size=rows)],
                    dtype="string",
                ),
                "constant": 7,
            },
            index=pd.Index([f"r{i}" for i in range(rows)])
            if number % 2
            else pd.RangeIndex(rows),
        )
    return frames


def _frames() -> dict[str, pd.DataFrame]:
    sys.path.insert(0, str(TESTS))
    strategies = importlib.import_module("strategies")
    frames = {name: make() for name, make in strategies.EDGE_FRAMES.items()}
    for name, make in strategies.LABEL_EDGE_FRAMES.items():
        frames[f"labels: {name}"] = make()
    return frames | _generated_frames()


def _analysis_records() -> dict[str, Entry]:
    sys.path.insert(0, str(TESTS))
    recipes = importlib.import_module("recipes")
    records: dict[str, Entry] = {}
    names = sorted(info.name for info in ae.list_analyses().analyses)
    for frame_name, frame in _frames().items():
        for name in names:
            arguments = recipes.recipe_for(name).required(frame)
            if arguments is None:
                continue
            variants = {"default": arguments}
            if issubclass(
                getattr(ae.results, ae.describe_analysis(name).result_class),
                ae.results.MaskableResult,
            ):
                sensitive = list(dict.fromkeys(str(label) for label in frame.columns))
                with contextlib.suppress(ae.errors.AgentEDAError):
                    variants["masked"] = {
                        **arguments,
                        "project": ae.Project(
                            data_policy=ae.DataPolicy(sensitive_columns=sensitive)
                        ),
                        "mask_sensitive": bool(sensitive),
                    }
            for variant, call in variants.items():
                key = f"{name} | {frame_name} | {variant}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        res = getattr(ae, name)(frame.copy(deep=True), **call)
                    except Exception as error:  # the baseline records any failure
                        records[key] = {"error": _error(error)}
                        continue
                records[key] = _record(res)
    return records


def _discovery_records() -> dict[str, Entry]:
    records: dict[str, Entry] = {}
    calls: dict[str, Callable[[], Any]] = {"list_analyses": ae.list_analyses}
    for info in ae.list_analyses().analyses:
        calls[f"describe_analysis({info.name})"] = lambda n=info.name: (
            ae.describe_analysis(n)
        )
    for query in ["missing values", "duplicate rows", "correlation", "zzzz", "target"]:
        calls[f"search_analyses({query})"] = lambda q=query: ae.search_analyses(q)
    calls["describe_analysis(unknown)"] = lambda: ae.describe_analysis("nothing")
    for key, call in calls.items():
        try:
            records[key] = _record(call())
        except Exception as error:  # the baseline records any failure
            records[key] = {"error": _error(error)}
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        ae.guide()
    records["guide"] = {"text": shown.getvalue()}
    return records


def _mcp_records() -> dict[str, Entry]:
    """The MCP tool definitions and one call of every tool; empty without mcp."""
    try:
        from mcp import Client

        from agent_eda.mcp import create_server
    except ImportError:
        return {}
    sys.path.insert(0, str(TESTS))
    recipes = importlib.import_module("recipes")
    shop = pd.DataFrame(
        {
            "order": ["A1", "A2", "A3", "A4", "A5", "A5"],
            "city": ["Oslo", "Lyon", "Oslo", "Oslo", None, None],
            "score": [1.5, 2.5, None, 4.0, 1.5, 1.5],
            "flag": [True, False, True, True, False, False],
        }
    )

    async def scenario(directory: Path) -> dict[str, Entry]:
        records: dict[str, Entry] = {}
        path = directory / "shop.csv"
        shop.to_csv(path, index=False)
        async with Client(create_server(), mode="legacy") as client:
            for tool in (await client.list_tools()).tools:
                records[f"tool: {tool.name}"] = {
                    "definition": json.dumps(
                        tool.model_dump(mode="json", by_alias=True), sort_keys=True
                    )
                }
                if tool.name in {"list_analyses", "guide"}:
                    arguments: dict[str, Any] | None = {}
                elif tool.name in {a.name for a in ae.list_analyses().analyses}:
                    arguments = recipes.recipe_for(tool.name).required(shop)
                    if arguments is not None:
                        arguments = {"data_path": str(path), **arguments}
                else:
                    arguments = None
                if arguments is None:
                    continue
                response = await client.call_tool(tool.name, arguments)
                records[f"call: {tool.name}"] = {
                    "error": str(response.is_error),
                    "text": "\n".join(getattr(b, "text", "") for b in response.content),
                    "structured": json.dumps(
                        response.structured_content, sort_keys=True, default=str
                    ),
                }
        return records

    with tempfile.TemporaryDirectory() as directory:
        return anyio.run(scenario, Path(directory))


def _message_records() -> dict[str, Entry]:
    """The code and message of every mistake in the three message tables."""
    sys.path.insert(0, str(TESTS))
    records: dict[str, Entry] = {}
    python = importlib.import_module("test_argument_messages")
    for name, (make, code, _fragments) in python.MISTAKES.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                make()
                records[f"python: {name}"] = {"message": "no error"}
            except Exception as error:  # the baseline records any failure
                records[f"python: {name}"] = {
                    "code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error).replace("WindowsPath", "PosixPath"),
                    "expected_code": code,
                }
    try:
        mcp_messages = importlib.import_module("test_mcp_argument_messages")
        mcp_fixes = importlib.import_module("test_mcp_fixes")
    except BaseException:  # Skipped, when the mcp extra is absent
        return records
    for name, (
        tool,
        arguments,
        kind,
        code,
        _fragments,
    ) in mcp_messages.MISTAKES.items():
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            filled = mcp_messages.arguments_of(directory, arguments, kind)
            response = anyio.run(mcp_messages.call, tool, filled)
            records[f"mcp: {name}"] = {
                "error": str(response.is_error),
                "code": str(response.structured_content.get("code")),
                "message": _stable(
                    str(response.structured_content.get("message")), directory
                ),
                "expected_code": code,
            }
    for name, (tool, arguments, _fixes) in mcp_fixes.MISTAKES.items():
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            path, data = mcp_fixes.data_file(directory, mcp_fixes.PARQUET.get(name))
            call = {"data_path": str(path), **arguments}
            project = directory / "project.toml"
            if name in mcp_fixes.PROJECT_FILES:
                project.write_text(mcp_fixes.PROJECT_FILES[name], encoding="utf-8")
                call["project_path"] = str(project)
            entry: dict[str, str] = {}
            try:
                mcp_fixes.call_in_python(tool, data, call)
                entry["python"] = "no error"
            except Exception as error:  # the baseline records any failure
                entry["python"] = _stable(str(error), directory)
            response = anyio.run(mcp_fixes.call, tool, call)
            entry["mcp"] = _stable(
                json.dumps(response.structured_content, sort_keys=True), directory
            )
            records[f"mcp fixes: {name}"] = entry
    return records


_PATH_TAIL = re.compile(r"(<tmp>|<cwd>)((?:(?:/|\\\\|\\(?=[\w.]))[\w.\-]+)+)")
_SEPARATOR = re.compile(r"\\\\|\\")


def _stable(text: str, directory: Path) -> str:
    """The text without the temporary directory a mistake ran in, and without
    the working directory a message names, the same on every operating system.

    A directory appears as its path, as that path inside a JSON string (where
    Windows' backslashes are doubled) and with forward slashes. What follows it
    in a Windows path reads with forward slashes too.
    """
    for place, name in ((directory, "<tmp>"), (Path.cwd(), "<cwd>")):
        for form in (str(place), json.dumps(str(place))[1:-1], place.as_posix()):
            text = text.replace(form, name)
    return _PATH_TAIL.sub(lambda found: found[1] + _SEPARATOR.sub("/", found[2]), text)


def collect(*, full: bool = False) -> dict[str, Entry]:
    """Every record of the baseline, as digests, or whole when `full`."""
    groups = (
        _analysis_records(),
        _discovery_records(),
        _mcp_records(),
        _message_records(),
    )
    records = {key: value for group in groups for key, value in group.items()}
    if full:
        return records
    # Messages stay whole, so that a change to one reads in the diff.
    return {
        key: value
        if key.startswith(("python:", "mcp:", "mcp fixes:"))
        else {part: _digest(_rounded(key, text)) for part, text in value.items()}
        for key, value in records.items()
    }


def differences(recorded: dict[str, Entry], fresh: dict[str, Entry]) -> list[str]:
    """What differs between two baselines, one line per record or part."""
    found = []
    for key in sorted(recorded.keys() | fresh.keys()):
        if key not in fresh:
            found.append(f"{key}: no longer recorded")
        elif key not in recorded:
            found.append(f"{key}: not in the baseline")
        elif recorded[key] != fresh[key]:
            old, new = recorded[key], fresh[key]
            assert isinstance(old, dict)
            assert isinstance(new, dict)
            for part in sorted(old.keys() | new.keys()):
                if old.get(part) != new.get(part):
                    found.append(
                        f"{key} [{part}]: {old.get(part)!r} -> {new.get(part)!r}"
                    )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare, don't write")
    parser.add_argument("--dump", type=Path, help="write every full text to a folder")
    options = parser.parse_args()
    if options.dump is not None:
        records = collect(full=True)
        options.dump.mkdir(parents=True, exist_ok=True)
        for key, value in records.items():
            name = _digest(key)[:12]
            (options.dump / f"{name}.json").write_text(
                json.dumps({"key": key, **value}, indent=1, ensure_ascii=False),
                encoding="utf-8",
                errors="surrogatepass",
            )
        print(f"{len(records)} records in {options.dump}")
        return 0
    fresh = collect()
    path = baseline_path()
    if options.check:
        recorded = json.loads(path.read_text(encoding="utf-8"))
        found = differences(recorded, fresh)
        print("\n".join(found) or f"{path.name} matches")
        return 1 if found else 0
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(fresh, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{len(fresh)} records in {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
