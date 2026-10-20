"""The one place a test registers an analysis the way a real one is registered."""

import contextlib
import inspect
from collections.abc import Generator
from typing import Annotated, Any

import pandas as pd
from pydantic import Field

from agent_eda._registry import _REGISTRY, analysis
from agent_eda.project import Project
from agent_eda.results import AnalysisResult, MaskableResult, Masking, Support


class CountedResult(AnalysisResult):
    """The stand-in result of an analysis that shows no raw values."""

    seen: int = Field(description="Rows seen.")


class SentinelsResult(MaskableResult):
    """The stand-in result of an analysis that takes declared values."""

    seen: int = Field(description="Rows seen.")


def _support(data: pd.DataFrame) -> Support:
    return Support(row_count=len(data), column_count=0, columns=[])


def register_stand_in(docstring: str) -> Any:
    """Register a stand-in analysis whose docstring is the one given, and
    unregister it again; registration raises when the docstring is broken.

    Returns the registry's entry for it, with the statuses and finding codes
    registration read from the docstring.
    """

    def stand_in_broken(data: pd.DataFrame) -> CountedResult:
        raise NotImplementedError

    stand_in_broken.__doc__ = docstring
    try:
        analysis()(stand_in_broken)
        return _REGISTRY["stand_in_broken"]
    finally:
        _REGISTRY.pop("stand_in_broken", None)


def decorator_options() -> list[str]:
    """The names of the options the registration decorator takes."""
    return list(inspect.signature(analysis).parameters)


@contextlib.contextmanager
def stand_in_analyses(
    *, forget_override: bool = False, unknown_override: bool = False
) -> Generator[dict[str, Any]]:
    """Register a counting analysis and a masking one, and unregister them on leaving.

    The masking analysis takes `sentinel_values`, a parameter that holds data
    values. It passes the masked ones to the result, or forgets to when
    `forget_override` is true. The counting analysis overrides a parameter it
    doesn't take when `unknown_override` is true.
    """

    @analysis()
    def stand_in_counted(
        data: pd.DataFrame, *, limit: Annotated[int, Field(ge=1)] = 5
    ) -> CountedResult:
        """How many rows does the stand-in see?

        Statuses:
            completed: Always.
        """
        return CountedResult.for_call(
            parameters={"no_such_parameter": 1} if unknown_override else None,
            status="completed",
            status_reason=None,
            support=_support(data),
            findings=[],
            limitations=[],
            seen=min(len(data), limit),
        )

    @analysis(masking=True)
    def stand_in_sentinels(
        data: pd.DataFrame,
        *,
        sentinel_values: dict[str, list[str]] | None = None,
        project: Project | None = None,
        mask_sensitive: bool | None = None,
    ) -> SentinelsResult:
        """Which declared values does the stand-in see?

        Statuses:
            completed: Always.
        """
        masking = Masking(
            applied=False,
            decided_by="default",
            columns=[],
            index_columns=[],
            absent_columns=[],
            near_miss_columns={},
        )
        overrides = (
            {}
            if forget_override or sentinel_values is None
            else {"sentinel_values": {name: ["<masked 1>"] for name in sentinel_values}}
        )
        return SentinelsResult.for_call(
            parameters=overrides,
            status="completed",
            status_reason=None,
            support=_support(data),
            findings=[],
            limitations=[],
            masking=masking,
            seen=len(data),
        )

    try:
        yield {"counted": stand_in_counted, "sentinels": stand_in_sentinels}
    finally:
        _REGISTRY.pop("stand_in_counted", None)
        _REGISTRY.pop("stand_in_sentinels", None)
