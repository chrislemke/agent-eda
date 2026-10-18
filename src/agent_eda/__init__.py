"""agent-eda: deterministic exploratory data analysis for LLM agents.

Call an analysis with a pandas DataFrame first and every other argument by
keyword, then print the result:

    import agent_eda as ae
    print(ae.describe_dataset(df))

ae.guide() prints where to start, ae.search_analyses("your question") finds an
analysis, ae.describe_analysis("name") shows how to call it and
ae.list_analyses() lists every analysis. Exceptions live in ae.errors and result
classes in ae.results. Attributes load on first use, so importing the package
stays cheap.
"""

import difflib
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eda import errors as errors
    from agent_eda import results as results
    from agent_eda._version import __version__
    from agent_eda.data.duplicates import find_duplicate_rows
    from agent_eda.data.provenance import describe_dataset
    from agent_eda.discovery import (
        describe_analysis,
        guide,
        list_analyses,
        search_analyses,
    )
    from agent_eda.results import AnalysisResult
    from agent_eda.statistics.descriptive import summarize_numeric

__all__ = [
    "AnalysisResult",
    "__version__",
    "describe_analysis",
    "describe_dataset",
    "find_duplicate_rows",
    "guide",
    "list_analyses",
    "search_analyses",
    "summarize_numeric",
]

# Each top-level export and the module that defines it. Every analysis is listed
# here; ae.list_analyses() imports these modules to find them.
_EXPORTS: dict[str, str] = {
    "AnalysisResult": "agent_eda.results",
    "__version__": "agent_eda._version",
    "describe_analysis": "agent_eda.discovery",
    "describe_dataset": "agent_eda.data.provenance",
    "find_duplicate_rows": "agent_eda.data.duplicates",
    "guide": "agent_eda.discovery",
    "list_analyses": "agent_eda.discovery",
    "search_analyses": "agent_eda.discovery",
    "summarize_numeric": "agent_eda.statistics.descriptive",
}

# Public submodules, also reachable as attributes such as ae.errors.
_SUBMODULES = ("errors", "results")

if not TYPE_CHECKING:
    # Hidden from type checkers, so that a misspelled attribute stays an error.

    def __getattr__(name: str) -> object:
        if name in _SUBMODULES:
            value = importlib.import_module(f"{__name__}.{name}")
        elif name in _EXPORTS:
            value = getattr(importlib.import_module(_EXPORTS[name]), name)
        else:
            closest = difflib.get_close_matches(name, [*__all__, *_SUBMODULES], n=1)
            hint = f" Did you mean {closest[0]!r}?" if closest else ""
            raise AttributeError(
                f"module 'agent_eda' has no attribute {name!r}.{hint} "
                "ae.list_analyses() lists every analysis.",
                name=name,
            )
        globals()[name] = value
        return value


def __dir__() -> list[str]:
    dunders = (name for name in globals() if name.startswith("__"))
    return sorted({*__all__, *_SUBMODULES, *dunders})
