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

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eda import data as data
    from agent_eda import discovery as discovery
    from agent_eda import errors as errors
    from agent_eda import mcp as mcp
    from agent_eda import project as project
    from agent_eda import results as results
    from agent_eda import statistics as statistics
    from agent_eda._version import __version__
    from agent_eda.data.duplicates import find_duplicate_rows
    from agent_eda.data.missingness import (
        detect_missing_sentinels,
        profile_missing_values,
    )
    from agent_eda.data.structure import describe_dataset, find_candidate_keys
    from agent_eda.data.targets import profile_target
    from agent_eda.data.validation import (
        ColumnSpec,
        infer_column_semantics,
        validate_column_types,
    )
    from agent_eda.discovery import (
        describe_analysis,
        guide,
        list_analyses,
        search_analyses,
    )
    from agent_eda.project import DataPolicy, Project
    from agent_eda.results import AnalysisResult, LogicalType
    from agent_eda.statistics.association import compute_association_matrix
    from agent_eda.statistics.descriptive import (
        summarize_categorical,
        summarize_numeric,
    )

__all__ = [
    "AnalysisResult",
    "ColumnSpec",
    "DataPolicy",
    "LogicalType",
    "Project",
    "__version__",
    "compute_association_matrix",
    "describe_analysis",
    "describe_dataset",
    "detect_missing_sentinels",
    "find_candidate_keys",
    "find_duplicate_rows",
    "guide",
    "infer_column_semantics",
    "list_analyses",
    "profile_missing_values",
    "profile_target",
    "search_analyses",
    "summarize_categorical",
    "summarize_numeric",
    "validate_column_types",
]

# Each top-level export and the module that defines it. Every analysis is listed
# here; ae.list_analyses() imports these modules to find them.
_EXPORTS: dict[str, str] = {
    "AnalysisResult": "agent_eda.results",
    "ColumnSpec": "agent_eda.data.validation",
    "DataPolicy": "agent_eda.project",
    "LogicalType": "agent_eda.results",
    "Project": "agent_eda.project",
    "__version__": "agent_eda._version",
    "compute_association_matrix": "agent_eda.statistics.association",
    "describe_analysis": "agent_eda.discovery",
    "describe_dataset": "agent_eda.data.structure",
    "detect_missing_sentinels": "agent_eda.data.missingness",
    "find_candidate_keys": "agent_eda.data.structure",
    "find_duplicate_rows": "agent_eda.data.duplicates",
    "guide": "agent_eda.discovery",
    "infer_column_semantics": "agent_eda.data.validation",
    "list_analyses": "agent_eda.discovery",
    "profile_missing_values": "agent_eda.data.missingness",
    "profile_target": "agent_eda.data.targets",
    "search_analyses": "agent_eda.discovery",
    "summarize_categorical": "agent_eda.statistics.descriptive",
    "summarize_numeric": "agent_eda.statistics.descriptive",
    "validate_column_types": "agent_eda.data.validation",
}

# Public submodules, also reachable as attributes such as ae.errors, whether or
# not something imported them yet. ae.mcp loads without the mcp extra; its
# create_server() raises MissingDependencyError with the install command.
_SUBMODULES = (
    "data",
    "discovery",
    "errors",
    "mcp",
    "project",
    "results",
    "statistics",
)

if not TYPE_CHECKING:
    # Hidden from type checkers, so that a misspelled attribute stays an error.

    def __getattr__(name: str) -> object:
        if name in _SUBMODULES:
            value = importlib.import_module(f"{__name__}.{name}")
        elif name in _EXPORTS:
            value = getattr(importlib.import_module(_EXPORTS[name]), name)
        else:
            _unknown(name)
            from agent_eda._text import closest_names

            closest = closest_names(name, [*__all__, *_SUBMODULES])[:1]
            hint = f" Did you mean {closest[0]!r}?" if closest else ""
            raise AttributeError(
                f"module 'agent_eda' has no attribute {name!r}.{hint} "
                "ae.list_analyses() lists every analysis.",
                name=name,
            )
        globals()[name] = value
        return value

    def _unknown(name: str) -> None:
        """Point a name that lives in ae.errors or ae.results there."""
        for home in ("errors", "results"):
            module = importlib.import_module(f"{__name__}.{home}")
            if name.isidentifier() and name[:1].isupper() and hasattr(module, name):
                raise AttributeError(
                    f"module 'agent_eda' has no attribute {name!r}. It lives in "
                    f"ae.{home}: use ae.{home}.{name}.",
                    name=name,
                )


def __dir__() -> list[str]:
    dunders = (name for name in globals() if name.startswith("__"))
    return sorted({*__all__, *_SUBMODULES, *dunders})
