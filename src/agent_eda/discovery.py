"""Functions that help an agent find the right analysis."""

from agent_eda._registry import registered
from agent_eda.results import AnalysisInfo, AnalysisListing


def list_analyses() -> AnalysisListing:
    """List every analysis with its module, one-line summary and masking support.

    Printing the listing shows one line per analysis, "name: summary", grouped
    under its module; the summary is the question the analysis answers. The
    `analyses` field holds what each analysis declares (parameters, methods,
    statuses, finding codes, result class and work limits), and `to_frame()`
    returns the same as a DataFrame. Listing never imports SciPy, statsmodels or
    scikit-learn.

    Returns:
        AnalysisListing: The analyses, ordered by module, then name.

    Examples:
        >>> import agent_eda as ae
        >>> listing = ae.list_analyses()
        >>> [info] = [i for i in listing.analyses if i.name == "describe_dataset"]
        >>> print(info.module, info.summary)
        agent_eda.data.provenance What does this dataset contain?
    """
    return AnalysisListing(
        analyses=[
            AnalysisInfo(
                name=spec.name,
                module=spec.module,
                summary=spec.summary,
                supports_masking=spec.supports_masking,
                tags=list(spec.tags),
                parameters=list(spec.parameters),
                methods=list(spec.methods),
                statuses=list(spec.statuses),
                finding_codes=list(spec.finding_codes),
                work_limit_parameters=list(spec.work_limits),
                result_class=spec.result_class.__name__,
            )
            for spec in registered()
        ]
    )
