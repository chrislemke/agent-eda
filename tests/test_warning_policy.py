import warnings

import pytest


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
