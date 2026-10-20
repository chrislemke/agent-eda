"""Result classes returned by agent-eda analyses.

Every analysis returns a result class named after it, such as
SummarizeNumericResult, a subclass of AnalysisResult; the records that
ae.list_analyses(), ae.search_analyses() and ae.describe_analysis() return are
here too. `ae.results.<Name>` and `from agent_eda.results import <Name>` give a
class; the module loads a class's definition on first use, so importing it
stays cheap. The classes are defined in private modules next to the analyses
that return them, so a class's `__module__` is not `agent_eda.results`.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eda._discovery_results import (
        AnalysisDescription,
        AnalysisInfo,
        AnalysisListing,
        FindingCodeDescription,
        LimitationCodeDescription,
        ParameterDescription,
        ReasonCodeDescription,
        ResultFieldDescription,
        SearchListing,
        SearchMatch,
        SearchTerm,
        StatusDescription,
    )
    from agent_eda._result_base import (
        AnalysisResult,
        Assessment,
        ColumnFlag,
        ColumnResult,
        ColumnSupport,
        ExclusionReason,
        Execution,
        Finding,
        Limitation,
        LimitationCode,
        LogicalType,
        MaskableResult,
        Masking,
        ResultModel,
        RowId,
        Status,
        StatusReason,
        StatusReasonCode,
        Support,
    )
    from agent_eda._written_form import (
        row_id_json,
    )
    from agent_eda.data._duplicates_results import (
        DuplicateGroup,
        FindDuplicateRowsResult,
    )
    from agent_eda.data._missingness_results import (
        DeclaredSentinelValue,
        DetectMissingSentinelsResult,
        ProfileMissingValuesColumn,
        ProfileMissingValuesResult,
        ProfileMissingValuesRowCount,
        SentinelCandidate,
        SentinelEvidence,
        SentinelValues,
    )
    from agent_eda.data._structure_results import (
        CandidateKey,
        DescribeDatasetColumn,
        DescribeDatasetIndex,
        DescribeDatasetResult,
        FindCandidateKeysResult,
        KeySearch,
        Provenance,
    )
    from agent_eda.data._targets_results import (
        ProfileTargetClass,
        ProfileTargetResult,
        ProfileTargetStatistics,
        ProfileTargetUnexpectedLabel,
    )
    from agent_eda.data._validation_results import (
        ColumnEvidence,
        InferColumnSemanticsColumn,
        InferColumnSemanticsResult,
        InvalidValue,
        LogicalTypeCandidate,
        ValidateColumnTypesColumn,
        ValidateColumnTypesResult,
        ValueTypeCounts,
    )
    from agent_eda.statistics._association_results import (
        AssociationColumn,
        AssociationMethod,
        AssociationPair,
        AssociationType,
        ComputeAssociationMatrixResult,
        LeftOutColumn,
        LeftOutReason,
        UndefinedReason,
    )
    from agent_eda.statistics._descriptive_results import (
        SummarizeCategoricalColumn,
        SummarizeCategoricalLevel,
        SummarizeCategoricalResult,
        SummarizeCategoricalVariants,
        SummarizeNumericColumn,
        SummarizeNumericResult,
        level_label,
    )

__all__ = [
    "AnalysisDescription",
    "AnalysisInfo",
    "AnalysisListing",
    "AnalysisResult",
    "Assessment",
    "AssociationColumn",
    "AssociationMethod",
    "AssociationPair",
    "AssociationType",
    "CandidateKey",
    "ColumnEvidence",
    "ColumnFlag",
    "ColumnResult",
    "ColumnSupport",
    "ComputeAssociationMatrixResult",
    "DeclaredSentinelValue",
    "DescribeDatasetColumn",
    "DescribeDatasetIndex",
    "DescribeDatasetResult",
    "DetectMissingSentinelsResult",
    "DuplicateGroup",
    "ExclusionReason",
    "Execution",
    "FindCandidateKeysResult",
    "FindDuplicateRowsResult",
    "Finding",
    "FindingCodeDescription",
    "InferColumnSemanticsColumn",
    "InferColumnSemanticsResult",
    "InvalidValue",
    "KeySearch",
    "LeftOutColumn",
    "LeftOutReason",
    "Limitation",
    "LimitationCode",
    "LimitationCodeDescription",
    "LogicalType",
    "LogicalTypeCandidate",
    "MaskableResult",
    "Masking",
    "ParameterDescription",
    "ProfileMissingValuesColumn",
    "ProfileMissingValuesResult",
    "ProfileMissingValuesRowCount",
    "ProfileTargetClass",
    "ProfileTargetResult",
    "ProfileTargetStatistics",
    "ProfileTargetUnexpectedLabel",
    "Provenance",
    "ReasonCodeDescription",
    "ResultFieldDescription",
    "ResultModel",
    "RowId",
    "SearchListing",
    "SearchMatch",
    "SearchTerm",
    "SentinelCandidate",
    "SentinelEvidence",
    "SentinelValues",
    "Status",
    "StatusDescription",
    "StatusReason",
    "StatusReasonCode",
    "SummarizeCategoricalColumn",
    "SummarizeCategoricalLevel",
    "SummarizeCategoricalResult",
    "SummarizeCategoricalVariants",
    "SummarizeNumericColumn",
    "SummarizeNumericResult",
    "Support",
    "UndefinedReason",
    "ValidateColumnTypesColumn",
    "ValidateColumnTypesResult",
    "ValueTypeCounts",
    "level_label",
    "row_id_json",
]

# Each public name and the module that defines it. tests/test_results_facade.py
# checks that every result class of the package is listed.
_EXPORTS: dict[str, str] = {
    "AnalysisDescription": "agent_eda._discovery_results",
    "AnalysisInfo": "agent_eda._discovery_results",
    "AnalysisListing": "agent_eda._discovery_results",
    "AnalysisResult": "agent_eda._result_base",
    "Assessment": "agent_eda._result_base",
    "AssociationColumn": "agent_eda.statistics._association_results",
    "AssociationMethod": "agent_eda.statistics._association_results",
    "AssociationPair": "agent_eda.statistics._association_results",
    "AssociationType": "agent_eda.statistics._association_results",
    "CandidateKey": "agent_eda.data._structure_results",
    "ColumnEvidence": "agent_eda.data._validation_results",
    "ColumnFlag": "agent_eda._result_base",
    "ColumnResult": "agent_eda._result_base",
    "ColumnSupport": "agent_eda._result_base",
    "ComputeAssociationMatrixResult": "agent_eda.statistics._association_results",
    "DeclaredSentinelValue": "agent_eda.data._missingness_results",
    "DescribeDatasetColumn": "agent_eda.data._structure_results",
    "DescribeDatasetIndex": "agent_eda.data._structure_results",
    "DescribeDatasetResult": "agent_eda.data._structure_results",
    "DetectMissingSentinelsResult": "agent_eda.data._missingness_results",
    "DuplicateGroup": "agent_eda.data._duplicates_results",
    "ExclusionReason": "agent_eda._result_base",
    "Execution": "agent_eda._result_base",
    "FindCandidateKeysResult": "agent_eda.data._structure_results",
    "FindDuplicateRowsResult": "agent_eda.data._duplicates_results",
    "Finding": "agent_eda._result_base",
    "FindingCodeDescription": "agent_eda._discovery_results",
    "InferColumnSemanticsColumn": "agent_eda.data._validation_results",
    "InferColumnSemanticsResult": "agent_eda.data._validation_results",
    "InvalidValue": "agent_eda.data._validation_results",
    "KeySearch": "agent_eda.data._structure_results",
    "LeftOutColumn": "agent_eda.statistics._association_results",
    "LeftOutReason": "agent_eda.statistics._association_results",
    "Limitation": "agent_eda._result_base",
    "LimitationCode": "agent_eda._result_base",
    "LimitationCodeDescription": "agent_eda._discovery_results",
    "LogicalType": "agent_eda._result_base",
    "LogicalTypeCandidate": "agent_eda.data._validation_results",
    "MaskableResult": "agent_eda._result_base",
    "Masking": "agent_eda._result_base",
    "ParameterDescription": "agent_eda._discovery_results",
    "ProfileMissingValuesColumn": "agent_eda.data._missingness_results",
    "ProfileMissingValuesResult": "agent_eda.data._missingness_results",
    "ProfileMissingValuesRowCount": "agent_eda.data._missingness_results",
    "ProfileTargetClass": "agent_eda.data._targets_results",
    "ProfileTargetResult": "agent_eda.data._targets_results",
    "ProfileTargetStatistics": "agent_eda.data._targets_results",
    "ProfileTargetUnexpectedLabel": "agent_eda.data._targets_results",
    "Provenance": "agent_eda.data._structure_results",
    "ReasonCodeDescription": "agent_eda._discovery_results",
    "ResultFieldDescription": "agent_eda._discovery_results",
    "ResultModel": "agent_eda._result_base",
    "RowId": "agent_eda._result_base",
    "SearchListing": "agent_eda._discovery_results",
    "SearchMatch": "agent_eda._discovery_results",
    "SearchTerm": "agent_eda._discovery_results",
    "SentinelCandidate": "agent_eda.data._missingness_results",
    "SentinelEvidence": "agent_eda.data._missingness_results",
    "SentinelValues": "agent_eda.data._missingness_results",
    "Status": "agent_eda._result_base",
    "StatusDescription": "agent_eda._discovery_results",
    "StatusReason": "agent_eda._result_base",
    "StatusReasonCode": "agent_eda._result_base",
    "SummarizeCategoricalColumn": "agent_eda.statistics._descriptive_results",
    "SummarizeCategoricalLevel": "agent_eda.statistics._descriptive_results",
    "SummarizeCategoricalResult": "agent_eda.statistics._descriptive_results",
    "SummarizeCategoricalVariants": "agent_eda.statistics._descriptive_results",
    "SummarizeNumericColumn": "agent_eda.statistics._descriptive_results",
    "SummarizeNumericResult": "agent_eda.statistics._descriptive_results",
    "Support": "agent_eda._result_base",
    "UndefinedReason": "agent_eda.statistics._association_results",
    "ValidateColumnTypesColumn": "agent_eda.data._validation_results",
    "ValidateColumnTypesResult": "agent_eda.data._validation_results",
    "ValueTypeCounts": "agent_eda.data._validation_results",
    "level_label": "agent_eda.statistics._descriptive_results",
    "row_id_json": "agent_eda._written_form",
}

# Classes that a result pickled by agent-eda 0.1.0 names as
# `agent_eda.results.<Name>`, though they are not public.
_PICKLED: dict[str, str] = {
    "_ColumnList": "agent_eda._result_base",
    "_ColumnsByName": "agent_eda._result_base",
}

if not TYPE_CHECKING:
    # Hidden from type checkers, so that a misspelled attribute stays an error.

    def __getattr__(name: str) -> object:
        module = _EXPORTS.get(name) or _PICKLED.get(name)
        if module is None:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}", name=name
            )
        value = getattr(importlib.import_module(module), name)
        globals()[name] = value
        return value


def __dir__() -> list[str]:
    dunders = (name for name in globals() if name.startswith("__"))
    return sorted({*_EXPORTS, *dunders})
