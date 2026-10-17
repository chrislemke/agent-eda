import doctest
import subprocess
import sys

import agent_eda as ae


def info_for(name):
    [info] = [a for a in ae.list_analyses().analyses if a.name == name]
    return info


def test_lists_each_analysis_with_its_module_summary_and_masking_support():
    info = info_for("describe_dataset")

    assert info.module == "agent_eda.data.provenance"
    assert info.summary == "What does this dataset contain?"
    assert info.supports_masking is False


def test_records_what_the_registration_decorator_declares():
    info = info_for("describe_dataset")

    assert info.parameters == ["data", "source", "source_version"]
    assert info.methods == []
    assert info.statuses == ["completed", "failed"]
    assert info.finding_codes == [
        "empty_data",
        "duplicated_column_labels",
        "colliding_column_labels",
        "multiindex_columns",
        "non_unique_index",
        "all_missing_columns",
        "constant_columns",
    ]
    assert info.result_class == "DescribeDatasetResult"
    assert info.work_limit_parameters == []
    assert "overview" in info.tags


def test_registered_analyses_are_skipped_by_pytest():
    assert ae.describe_dataset.__test__ is False  # ty: ignore[unresolved-attribute]  # set at registration


def test_lists_analyses_by_module_then_name():
    analyses = ae.list_analyses().analyses

    keys = [(a.module, a.name) for a in analyses]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_prints_one_line_per_analysis_under_its_module():
    listing = ae.list_analyses()

    lines = str(listing).splitlines()

    count = len(listing.analyses)
    assert lines[0] == f"{count} {'analysis' if count == 1 else 'analyses'}, by module:"
    assert "agent_eda.data.provenance" in lines
    assert "  describe_dataset: What does this dataset contain?" in lines
    modules = {a.module for a in listing.analyses}
    assert len(lines) == 1 + len(modules) + count
    assert repr(listing) == str(listing)


def test_reloading_an_analysis_module_keeps_one_entry_per_analysis():
    code = (
        "import importlib\n"
        "import agent_eda as ae\n"
        "import agent_eda.data.provenance as provenance\n"
        "importlib.reload(provenance)\n"
        "print([a.name for a in ae.list_analyses().analyses].count('describe_dataset'))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1"


def test_docstring_example_runs():
    runner = doctest.DocTestRunner()
    report = []
    finder = doctest.DocTestFinder(recurse=False)
    for test in finder.find(ae.list_analyses, name="list_analyses", globs={}):
        runner.run(test, out=report.append)

    assert runner.tries > 0
    assert runner.failures == 0, "".join(report)


def test_to_frame_has_one_row_per_analysis():
    listing = ae.list_analyses()

    frame = listing.to_frame()

    assert frame["name"].tolist() == [a.name for a in listing.analyses]
    assert {"name", "module", "summary", "supports_masking"} <= set(frame.columns)
