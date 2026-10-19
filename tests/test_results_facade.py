"""ae.results gives every result class by name, though each is defined elsewhere.

A class lives in a private module next to the analysis that returns it, so that
one module holds an analysis and its result types. ae.results is the public way
to reach any of them, as it was when one module held them all.
"""

import subprocess
import sys

import pytest

import agent_eda as ae


def loaded_subclasses(base):
    """Every class of agent_eda that derives from `base`, after the analyses load."""
    ae.list_analyses()  # imports every analysis module
    found, pending = {}, [base]
    while pending:
        for cls in pending.pop().__subclasses__():
            if cls.__module__.startswith("agent_eda.") and cls not in found.values():
                found[cls.__qualname__] = cls
                pending.append(cls)
    return found


def test_every_result_class_of_an_analysis_is_given_by_name():
    for cls in loaded_subclasses(ae.AnalysisResult).values():
        assert getattr(ae.results, cls.__name__) is cls, cls


def test_every_public_record_class_is_given_by_name():
    for qualname, cls in loaded_subclasses(ae.results.ResultModel).items():
        if not qualname.startswith("_"):
            assert getattr(ae.results, cls.__name__) is cls, cls


def test_every_name_the_module_lists_resolves_and_can_be_imported():
    for name in dir(ae.results):
        if name.startswith("__"):
            continue
        namespace: dict[str, object] = {}
        exec(  # the import statement under test, with a name from dir()
            f"from agent_eda.results import {name}", namespace
        )
        assert namespace[name] is getattr(ae.results, name)


def test_every_analysis_names_a_result_class_the_module_gives():
    for info in ae.list_analyses().analyses:
        cls = getattr(ae.results, info.result_class)
        assert issubclass(cls, ae.AnalysisResult)
        assert cls.__name__ == info.result_class


def test_a_name_the_module_does_not_have_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'NoSuchResult'"):
        ae.results.NoSuchResult  # noqa: B018 # ty: ignore[unresolved-attribute]  # the missing name is the test
    with pytest.raises(ImportError, match="NoSuchResult"):
        exec(
            "from agent_eda.results import NoSuchResult", {}
        )  # the import statement under test


def test_importing_the_module_defines_no_result_class_until_one_is_used():
    code = (
        "import sys, agent_eda.results as results\n"
        "def defined():\n"
        "    return [m for m in sys.modules if m.endswith('_results')]\n"
        "print(len(defined()))\n"
        "results.DescribeDatasetResult\n"
        "print(len(defined()))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    before, after = map(int, completed.stdout.split())
    assert before == 0
    assert after == 1


def test_a_result_class_is_defined_in_a_private_module_next_to_its_analysis():
    for info in ae.list_analyses().analyses:
        cls = getattr(ae.results, info.result_class)
        analysis = getattr(ae, info.name)
        package = analysis.__module__.rpartition(".")[0]
        module, _, last = cls.__module__.rpartition(".")
        assert module == package, info.name
        assert last.startswith("_"), info.name
