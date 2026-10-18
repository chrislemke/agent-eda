"""Smoke test of an installed agent-eda: print a result, then run every example.

CI's packaging job runs this script in a fresh environment that holds only the
six required dependencies at their minimum versions, after installing the wheel
with `pip install --no-deps`, as an offline AgentCore sandbox would. So an
analysis that needs anything else fails here. Run it with that environment's
Python: python -I tests/smoke.py
"""

import doctest
import sys

import pandas as pd

import agent_eda as ae

print(f"agent-eda {ae.__version__} from {ae.__file__}")
print(ae.describe_dataset(pd.DataFrame({"price": [1.5, 2.0, None]})))

# Each analysis's docstring examples print its result; the contract test runs
# them the same way.
finder = doctest.DocTestFinder(recurse=False)
runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
analyses = [info.name for info in ae.list_analyses().analyses]
for name in analyses:
    for test in finder.find(getattr(ae, name), name=name, globs={}):
        runner.run(test)  # reports each failing example
print(f"{len(analyses)} analyses ran {runner.tries} examples, {runner.failures} failed")
sys.exit(1 if runner.failures or not runner.tries else 0)
