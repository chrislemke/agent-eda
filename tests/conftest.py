"""Settings and fixtures shared by the whole suite."""

import pytest
from hypothesis import settings

# Derandomized, so every run generates the same examples and never depends on
# luck. No deadline: the first call into pandas or pyarrow in a process is slow,
# so a deadline would fail at random.
settings.register_profile(
    "suite", derandomize=True, deadline=None, max_examples=50, print_blob=True
)
# The same, with more examples: uv run pytest --hypothesis-profile=thorough
settings.register_profile("thorough", settings.get_profile("suite"), max_examples=1000)
settings.load_profile("suite")

_BENCHMARK = pytest.StashKey[list[str]]()


@pytest.fixture(scope="session")
def benchmark_report(pytestconfig: pytest.Config) -> list[str]:
    """Lines the benchmark prints at the end of the run."""
    return pytestconfig.stash.setdefault(_BENCHMARK, [])


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    lines = config.stash.get(_BENCHMARK, [])
    if lines:
        terminalreporter.section("benchmark")
        for line in lines:
            terminalreporter.write_line(line)
