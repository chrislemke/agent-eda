"""The installed version of agent-eda, read from the distribution's metadata.

importlib.metadata also finds the metadata inside a wheel file on sys.path.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("agent-eda")
except PackageNotFoundError:  # a source tree on sys.path, without metadata
    __version__ = "0+unknown"
