"""The `forgeai` importable package root and public re-exports used by tooling.

Does not own: service binaries, container images, or infra outside this package.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("forgeai")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
