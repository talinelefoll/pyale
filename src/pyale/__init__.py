# src/pyale/__init__.py

# Import specific functions or classes to make them accessible directly from the package
from .ale_1d_calcu import calculate_ale_1d

# Optionally define the version of the package
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyale")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Default version if package metadata is not available



# # read version from installed package
# from importlib.metadata import version
# __version__ = version("pyale")