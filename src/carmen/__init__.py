from .carmen import CarbonCycle

# For versioning with poetry-dynamic-versioning
try:
    from importlib.metadata import version
except ImportError:
    # For Python <3.8
    from importlib_metadata import version

__version__ = version("carmen")