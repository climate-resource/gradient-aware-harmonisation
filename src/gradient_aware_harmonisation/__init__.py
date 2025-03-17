"""
gradient aware harmonisation of timeseries
"""

import importlib.metadata

from gradient_aware_harmonisation import harmonise, utils

__version__ = importlib.metadata.version("gradient_aware_harmonisation")

__all__ = [
    "harmonise",
    "utils",
]
