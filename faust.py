"""Public import shim for Faust game scripts.

The intended game-facing style is:
    from faust import engine
"""

from importlib import import_module

engine = import_module("engine")

__all__ = ["engine"]
