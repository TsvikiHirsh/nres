"""
nres: Simple yet powerful package for neutron resonance fitting
"""

from __future__ import annotations

from importlib.metadata import version

__all__ = ("__version__",)
__version__ = version(__name__)

from cross_section import Cross_section,total_xs
import utils
