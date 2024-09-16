"""
nres: Simple yet powerful package for neutron resonance fitting
"""

from __future__ import annotations

from importlib.metadata import version

__all__ = ("__version__",)
__version__ = version(__name__)

from cross_section import CrossSection
from response import Response
from models import TransmissionModel
import utils
