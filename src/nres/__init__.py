"""
nres: Simple yet powerful package for neutron resonance fitting
"""

from __future__ import annotations
from pathlib import Path
from importlib.metadata import version
import shelve

__all__ = ("__version__",)
__version__ = version(__name__)

from nres.cross_section import CrossSection
from nres.response import Response
from nres.models import TransmissionModel
import nres.utils as utils

materials, elements = utils.load_or_create_cache()