"""
nres: Simple yet powerful package for neutron resonance fitting
"""

from __future__ import annotations

from importlib.metadata import version

__all__ = ("__version__",)
__version__ = version(__name__)

import nres.utils as utils
from nres.cross_section import CrossSection
from nres.data import Data
from nres.grouped_fit import GroupedFitResult
from nres.models import TransmissionModel
from nres.response import Background, Response

materials, elements, isotopes = utils.load_materials_data()
