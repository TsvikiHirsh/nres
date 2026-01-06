from __future__ import annotations

import unittest

import numpy as np

from nres import materials
from nres.cross_section import CrossSection
from nres.models import TransmissionModel
from nres.response import Background


class TestParameterValidation(unittest.TestCase):
    """Test that thickness and background parameters have proper validation to prevent negative values"""

    def setUp(self):
        """Set up the model with a real cross section for testing."""
        self.pe = materials["Polyethylene, Non-borated"]
        self.xs = CrossSection(pe=self.pe, splitby="materials")

    def test_thickness_has_min_bound(self):
        """Test that thickness parameter has min=0 bound"""
        model = TransmissionModel(self.xs)
        self.assertEqual(model.params["thickness"].min, 0.0)

    def test_norm_has_min_bound(self):
        """Test that norm parameter has min=0 bound"""
        model = TransmissionModel(self.xs)
        self.assertEqual(model.params["norm"].min, 0.0)

    def test_background_polynomial3_has_bounds(self):
        """Test that polynomial3 background parameters have proper bounds"""
        model = TransmissionModel(
            self.xs, vary_background=True, background="polynomial3"
        )
        self.assertEqual(model.params["b0"].min, -1e6)
        self.assertEqual(model.params["b0"].max, 1e6)
        self.assertEqual(model.params["b1"].min, -1e6)
        self.assertEqual(model.params["b1"].max, 1e6)
        self.assertEqual(model.params["b2"].min, -1e6)
        self.assertEqual(model.params["b2"].max, 1e6)

    def test_background_polynomial5_has_bounds(self):
        """Test that polynomial5 background parameters have proper bounds"""
        model = TransmissionModel(
            self.xs, vary_background=True, background="polynomial5"
        )
        for i in range(5):
            param_name = f"b{i}"
            self.assertEqual(model.params[param_name].min, -1e6)
            self.assertEqual(model.params[param_name].max, 1e6)

    def test_background_sample_dependent_has_bounds(self):
        """Test that sample_dependent background parameters have proper bounds"""
        model = TransmissionModel(
            self.xs, vary_background=True, background="sample_dependent"
        )
        self.assertEqual(model.params["b0"].min, -1e6)
        self.assertEqual(model.params["b0"].max, 1e6)
        self.assertEqual(model.params["b1"].min, -1e6)
        self.assertEqual(model.params["b1"].max, 1e6)
        self.assertEqual(model.params["b2"].min, -1e6)
        self.assertEqual(model.params["b2"].max, 1e6)
        self.assertEqual(model.params["k"].min, 0.0)
        self.assertEqual(model.params["k"].max, 10.0)

    def test_background_constant_has_bounds(self):
        """Test that constant background parameter has proper bounds"""
        model = TransmissionModel(self.xs, vary_background=True, background="constant")
        self.assertEqual(model.params["b0"].min, -1e6)
        self.assertEqual(model.params["b0"].max, 1e6)

    def test_background_class_polynomial3_has_bounds(self):
        """Test that Background class directly sets bounds for polynomial3"""
        bg = Background(kind="polynomial3", vary=True)
        self.assertEqual(bg.params["b0"].min, -1e6)
        self.assertEqual(bg.params["b0"].max, 1e6)
        self.assertEqual(bg.params["b1"].min, -1e6)
        self.assertEqual(bg.params["b1"].max, 1e6)
        self.assertEqual(bg.params["b2"].min, -1e6)
        self.assertEqual(bg.params["b2"].max, 1e6)

    def test_background_class_polynomial5_has_bounds(self):
        """Test that Background class directly sets bounds for polynomial5"""
        bg = Background(kind="polynomial5", vary=True)
        for i in range(5):
            param_name = f"b{i}"
            self.assertEqual(bg.params[param_name].min, -1e6)
            self.assertEqual(bg.params[param_name].max, 1e6)

    def test_background_class_constant_has_bounds(self):
        """Test that Background class directly sets bounds for constant"""
        bg = Background(kind="constant", vary=True)
        self.assertEqual(bg.params["b0"].min, -1e6)
        self.assertEqual(bg.params["b0"].max, 1e6)

    def test_background_class_sample_dependent_has_bounds(self):
        """Test that Background class directly sets bounds for sample_dependent"""
        bg = Background(kind="sample_dependent", vary=True)
        self.assertEqual(bg.params["b0"].min, -1e6)
        self.assertEqual(bg.params["b0"].max, 1e6)
        self.assertEqual(bg.params["b1"].min, -1e6)
        self.assertEqual(bg.params["b1"].max, 1e6)
        self.assertEqual(bg.params["b2"].min, -1e6)
        self.assertEqual(bg.params["b2"].max, 1e6)
        self.assertEqual(bg.params["k"].min, 0.0)
        self.assertEqual(bg.params["k"].max, 10.0)


if __name__ == "__main__":
    unittest.main()
