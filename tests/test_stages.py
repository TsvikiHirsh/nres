import unittest
import numpy as np
import pandas as pd
from nres.models import TransmissionModel
from nres.cross_section import CrossSection
from nres import materials


class TestStages(unittest.TestCase):
    """Test the model.stages property and stages_summary functionality"""

    def setUp(self):
        """Set up the model with a real cross section for testing."""
        self.pe = materials["Polyethylene, Non-borated"]
        self.xs = CrossSection(pe=self.pe, splitby="materials")

    def test_stages_initialized_correctly(self):
        """Test that _stages is initialized based on vary_* parameters"""
        # Model with all vary flags set to True
        model = TransmissionModel(self.xs, vary_background=True, vary_response=True, vary_tof=True)

        # Check that stages includes the expected groups
        self.assertIn("basic", model.stages)
        self.assertIn("background", model.stages)
        self.assertIn("response", model.stages)
        self.assertIn("tof", model.stages)

    def test_stages_respects_vary_false(self):
        """Test that stages dictionary doesn't include groups with vary=False"""
        # Model with vary_background=False
        model = TransmissionModel(self.xs, vary_background=False, vary_response=False)

        # Background and response should not be in stages
        self.assertNotIn("background", model.stages)
        self.assertNotIn("response", model.stages)

        # Basic should always be included
        self.assertIn("basic", model.stages)

    def test_stages_getter(self):
        """Test that the stages getter returns the _stages dictionary"""
        model = TransmissionModel(self.xs, vary_background=True)

        stages = model.stages
        self.assertIsInstance(stages, dict)
        self.assertEqual(stages, model._stages)

    def test_stages_setter_with_dict(self):
        """Test setting stages with a dictionary"""
        model = TransmissionModel(self.xs, vary_background=True, vary_response=True)

        # Set custom stages
        custom_stages = {
            "Stage 1": ["norm", "thickness"],
            "Stage 2": "background",
            "Stage 3": ["b0", "b1"]
        }
        model.stages = custom_stages

        self.assertEqual(model.stages, custom_stages)

    def test_stages_setter_with_all(self):
        """Test setting stages to 'all'"""
        model = TransmissionModel(self.xs, vary_background=True)

        model.stages = "all"
        self.assertEqual(model.stages, {"all": "all"})

    def test_stages_setter_invalid_string(self):
        """Test that setter raises error for invalid string"""
        model = TransmissionModel(self.xs)

        with self.assertRaises(ValueError) as cm:
            model.stages = "invalid"

        self.assertIn("must be 'all'", str(cm.exception))

    def test_stages_setter_invalid_type(self):
        """Test that setter raises error for invalid type"""
        model = TransmissionModel(self.xs)

        with self.assertRaises(ValueError) as cm:
            model.stages = ["list", "not", "dict"]

        self.assertIn("must be a string", str(cm.exception))

    def test_stages_setter_invalid_group_name(self):
        """Test that setter raises error for invalid group name"""
        model = TransmissionModel(self.xs, vary_background=True)

        with self.assertRaises(ValueError) as cm:
            model.stages = {"Stage 1": "invalid_group"}

        self.assertIn("valid group name", str(cm.exception))

    def test_stages_setter_validates_stage_names(self):
        """Test that setter validates stage names are strings"""
        model = TransmissionModel(self.xs)

        with self.assertRaises(ValueError) as cm:
            model.stages = {123: ["norm"]}

        self.assertIn("Stage names must be strings", str(cm.exception))

    def test_stages_summary_after_rietveld_fit(self):
        """Test that stages_summary is available after rietveld fit"""
        # Create synthetic data
        np.random.seed(42)
        energy = np.logspace(5, 7, 50)
        model_temp = TransmissionModel(self.xs)
        true_trans = model_temp.eval(E=energy, thickness=1.0, norm=1.0)
        noise = np.random.normal(0, 0.01, len(true_trans))
        trans_data = true_trans + noise
        err_data = np.full_like(trans_data, 0.01)

        data = pd.DataFrame({
            'energy': energy,
            'trans': trans_data,
            'err': err_data
        })

        # Fit with rietveld method
        model = TransmissionModel(self.xs, vary_background=True, vary_response=False)
        result = model.fit(data, method="rietveld", emin=1e5, emax=1e7, progress_bar=False)

        # Check that stages_summary exists
        self.assertTrue(hasattr(model, 'stages_summary'))
        self.assertTrue(hasattr(result, 'stages_summary'))

        # Check that it's a DataFrame
        self.assertIsInstance(model.stages_summary, (pd.DataFrame, pd.io.formats.style.Styler))

    def test_get_stages_summary_table_method(self):
        """Test the get_stages_summary_table method"""
        # Create synthetic data
        np.random.seed(42)
        energy = np.logspace(5, 7, 50)
        model_temp = TransmissionModel(self.xs)
        true_trans = model_temp.eval(E=energy, thickness=1.0, norm=1.0)
        noise = np.random.normal(0, 0.01, len(true_trans))
        trans_data = true_trans + noise
        err_data = np.full_like(trans_data, 0.01)

        data = pd.DataFrame({
            'energy': energy,
            'trans': trans_data,
            'err': err_data
        })

        # Fit with rietveld method
        model = TransmissionModel(self.xs, vary_background=True, vary_response=False)
        result = model.fit(data, method="rietveld", emin=1e5, emax=1e7, progress_bar=False)

        # Get summary table
        summary = model.get_stages_summary_table()

        # Should be a DataFrame or Styler
        self.assertIsInstance(summary, (pd.DataFrame, pd.io.formats.style.Styler))

    def test_get_stages_summary_raises_without_fit(self):
        """Test that get_stages_summary_table raises error if no rietveld fit was done"""
        model = TransmissionModel(self.xs)

        with self.assertRaises(ValueError) as cm:
            model.get_stages_summary_table()

        self.assertIn("No stages summary available", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
