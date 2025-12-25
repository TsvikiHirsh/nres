import unittest
import numpy as np
import pandas as pd
from nres.models import TransmissionModel
from nres.cross_section import CrossSection
from nres import materials


class TestRietveldVaryFalse(unittest.TestCase):
    """Test that rietveld method respects vary=False parameters"""

    def setUp(self):
        """Set up the model with a real cross section for testing."""
        self.pe = materials["Polyethylene, Non-borated"]
        self.xs = CrossSection(pe=self.pe, splitby="materials")

        # Create synthetic data for fitting
        np.random.seed(42)
        energy = np.logspace(5, 7, 100)

        # Create a simple transmission model
        model_temp = TransmissionModel(self.xs)
        true_trans = model_temp.eval(E=energy, thickness=1.0, norm=1.0)

        # Add noise
        noise = np.random.normal(0, 0.01, len(true_trans))
        trans_data = true_trans + noise
        err_data = np.full_like(trans_data, 0.01)

        self.data = pd.DataFrame({
            'energy': energy,
            'trans': trans_data,
            'err': err_data
        })

    def test_vary_false_background_not_in_default_stages(self):
        """Test that background parameters with vary=False are not included in default rietveld stages"""
        # Create model with vary_background=False
        model = TransmissionModel(self.xs, vary_background=False, vary_response=False)

        # Check that background parameters are not set to vary
        self.assertFalse(model.params['b0'].vary)
        self.assertFalse(model.params['b1'].vary)
        self.assertFalse(model.params['b2'].vary)

        # Perform rietveld fit with default stages
        result = model.fit(self.data, method="rietveld", emin=1e5, emax=1e7, progress_bar=False)

        # Check that background parameters were not varied during fitting
        # They should still have their initial values (0.0)
        self.assertAlmostEqual(result.params['b0'].value, 0.0, places=5)
        self.assertAlmostEqual(result.params['b1'].value, 0.0, places=5)
        self.assertAlmostEqual(result.params['b2'].value, 0.0, places=5)

    def test_vary_false_response_not_in_default_stages(self):
        """Test that response parameters with vary=False are not included in default rietveld stages"""
        # Create model with vary_response=False
        model = TransmissionModel(self.xs, vary_background=False, vary_response=False)

        # Store initial response parameter values
        initial_response_vals = {}
        if hasattr(model.response, 'params') and model.response.params:
            for param_name in model.response.params:
                if param_name in model.params:
                    self.assertFalse(model.params[param_name].vary)
                    initial_response_vals[param_name] = model.params[param_name].value

        # Perform rietveld fit
        result = model.fit(self.data, method="rietveld", emin=1e5, emax=1e7, progress_bar=False)

        # Response parameters should remain at their initial values
        for param_name, initial_val in initial_response_vals.items():
            final_val = result.params[param_name].value
            # Should be exactly the same since they weren't varied
            self.assertAlmostEqual(final_val, initial_val, places=5)

    def test_explicit_background_in_stages_still_skips_vary_false(self):
        """Test that even if 'background' group is explicitly in stages, vary=False params are skipped"""
        # Create model with vary_background=False
        model = TransmissionModel(self.xs, vary_background=False, vary_response=False)

        # Explicitly include 'background' in param_groups
        param_groups = ["basic", "background"]

        # Perform rietveld fit
        result = model.fit(self.data, method="rietveld", param_groups=param_groups,
                          emin=1e5, emax=1e7, progress_bar=False)

        # Background parameters should still not have been varied
        self.assertAlmostEqual(result.params['b0'].value, 0.0, places=5)
        self.assertAlmostEqual(result.params['b1'].value, 0.0, places=5)
        self.assertAlmostEqual(result.params['b2'].value, 0.0, places=5)

    def test_mixed_vary_true_and_false(self):
        """Test that some parameters can vary while others are fixed"""
        # Create model with vary_background=True but vary_response=False
        model = TransmissionModel(self.xs, vary_background=True, vary_response=False)

        # Check initial vary states
        self.assertTrue(model.params['b0'].vary)
        self.assertTrue(model.params['norm'].vary)
        self.assertTrue(model.params['thickness'].vary)

        # Perform rietveld fit
        param_groups = ["basic", "background"]
        result = model.fit(self.data, method="rietveld", param_groups=param_groups,
                          emin=1e5, emax=1e7, progress_bar=False)

        # Background parameters should have changed from initial values
        # (at least one should be different from 0.0)
        bg_changed = (abs(result.params['b0'].value) > 1e-6 or
                     abs(result.params['b1'].value) > 1e-6 or
                     abs(result.params['b2'].value) > 1e-6)
        self.assertTrue(bg_changed, "At least one background parameter should have changed")

        # Basic parameters should also have changed
        self.assertNotAlmostEqual(result.params['norm'].value, 1.0, places=3)

    def test_group_map_filters_vary_false_params(self):
        """Test that group_map in rietveld fit only includes parameters with vary=True"""
        # Create model with vary_background=False
        model = TransmissionModel(self.xs, vary_background=False, vary_response=False)

        # Access the model's internal group resolution by checking available params
        # The show_available_params method should reflect the filtering
        import io
        import sys

        # Capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        model.show_available_params(show_groups=True, show_params=False)
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # The background group should be empty or not show b0, b1, b2
        # since vary_background=False
        self.assertNotIn("'background': ['b0'", output)


if __name__ == "__main__":
    unittest.main()
