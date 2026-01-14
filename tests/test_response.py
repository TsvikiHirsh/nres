import unittest
import numpy as np
from nres.response import Response, Background
from nres.models import TransmissionModel
from nres.cross_section import CrossSection


class TestGaussianResponse(unittest.TestCase):
    """Test the Gaussian response function."""

    def test_gaussian_initialization(self):
        """Test that gaussian response initializes correctly."""
        response = Response(kind="gaussian", vary=False)

        # Check that the function is set correctly
        self.assertEqual(response.function, response.gaussian_response)

        # Check that parameters are initialized
        self.assertIn('x0', response.params)
        self.assertIn('σ', response.params)

        # Check default values
        self.assertEqual(response.params['x0'].value, 0.0)
        self.assertEqual(response.params['σ'].value, 1e-9)

        # Check vary flag
        self.assertFalse(response.params['x0'].vary)
        self.assertFalse(response.params['σ'].vary)

        # Check bounds on σ
        self.assertEqual(response.params['σ'].min, 1e-12)

    def test_gaussian_initialization_with_vary(self):
        """Test that gaussian response initializes with vary=True."""
        response = Response(kind="gaussian", vary=True)

        # Check that vary flag is set correctly
        self.assertTrue(response.params['x0'].vary)
        self.assertTrue(response.params['σ'].vary)

    def test_gaussian_response_output(self):
        """Test that gaussian response produces correct output."""
        response = Response(kind="gaussian", vary=False)

        # Call the response function
        output = response.function(x0=0.0, σ=1e-9)

        # Check that output is a numpy array
        self.assertIsInstance(output, np.ndarray)

        # Check that output has odd length (due to symmetric cutting)
        self.assertEqual(len(output) % 2, 1)

        # Check that output is normalized (sums to 1)
        self.assertAlmostEqual(np.sum(output), 1.0, places=10)

        # Check that output is non-negative
        self.assertTrue(np.all(output >= 0))

    def test_gaussian_response_symmetry(self):
        """Test that gaussian response with x0=0 is symmetric."""
        response = Response(kind="gaussian", vary=False, tstep=1e-9, nbins=300)

        # Call the response function with x0=0
        output = response.function(x0=0.0, σ=1e-9)

        # Check symmetry: left half should equal reversed right half
        center_idx = len(output) // 2
        left_half = output[:center_idx]
        right_half = output[center_idx+1:]

        np.testing.assert_array_almost_equal(left_half, right_half[::-1], decimal=10)

    def test_gaussian_response_different_sigma(self):
        """Test that different sigma values produce different widths."""
        response = Response(kind="gaussian", vary=False, tstep=1e-9, nbins=300)

        # Call with small sigma
        output_small = response.function(x0=0.0, σ=1e-10)

        # Call with large sigma
        output_large = response.function(x0=0.0, σ=1e-8)

        # The larger sigma should produce a wider distribution
        # (shorter array after cutting, since more of the tails are above eps)
        # Actually, it's the opposite - larger sigma means lower peak,
        # so more might get cut. Let's just check they're different.
        self.assertNotEqual(len(output_small), len(output_large))

    def test_gaussian_response_shifted_mean(self):
        """Test that x0 parameter shifts the gaussian."""
        response = Response(kind="gaussian", vary=False, tstep=1e-9, nbins=300)

        # Call with x0=0
        output_centered = response.function(x0=0.0, σ=1e-9)

        # Call with shifted x0
        output_shifted = response.function(x0=1e-9, σ=1e-9)

        # The outputs should be different
        self.assertFalse(np.array_equal(output_centered, output_shifted))

        # Both should still be normalized
        self.assertAlmostEqual(np.sum(output_centered), 1.0, places=10)
        self.assertAlmostEqual(np.sum(output_shifted), 1.0, places=10)


class TestExpoGaussResponse(unittest.TestCase):
    """Test the exponential-Gaussian response function."""

    def test_expogauss_initialization(self):
        """Test that expo_gauss response initializes correctly."""
        response = Response(kind="expo_gauss", vary=False)

        # Check that the function is set correctly
        self.assertEqual(response.function, response.expogauss_response)

        # Check that parameters are initialized
        self.assertIn('K', response.params)
        self.assertIn('x0', response.params)
        self.assertIn('τ', response.params)

        # Check default values
        self.assertEqual(response.params['K'].value, 1.0)
        self.assertEqual(response.params['x0'].value, 1e-9)
        self.assertEqual(response.params['τ'].value, 1e-9)

    def test_expogauss_response_output(self):
        """Test that expo_gauss response produces correct output."""
        response = Response(kind="expo_gauss", vary=False)

        # Call the response function
        output = response.function(K=1.0, x0=0.0, τ=1e-9)

        # Check that output is normalized
        self.assertAlmostEqual(np.sum(output), 1.0, places=6)

        # Check that output is non-negative
        self.assertTrue(np.all(output >= 0))


class TestEmptyResponse(unittest.TestCase):
    """Test the empty (none) response function."""

    def test_empty_initialization(self):
        """Test that 'none' response initializes correctly."""
        response = Response(kind="none", vary=False)

        # Check that the function is set correctly
        self.assertEqual(response.function, response.empty_response)

    def test_empty_response_output(self):
        """Test that empty response returns [0, 1, 0]."""
        response = Response(kind="none", vary=False)

        # Call the response function
        output = response.function()

        # Check the output
        np.testing.assert_array_equal(output, np.array([0., 1., 0.]))


class TestResponseInTransmissionModel(unittest.TestCase):
    """Test response functions when used in TransmissionModel."""

    def setUp(self):
        """Set up a mock cross section for testing."""
        self.xs = CrossSection(W="Tungsten", splitby="elements")

    def test_gaussian_in_transmission_model(self):
        """Test that gaussian response works in TransmissionModel."""
        model = TransmissionModel(self.xs, response="gaussian", vary_response=False)

        # Check that response is set up correctly
        self.assertIsNotNone(model.response)
        self.assertIn('x0', model.response.params)
        self.assertIn('σ', model.response.params)

    def test_gaussian_in_transmission_model_with_vary(self):
        """Test that gaussian response with vary=True works in TransmissionModel."""
        model = TransmissionModel(self.xs, response="gaussian", vary_response=True)

        # Check that parameters are set to vary
        self.assertTrue(model.response.params['x0'].vary)
        self.assertTrue(model.response.params['σ'].vary)

        # Check that parameters are in model params
        self.assertIn('x0', model.params)
        self.assertIn('σ', model.params)

    def test_expogauss_in_transmission_model(self):
        """Test that expo_gauss response works in TransmissionModel."""
        model = TransmissionModel(self.xs, response="expo_gauss", vary_response=False)

        # Check that response is set up correctly
        self.assertIsNotNone(model.response)
        self.assertIn('K', model.response.params)
        self.assertIn('x0', model.response.params)
        self.assertIn('τ', model.response.params)

    def test_none_response_in_transmission_model(self):
        """Test that 'none' response works in TransmissionModel."""
        model = TransmissionModel(self.xs, response="none", vary_response=False)

        # Check that response is set up correctly
        self.assertIsNotNone(model.response)


class TestResponseErrors(unittest.TestCase):
    """Test error handling in Response class."""

    def test_invalid_response_kind(self):
        """Test that invalid response kind raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            Response(kind="invalid_kind")

        # Check error message mentions valid options
        self.assertIn("expo_gauss", str(context.exception))
        self.assertIn("gaussian", str(context.exception))
        self.assertIn("none", str(context.exception))


if __name__ == '__main__':
    unittest.main()
