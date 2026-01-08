from __future__ import annotations

import inspect

import pytest

from nres.models import TransmissionModel


class TestFitMethodDefault:
    """Test that the default fit method is rietveld"""

    def test_default_method_is_rietveld(self):
        """Test that the default fit method is 'rietveld' by checking function signature"""
        # Get the signature of the fit method
        sig = inspect.signature(TransmissionModel.fit)

        # Check that 'method' parameter has default value 'rietveld'
        assert "method" in sig.parameters, "fit method should have a 'method' parameter"
        method_param = sig.parameters["method"]
        assert (
            method_param.default == "rietveld"
        ), f"Default method should be 'rietveld', got {method_param.default}"

    def test_method_parameter_exists(self):
        """Test that the method parameter exists and is configurable"""
        sig = inspect.signature(TransmissionModel.fit)
        assert "method" in sig.parameters, "fit method should have a 'method' parameter"

        # Verify it has a default value (not EMPTY)
        method_param = sig.parameters["method"]
        assert (
            method_param.default != inspect.Parameter.empty
        ), "method parameter should have a default value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
