from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from nres import CrossSection, Data, TransmissionModel


def test_model_save_load():
    """Test saving and loading a TransmissionModel."""
    # Create a simple cross-section using a material from the database
    xs = CrossSection(Fe="Fe", splitby="isotopes")

    # Create a model
    model = TransmissionModel(
        cross_section=xs,
        response="expo_gauss",
        background="polynomial3",
        vary_weights=False,
        vary_background=False,
        vary_response=False,
    )

    # Modify some parameters
    model.params["thickness"].value = 0.5
    model.params["norm"].value = 0.95

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        # Save the model
        model.save(temp_file)

        # Load the model
        loaded_model = TransmissionModel.load(temp_file)

        # Verify the loaded model has the same parameters
        assert loaded_model.params["thickness"].value == model.params["thickness"].value
        assert loaded_model.params["norm"].value == model.params["norm"].value
        assert loaded_model.n == model.n

        # Verify cross-section is preserved
        assert loaded_model.cross_section.name == model.cross_section.name

        # Verify response and background objects exist
        assert loaded_model.response is not None
        assert loaded_model.background is not None

        # Verify response and background parameters match
        if model.response and model.response.params:
            for param_name in model.response.params:
                assert (
                    loaded_model.response.params[param_name].value
                    == model.response.params[param_name].value
                )

        if model.background and model.background.params:
            for param_name in model.background.params:
                assert (
                    loaded_model.background.params[param_name].value
                    == model.background.params[param_name].value
                )

        print("✓ Model save/load test passed")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_result_save_load():
    """Test saving and loading a fit result with full model."""
    # Create a simple cross-section using a material from the database
    xs = CrossSection(Ni="Ni", splitby="isotopes")

    # Create a model
    model = TransmissionModel(
        cross_section=xs,
        response="expo_gauss",
        background="polynomial3",
        vary_weights=False,
        vary_background=True,
        vary_response=False,
    )

    # Create synthetic data
    E = np.logspace(5, 7, 100)  # Energy range from 10^5 to 10^7 eV

    # Generate synthetic transmission data with some noise
    np.random.seed(42)
    T_true = model.eval(E=E)
    noise = np.random.normal(0, 0.01, size=T_true.shape)
    T_data = T_true + noise
    err = np.ones_like(T_data) * 0.01

    # Create a DataFrame
    data = pd.DataFrame({"energy": E, "trans": T_data, "err": err})

    # Fit the model
    result = model.fit(data, emin=1e5, emax=1e7, progress_bar=False)

    # Save to a temporary file (with full model)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        # Save the result with full model
        result.save(temp_file, include_model=True)

        # Load the result
        loaded_model, loaded_result_data = TransmissionModel.load_result(temp_file)

        # Verify the loaded result has the same fit parameters
        for param_name in result.params:
            assert (
                abs(
                    loaded_result_data["params"][param_name]["value"]
                    - result.params[param_name].value
                )
                < 1e-10
            )

        # Verify fit statistics
        assert abs(loaded_result_data["redchi"] - result.redchi) < 1e-10
        assert abs(loaded_result_data["chisqr"] - result.chisqr) < 1e-10

        # Verify model parameters were loaded correctly (should have fit values, not initial values)
        assert (
            abs(
                loaded_model.params["thickness"].value
                - result.params["thickness"].value
            )
            < 1e-10
        )
        assert (
            abs(loaded_model.params["norm"].value - result.params["norm"].value) < 1e-10
        )

        print("✓ Result save/load test passed (full model)")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_result_save_load_compressed():
    """Test saving and loading a compressed fit result (without model)."""
    # Create a simple cross-section using a material from the database
    xs = CrossSection(Cu="Cu", splitby="isotopes")

    # Create a model
    model = TransmissionModel(
        cross_section=xs,
        response="expo_gauss",
        background="polynomial3",
        vary_weights=False,
        vary_background=True,
        vary_response=False,
    )

    # Create synthetic data
    E = np.logspace(5, 7, 100)
    np.random.seed(123)
    T_true = model.eval(E=E)
    noise = np.random.normal(0, 0.005, size=T_true.shape)
    T_data = T_true + noise
    err = np.ones_like(T_data) * 0.005

    data = pd.DataFrame({"energy": E, "trans": T_data, "err": err})

    # Fit the model
    result = model.fit(data, emin=1e5, emax=1e7, progress_bar=False)

    # Save to a temporary file (compressed, without model)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        # Save the result without the model (compressed)
        result.save(temp_file, include_model=False)

        # Load the result (must provide the model)
        loaded_model, loaded_result_data = TransmissionModel.load_result(
            temp_file, model=model
        )

        # Verify the loaded result has the same fit parameters
        for param_name in result.params:
            assert (
                abs(
                    loaded_result_data["params"][param_name]["value"]
                    - result.params[param_name].value
                )
                < 1e-10
            )

        # Verify fit statistics
        assert abs(loaded_result_data["redchi"] - result.redchi) < 1e-10

        print("✓ Compressed result save/load test passed")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    test_model_save_load()
    test_result_save_load()
    test_result_save_load_compressed()
    print("\nAll save/load tests passed! ✓")
