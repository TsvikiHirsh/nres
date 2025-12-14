import numpy as np
import pandas as pd
import tempfile
import os
from nres import CrossSection, TransmissionModel, Data


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
        vary_response=False
    )

    # Modify some parameters
    model.params['thickness'].value = 0.5
    model.params['norm'].value = 0.95

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_file = f.name

    try:
        # Save the model
        model.save(temp_file)

        # Load the model
        loaded_model = TransmissionModel.load(temp_file)

        # Verify the loaded model has the same parameters
        assert loaded_model.params['thickness'].value == model.params['thickness'].value
        assert loaded_model.params['norm'].value == model.params['norm'].value
        assert loaded_model.n == model.n

        # Verify cross-section is preserved
        assert loaded_model.cross_section.name == model.cross_section.name

        # Verify response and background objects exist
        assert loaded_model.response is not None
        assert loaded_model.background is not None

        # Verify response and background parameters match
        if model.response and model.response.params:
            for param_name in model.response.params:
                assert loaded_model.response.params[param_name].value == model.response.params[param_name].value

        if model.background and model.background.params:
            for param_name in model.background.params:
                assert loaded_model.background.params[param_name].value == model.background.params[param_name].value

        print("✓ Model save/load test passed")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_result_save_load():
    """Test saving and loading a fit result."""
    # Create a simple cross-section using a material from the database
    xs = CrossSection(Ni="Ni", splitby="isotopes")

    # Create a model
    model = TransmissionModel(
        cross_section=xs,
        response="expo_gauss",
        background="polynomial3",
        vary_weights=False,
        vary_background=True,
        vary_response=False
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
    data = pd.DataFrame({
        'energy': E,
        'trans': T_data,
        'err': err
    })

    # Fit the model
    result = model.fit(data, emin=1e5, emax=1e7, progress_bar=False)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_file = f.name

    try:
        # Save the result
        result.save(temp_file)

        # Load the result
        loaded_model, loaded_result = TransmissionModel.load_result(temp_file)

        # Verify the loaded result has the same fit parameters
        for param_name in result.params:
            assert abs(loaded_result.params[param_name].value - result.params[param_name].value) < 1e-10

        # Verify fit statistics
        assert loaded_result.redchi == result.redchi
        assert loaded_result.chisqr == result.chisqr

        # Verify the plot method is attached
        assert hasattr(loaded_result, 'plot')
        assert callable(loaded_result.plot)

        # Verify save method is attached
        assert hasattr(loaded_result, 'save')
        assert callable(loaded_result.save)

        print("✓ Result save/load test passed")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_model_with_fit_result_save_load():
    """Test saving and loading a model that contains a fit result."""
    # Create a simple cross-section using a material from the database
    xs = CrossSection(Cu="Cu", splitby="isotopes")

    # Create a model
    model = TransmissionModel(
        cross_section=xs,
        response="expo_gauss",
        background="polynomial3",
        vary_weights=False,
        vary_background=True,
        vary_response=False
    )

    # Create synthetic data
    E = np.logspace(5, 7, 100)
    np.random.seed(123)
    T_true = model.eval(E=E)
    noise = np.random.normal(0, 0.005, size=T_true.shape)
    T_data = T_true + noise
    err = np.ones_like(T_data) * 0.005

    data = pd.DataFrame({
        'energy': E,
        'trans': T_data,
        'err': err
    })

    # Fit the model
    result = model.fit(data, emin=1e5, emax=1e7, progress_bar=False)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_file = f.name

    try:
        # Save the model (which now contains fit_result)
        model.save(temp_file)

        # Load the model
        loaded_model = TransmissionModel.load(temp_file)

        # Verify the loaded model has the fit result
        assert hasattr(loaded_model, 'fit_result')
        assert loaded_model.fit_result.redchi == result.redchi

        # Verify parameters match
        for param_name in result.params:
            assert abs(loaded_model.fit_result.params[param_name].value -
                      result.params[param_name].value) < 1e-10

        print("✓ Model with fit result save/load test passed")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == '__main__':
    test_model_save_load()
    test_result_save_load()
    test_model_with_fit_result_save_load()
    print("\nAll save/load tests passed! ✓")
