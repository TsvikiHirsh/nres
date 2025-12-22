# Parameter Loading Feature

The `TransmissionModel` now supports loading parameter values from a previous fit as initial guesses for a new fit. This is useful for sequential fitting workflows.

## Key Features

- **Values only**: Only parameter values are loaded; vary flags, bounds, and expressions are preserved from the new model's initialization
- **Selective loading**: Only parameters that exist in both source and target are updated
- **Flexible workflow**: Works with both regular and grouped fits
- **Clean API**: Simply pass `params=result.params` to the model constructor

## Usage Examples

### Basic Sequential Fitting

```python
from nres import TransmissionModel, CrossSection

# Create cross-section
xs = CrossSection(Fe56={"Fe-56": 1.0})

# First fit: fit background and TOF parameters
model1 = TransmissionModel(xs, vary_background=True, vary_tof=True)
result1 = model1.fit(data1)

# Second fit: use result1 as initial guess, but only fit background
# Note: vary_tof=False means TOF params won't vary, but their values
# will be initialized from result1
model2 = TransmissionModel(xs, vary_background=True, params=result1.params)
result2 = model2.fit(data2)

print(f"L0 from fit1: {result1.params['L0'].value}")
print(f"L0 in model2: {model2.params['L0'].value}")  # Same value, but vary=False
```

### Grouped Fitting with Reference Fit

```python
from nres import TransmissionModel, Data

# Load grouped data (e.g., spatial map)
data = Data.from_hdf("map_data.h5", group_by=['x', 'y'])

# First, fit a reference dataset with all parameters
reference_data = Data.from_hdf("reference.h5")
ref_model = TransmissionModel(xs, vary_background=True, vary_tof=True, vary_response=True)
ref_result = ref_model.fit(reference_data)

# Now fit the grouped data using reference parameters as initial guesses
# but only vary thickness (for thickness mapping)
map_model = TransmissionModel(xs, params=ref_result.params)  # All params from reference
map_results = map_model.fit(data, n_jobs=10)  # Parallel fitting

# Extract thickness map
thickness_map = map_results.get_param_map('thickness')
thickness_map.plot()
```

### Sequential Rietveld Refinement

```python
# Complex material with multiple isotopes
xs = CrossSection(
    Fe56={"Fe-56": 0.9172},
    Fe57={"Fe-57": 0.0212},
    Fe58={"Fe-58": 0.0028}
)

# Stage 1: Identify isotopes with pick-one
model1 = TransmissionModel(xs, vary_weights=True)
param_groups1 = {
    "Identify": ["basic", "pick-one"],
}
result1 = model1.fit(data, method="rietveld", param_groups=param_groups1)

# Stage 2: Use identified isotope weights as starting point,
# now refine background and TOF
model2 = TransmissionModel(xs, vary_background=True, vary_tof=True,
                          params=result1.params)  # Weights from pick-one
param_groups2 = {
    "Background": ["background"],
    "TOF": ["tof"]
}
result2 = model2.fit(data, method="rietveld", param_groups=param_groups2)
```

### Different Parameter Sets

```python
# Model 1: Only basic + background parameters
model1 = TransmissionModel(xs, vary_background=True)
result1 = model1.fit(data1)

# Model 2: Add TOF parameters, initialize from model1 where they overlap
# background params will be loaded, but TOF params start at defaults
model2 = TransmissionModel(xs, vary_background=True, vary_tof=True,
                          params=result1.params)
result2 = model2.fit(data2)

# Model1 had: thickness, norm, b0, b1, b2
# Model2 has: thickness, norm, b0, b1, b2, L0, t0 (new!)
# Only thickness, norm, b0, b1, b2 values are loaded from result1
# L0, t0 start at their default values
```

## Important Notes

1. **Vary flags are independent**: The `params` argument only affects initial values, not which parameters vary. Use the `vary_*` arguments to control what varies.

2. **Bounds and expressions preserved**: Min, max, and expression constraints from the new model are kept, only values change.

3. **Partial overlap is fine**: If source has parameters that target doesn't (or vice versa), only the common parameters are updated.

4. **Works with fit results**: You can pass `result.params` (from a fit) or `model.params` (from another model).

## Under the Hood

The `_load_param_values()` method:
```python
def _load_param_values(self, source_params):
    """Load parameter values from source, preserving vary, min, max, expr."""
    for param_name in self.params:
        if param_name in source_params:
            self.params[param_name].value = source_params[param_name].value
```

Simple and clean - only values are copied!
