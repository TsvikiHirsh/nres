#!/usr/bin/env python3
"""
Test script for the pick-one feature in nres stages language.

This script demonstrates the new "pick-one" feature that allows
automatic selection of the best-fitting isotope during Rietveld fitting.
"""

import nres
import numpy as np

print("=" * 60)
print("Testing pick-one feature in nres")
print("=" * 60)

# Load Silicon cross section with isotopes split
print("\n1. Loading Silicon cross-section with isotopes...")
xs = nres.CrossSection("Silicon", splitby="isotopes")
print(f"   Isotopes: {list(xs.weights.index)}")
print(f"   Natural abundances: {xs.weights.to_dict()}")

# Load example data
print("\n2. Loading silicon transmission data...")
try:
    data = nres.Data.from_transmission("silicon.dat")
    print(f"   Data loaded: {len(data.table)} points")
except Exception as e:
    print(f"   Warning: Could not load silicon.dat: {e}")
    print("   Creating synthetic data instead...")
    # Create synthetic data for testing
    energies = np.linspace(0.4e6, 1.7e6, 100)
    # Simple exponential decay with noise
    trans = np.exp(-xs.table.loc[energies, 'total'].values * 0.5 * 0.05) + np.random.normal(0, 0.01, len(energies))
    err = np.ones_like(trans) * 0.01
    import pandas as pd
    data = nres.Data(pd.DataFrame({'energy': energies, 'trans': trans, 'err': err}))
    print(f"   Synthetic data created: {len(data.table)} points")

# Create model with varying weights and background
print("\n3. Creating transmission model...")
model = nres.TransmissionModel(xs, vary_weights=True, vary_background=True)
print(f"   Parameters: {list(model.params.keys())}")

# Define stages with pick-one feature
print("\n4. Defining Rietveld stages with pick-one...")
stages = {
    "Stage 1: Basic": ["norm", "thickness"],
    "Stage 2: Background": ["background"],
    "Stage 3: Pick-one": ["weights", "pick-one"],  # This will test each isotope
}

print("   Stages defined:")
for stage_name, params in stages.items():
    print(f"     {stage_name}: {params}")

# Run the fit
print("\n5. Running Rietveld fit with pick-one...")
try:
    result = model.fit(
        data,
        method="rietveld",
        param_groups=stages,
        emin=0.4e6,
        emax=1.7e6,
        verbose=True
    )

    print("\n6. Results:")
    print(f"   Final reduced χ²: {result.redchi:.4f}")
    print(f"   Fitted parameters:")
    for param_name in ['thickness', 'norm', 'Si28', 'Si29', 'Si30']:
        if param_name in result.params:
            param = result.params[param_name]
            print(f"     {param_name}: {param.value:.6f}")

    # Check which isotope was selected
    print("\n7. Isotope selection results:")
    weights = {
        'Si28': result.params['Si28'].value,
        'Si29': result.params['Si29'].value,
        'Si30': result.params['Si30'].value
    }
    best_isotope = max(weights, key=weights.get)
    print(f"   Selected isotope: {best_isotope}")
    print(f"   Final weights:")
    for iso, weight in weights.items():
        print(f"     {iso}: {weight:.6f}")

    # Check if the pick-one actually selected one isotope
    max_weight = max(weights.values())
    if max_weight > 0.99:
        print("\n✓ Pick-one feature working correctly!")
        print(f"  {best_isotope} was selected with weight {max_weight:.4f}")
    else:
        print("\n⚠ Warning: No isotope has weight close to 1.0")
        print("  This may indicate an issue with the pick-one implementation")

    print("\n8. Accessing stages summary...")
    if hasattr(result, 'stages_summary'):
        print("   Stages summary available:")
        print(result.stages_summary)

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error during fitting: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 60)
    print("Test failed!")
    print("=" * 60)
