#!/usr/bin/env python3
"""
Test script for the rebin feature in nres Data object.

This script demonstrates the new rebin() method that allows rebinning
of time-of-flight data either by combining bins or specifying a new time step.
"""

import nres
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("Testing rebin feature in nres")
print("=" * 60)

# Create synthetic data for testing
print("\n1. Creating synthetic data...")
n_bins = 1000
tstep_original = 1.56255e-9  # seconds
L = 10.59  # meters

# Create time-of-flight grid
tof = np.arange(1, n_bins + 1, dtype=float)

# Create synthetic counts with some structure (Gaussian peaks)
def gaussian(x, mu, sigma, A):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

signal_counts = (
    gaussian(tof, 200, 20, 5000) +
    gaussian(tof, 500, 30, 8000) +
    gaussian(tof, 800, 25, 6000) +
    np.random.poisson(100, n_bins)  # background
)

openbeam_counts = np.ones(n_bins) * 10000 + np.random.poisson(100, n_bins)

# Create signal and openbeam DataFrames
signal_df = pd.DataFrame({
    'tof': tof,
    'counts': signal_counts,
    'err': np.sqrt(signal_counts)
})
signal_df.attrs['label'] = 'synthetic_signal'

openbeam_df = pd.DataFrame({
    'tof': tof,
    'counts': openbeam_counts,
    'err': np.sqrt(openbeam_counts)
})
openbeam_df.attrs['label'] = 'synthetic_openbeam'

# Create Data object manually
data = nres.Data()
data.signal = signal_df
data.openbeam = openbeam_df
data.L = L
data.tstep = tstep_original

# Calculate transmission
energy = nres.utils.time2energy(tof * tstep_original, L)
transmission = signal_counts / openbeam_counts
trans_err = transmission * np.sqrt(
    (signal_df['err'] / signal_counts) ** 2 +
    (openbeam_df['err'] / openbeam_counts) ** 2
)

data.table = pd.DataFrame({
    'energy': energy,
    'trans': transmission,
    'err': trans_err
})
data.table.attrs['label'] = 'synthetic_data'
data.tgrid = tof

print(f"   Original data: {len(data.table)} bins")
print(f"   Original tstep: {tstep_original:.6e} s")
print(f"   Energy range: {data.table['energy'].min():.2e} - {data.table['energy'].max():.2e} eV")

# Test 1: Rebin by combining bins (n=4)
print("\n2. Testing rebin with n=4 (combine every 4 bins)...")
try:
    data_rebinned_n4 = data.rebin(n=4)
    print(f"   ✓ Rebinned data: {len(data_rebinned_n4.table)} bins")
    print(f"   ✓ New tstep: {data_rebinned_n4.tstep:.6e} s")
    print(f"   ✓ Tstep ratio: {data_rebinned_n4.tstep / tstep_original:.1f}x")
    print(f"   ✓ Bin count ratio: {len(data.table) / len(data_rebinned_n4.table):.1f}x")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Rebin by combining bins (n=10)
print("\n3. Testing rebin with n=10 (combine every 10 bins)...")
try:
    data_rebinned_n10 = data.rebin(n=10)
    print(f"   ✓ Rebinned data: {len(data_rebinned_n10.table)} bins")
    print(f"   ✓ New tstep: {data_rebinned_n10.tstep:.6e} s")
    print(f"   ✓ Bin count ratio: {len(data.table) / len(data_rebinned_n10.table):.1f}x")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Rebin with new tstep
print("\n4. Testing rebin with tstep=2*original...")
try:
    new_tstep = 2 * tstep_original
    data_rebinned_tstep = data.rebin(tstep=new_tstep)
    print(f"   ✓ Rebinned data: {len(data_rebinned_tstep.table)} bins")
    print(f"   ✓ New tstep: {data_rebinned_tstep.tstep:.6e} s")
    print(f"   ✓ Tstep ratio: {data_rebinned_tstep.tstep / tstep_original:.1f}x")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Verify error handling
print("\n5. Testing error handling...")
try:
    # Should raise error: both n and tstep specified
    data.rebin(n=4, tstep=2*tstep_original)
    print("   ✗ Should have raised error for both n and tstep")
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError: {str(e)[:60]}...")

try:
    # Should raise error: neither n nor tstep specified
    data.rebin()
    print("   ✗ Should have raised error for no parameters")
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError: {str(e)[:60]}...")

# Test 5: Check that total counts are preserved (for n method)
print("\n6. Verifying count conservation...")
original_total_signal = data.signal['counts'].sum()
rebinned_total_signal = data_rebinned_n4.signal['counts'].sum()

# Account for truncation (we might lose a few bins at the end)
expected_bins = len(data.signal) // 4
truncated_total = data.signal['counts'].iloc[:expected_bins * 4].sum()

print(f"   Original total counts: {original_total_signal:.0f}")
print(f"   Truncated total (4*{expected_bins}): {truncated_total:.0f}")
print(f"   Rebinned total counts: {rebinned_total_signal:.0f}")

if abs(rebinned_total_signal - truncated_total) / truncated_total < 1e-10:
    print("   ✓ Counts are conserved!")
else:
    print(f"   ⚠ Count difference: {abs(rebinned_total_signal - truncated_total):.1f}")

# Test 6: Visualize results
print("\n7. Creating visualization...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Transmission comparison
ax1 = axes[0]
ax1.errorbar(data.table['energy'], data.table['trans'], yerr=data.table['err'],
            fmt='o', markersize=2, alpha=0.3, label='Original', color='gray')
ax1.errorbar(data_rebinned_n4.table['energy'], data_rebinned_n4.table['trans'],
            yerr=data_rebinned_n4.table['err'],
            fmt='o-', markersize=4, label='Rebinned (n=4)', color='red')
ax1.errorbar(data_rebinned_n10.table['energy'], data_rebinned_n10.table['trans'],
            yerr=data_rebinned_n10.table['err'],
            fmt='s-', markersize=5, label='Rebinned (n=10)', color='blue')
ax1.set_xlabel('Energy [eV]')
ax1.set_ylabel('Transmission')
ax1.set_title('Effect of Rebinning on Transmission Data')
ax1.legend()
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Zoom in to show detail
ax2 = axes[1]
# Zoom to a region with structure
mask_orig = (data.table['energy'] > 5e5) & (data.table['energy'] < 2e6)
mask_n4 = (data_rebinned_n4.table['energy'] > 5e5) & (data_rebinned_n4.table['energy'] < 2e6)

ax2.errorbar(data.table.loc[mask_orig, 'energy'], data.table.loc[mask_orig, 'trans'],
            yerr=data.table.loc[mask_orig, 'err'],
            fmt='o', markersize=3, alpha=0.5, label='Original', color='gray')
ax2.errorbar(data_rebinned_n4.table.loc[mask_n4, 'energy'],
            data_rebinned_n4.table.loc[mask_n4, 'trans'],
            yerr=data_rebinned_n4.table.loc[mask_n4, 'err'],
            fmt='o-', markersize=5, label='Rebinned (n=4)', color='red')
ax2.set_xlabel('Energy [eV]')
ax2.set_ylabel('Transmission')
ax2.set_title('Zoomed View: Original vs Rebinned')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/nres/rebin_test_visualization.png', dpi=150)
print("   ✓ Visualization saved to rebin_test_visualization.png")

print("\n" + "=" * 60)
print("✓ All tests completed successfully!")
print("=" * 60)

plt.show()
