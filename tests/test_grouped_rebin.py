"""
Tests for rebinning grouped data.

This module tests the rebinning functionality for grouped data,
including both n-binning and tstep-based rebinning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nres import Data, utils


class TestGroupedDataRebin:
    """Test rebinning of grouped data."""

    def test_grouped_rebin_with_n(self):
        """Test rebinning grouped data using n parameter."""
        # Create test grouped data
        tof = np.arange(1, 1001, dtype=float)
        L = 10.0
        tstep = 1e-6

        # Create 5 groups with different transmission patterns
        n_groups = 5
        trans_2d = np.zeros((n_groups, len(tof)))
        err_2d = np.zeros((n_groups, len(tof)))

        for i in range(n_groups):
            # Each group has slightly different transmission
            trans_2d[i, :] = 0.8 + 0.05 * np.sin(tof / 100 + i)
            err_2d[i, :] = 0.02

        # Create grouped data
        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=L, tstep=tstep
        )

        # Rebin with n=4
        rebinned = data.rebin(n=4)

        # Check that it's still grouped
        assert rebinned.is_grouped
        assert len(rebinned.indices) == n_groups

        # Check that the number of energy bins is reduced
        expected_n_bins = len(tof) // 4
        assert len(rebinned.table) == expected_n_bins

        # Check that all groups have the same number of bins
        for idx in rebinned.indices:
            assert len(rebinned.groups[idx]) == expected_n_bins

        # Check that grouped arrays have correct shape
        assert rebinned.grouped_trans.shape == (n_groups, expected_n_bins)
        assert rebinned.grouped_err.shape == (n_groups, expected_n_bins)

        # Check that transmission values are reasonable (should be close to original)
        for i in range(n_groups):
            # Transmission should still be in range [0, 1]
            assert np.all(rebinned.grouped_trans[i, :] >= 0)
            assert np.all(rebinned.grouped_trans[i, :] <= 1)
            # Should be close to original average
            assert (
                np.abs(rebinned.grouped_trans[i, :].mean() - trans_2d[i, :].mean())
                < 0.1
            )

    def test_grouped_rebin_with_tstep(self):
        """Test rebinning grouped data using tstep parameter."""
        # Create test grouped data
        tof = np.arange(1, 501, dtype=float)
        L = 10.0
        tstep = 1e-6

        # Create 3 groups
        n_groups = 3
        trans_2d = np.random.rand(n_groups, len(tof)) * 0.2 + 0.7
        err_2d = np.ones((n_groups, len(tof))) * 0.02

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=L, tstep=tstep
        )

        # Rebin with new tstep (2x original)
        new_tstep = 2 * tstep
        rebinned = data.rebin(tstep=new_tstep)

        # Check that tstep was updated
        assert rebinned.tstep == new_tstep

        # Check that it's still grouped
        assert rebinned.is_grouped
        assert len(rebinned.indices) == n_groups

        # Check that transmission values are still valid
        for i in range(n_groups):
            assert np.all(rebinned.grouped_trans[i, :] >= 0)
            assert np.all(rebinned.grouped_trans[i, :] <= 1)

    def test_grouped_rebin_preserves_indices(self):
        """Test that rebinning preserves group indices."""
        # Create grouped data with custom indices
        tof = np.arange(1, 201, dtype=float)
        trans_2d = np.random.rand(10, len(tof)) * 0.3 + 0.6
        err_2d = np.ones((10, len(tof))) * 0.03

        # 2D indices (5x2 grid)
        indices_2d = [(i, j) for i in range(5) for j in range(2)]

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=10.0, tstep=1e-6, indices=indices_2d
        )

        # Rebin
        rebinned = data.rebin(n=2)

        # Check that indices are preserved
        assert rebinned.indices == data.indices
        assert rebinned.group_shape == data.group_shape

        # Check that all groups are present
        for idx in indices_2d:
            assert idx in rebinned.groups

    def test_grouped_rebin_energy_grid(self):
        """Test that energy grid is correctly updated after rebinning."""
        tof = np.arange(1, 101, dtype=float)
        L = 10.0
        tstep = 1e-6

        trans_2d = np.random.rand(3, len(tof)) * 0.4 + 0.5
        err_2d = np.ones((3, len(tof))) * 0.02

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=L, tstep=tstep
        )

        # Get original energy range
        orig_energy_min = data.table["energy"].min()
        orig_energy_max = data.table["energy"].max()

        # Rebin
        rebinned = data.rebin(n=5)

        # Energy range should be positive
        assert rebinned.table["energy"].min() > 0
        assert rebinned.table["energy"].max() > rebinned.table["energy"].min()

        # Energy grid should be monotonic (descending for TOF data since E ∝ 1/t²)
        energy_diffs = np.diff(rebinned.table["energy"])
        # Check that all differences have the same sign (monotonic)
        assert np.all(energy_diffs < 0) or np.all(energy_diffs > 0)

    def test_grouped_rebin_errors_combined_correctly(self):
        """Test that uncertainties are properly combined during rebinning."""
        tof = np.arange(1, 201, dtype=float)
        n_groups = 4

        # Create data with known errors
        trans_2d = np.ones((n_groups, len(tof))) * 0.8
        err_2d = np.ones((n_groups, len(tof))) * 0.04  # Constant 4% error

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=10.0, tstep=1e-6
        )

        # Rebin with n=4
        rebinned = data.rebin(n=4)

        # For averaging, error should be reduced by sqrt(n) / n = 1 / sqrt(n)
        # err_mean = sqrt(sum(err^2)) / n = sqrt(n * err^2) / n = err / sqrt(n)
        expected_err = 0.04 / np.sqrt(4)

        # Check that errors are approximately as expected
        for i in range(n_groups):
            mean_err = rebinned.grouped_err[i, :].mean()
            assert np.abs(mean_err - expected_err) / expected_err < 0.01

    def test_grouped_rebin_1d_detector(self):
        """Test rebinning for 1D detector array (linear pixels)."""
        # Simulate a 1D linear detector with 20 pixels
        n_pixels = 20
        n_energy = 200

        tof = np.arange(1, n_energy + 1, dtype=float)
        trans_2d = np.random.rand(n_pixels, n_energy) * 0.5 + 0.4
        err_2d = np.ones((n_pixels, n_energy)) * 0.03

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=10.0, tstep=1e-6
        )

        # Rebin energy axis
        rebinned = data.rebin(n=10)

        # Should have same number of pixels but fewer energy bins
        assert rebinned.grouped_trans.shape[0] == n_pixels
        assert rebinned.grouped_trans.shape[1] == n_energy // 10

    def test_grouped_rebin_2d_detector(self):
        """Test rebinning for 2D imaging detector."""
        # Simulate a 2D imaging detector (8x8 pixels)
        nx, ny = 8, 8
        n_pixels = nx * ny
        n_energy = 100

        tof = np.arange(1, n_energy + 1, dtype=float)
        trans_2d = np.random.rand(n_pixels, n_energy) * 0.6 + 0.3
        err_2d = np.ones((n_pixels, n_energy)) * 0.025

        # Create 2D indices
        indices_2d = [(i, j) for i in range(nx) for j in range(ny)]

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=10.0, tstep=1e-6, indices=indices_2d
        )

        # Rebin
        rebinned = data.rebin(n=5)

        # Check shape
        assert rebinned.grouped_trans.shape == (n_pixels, n_energy // 5)

        # Check that 2D structure is preserved
        # group_shape may be a tuple with additional metadata
        if isinstance(rebinned.group_shape, tuple) and len(rebinned.group_shape) > 2:
            assert rebinned.group_shape[0] == (nx, ny)
        else:
            assert rebinned.group_shape == (nx, ny)

        # Check that we can still access by 2D index
        assert (3, 4) in rebinned.groups

    def test_grouped_rebin_no_binning_n1(self):
        """Test that n=1 returns a copy without modification."""
        tof = np.arange(1, 51, dtype=float)
        trans_2d = np.random.rand(3, len(tof)) * 0.4 + 0.5
        err_2d = np.ones((3, len(tof))) * 0.02

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=10.0, tstep=1e-6
        )

        # Rebin with n=1 (no rebinning)
        rebinned = data.rebin(n=1)

        # Should have same number of bins
        assert len(rebinned.table) == len(data.table)

        # Values should be identical (or very close)
        np.testing.assert_array_almost_equal(rebinned.grouped_trans, data.grouped_trans)

    def test_grouped_rebin_validates_parameters(self):
        """Test that rebinning validates parameters correctly."""
        tof = np.arange(1, 101, dtype=float)
        trans_2d = np.random.rand(2, len(tof)) * 0.5 + 0.4
        err_2d = np.ones((2, len(tof))) * 0.02

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=10.0, tstep=1e-6
        )

        # Should raise error if both n and tstep provided
        with pytest.raises(ValueError, match="Cannot specify both"):
            data.rebin(n=2, tstep=2e-6)

        # Should raise error if neither provided
        with pytest.raises(ValueError, match="Must specify either"):
            data.rebin()

    def test_grouped_rebin_large_dataset(self):
        """Test rebinning with larger dataset (performance test)."""
        # Simulate realistic imaging data: 64x64 pixels, 1000 energy bins
        nx, ny = 64, 64
        n_pixels = nx * ny
        n_energy = 1000

        tof = np.arange(1, n_energy + 1, dtype=float)
        trans_2d = np.random.rand(n_pixels, n_energy) * 0.7 + 0.2
        err_2d = np.ones((n_pixels, n_energy)) * 0.03

        indices_2d = [(i, j) for i in range(nx) for j in range(ny)]

        data = Data.from_grouped_arrays(
            tof=tof, trans=trans_2d, err=err_2d, L=10.0, tstep=1e-6, indices=indices_2d
        )

        # Rebin to reduce data size
        rebinned = data.rebin(n=10)

        # Check that rebinning completed successfully
        assert rebinned.grouped_trans.shape == (n_pixels, n_energy // 10)
        assert rebinned.is_grouped
        assert len(rebinned.indices) == n_pixels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
