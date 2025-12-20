import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from nres.data import Data
from nres import utils


class TestDataL0T0:
    """Test L0 and t0 parameters in Data.from_counts"""

    def test_default_L0_t0(self, tmp_path):
        """Test that default L0=1.0 and t0=0.0 don't affect data"""
        # Create simple test data
        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        # Create mock data
        tof = np.array([100, 200, 300, 400, 500])
        signal_counts = np.array([900, 800, 700, 600, 500])
        openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

        # Write files
        pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
            signal_file, index=False
        )
        pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
            openbeam_file, index=False
        )

        # Load data with defaults
        data1 = Data.from_counts(str(signal_file), str(openbeam_file))

        # Load data with explicit L0=1.0, t0=0.0
        data2 = Data.from_counts(str(signal_file), str(openbeam_file), L0=1.0, t0=0.0)

        # Should be identical
        assert np.allclose(data1.table["energy"], data2.table["energy"])
        assert np.allclose(data1.table["trans"], data2.table["trans"])

    def test_L0_affects_energy(self, tmp_path):
        """Test that L0 parameter affects energy calibration"""
        # Create simple test data
        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        tof = np.array([100, 200, 300, 400, 500])
        signal_counts = np.array([900, 800, 700, 600, 500])
        openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

        pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
            signal_file, index=False
        )
        pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
            openbeam_file, index=False
        )

        # Load with default L0=1.0
        data_default = Data.from_counts(str(signal_file), str(openbeam_file))

        # Load with L0=1.1 (10% longer path)
        data_L0 = Data.from_counts(str(signal_file), str(openbeam_file), L0=1.1)

        # Energies should be different
        assert not np.allclose(data_default.table["energy"], data_L0.table["energy"])

        # Transmission should be the same (same counts)
        assert np.allclose(data_default.table["trans"], data_L0.table["trans"])

    def test_t0_affects_energy(self, tmp_path):
        """Test that t0 parameter affects energy calibration"""
        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        tof = np.array([100, 200, 300, 400, 500])
        signal_counts = np.array([900, 800, 700, 600, 500])
        openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

        pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
            signal_file, index=False
        )
        pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
            openbeam_file, index=False
        )

        # Load with default t0=0.0
        data_default = Data.from_counts(str(signal_file), str(openbeam_file))

        # Load with t0=10.0 (time offset)
        data_t0 = Data.from_counts(str(signal_file), str(openbeam_file), t0=10.0)

        # Energies should be different
        assert not np.allclose(data_default.table["energy"], data_t0.table["energy"])

        # Transmission should be the same
        assert np.allclose(data_default.table["trans"], data_t0.table["trans"])

    def test_L0_t0_stored(self, tmp_path):
        """Test that L and tstep are stored in Data object"""
        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        tof = np.array([100, 200, 300])
        counts = np.array([900, 800, 700])

        pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
            signal_file, index=False
        )
        pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
            openbeam_file, index=False
        )

        L = 10.59
        tstep = 1.56255e-9

        data = Data.from_counts(str(signal_file), str(openbeam_file), L=L, tstep=tstep)

        assert data.L == L
        assert data.tstep == tstep
        assert data.signal is not None
        assert data.openbeam is not None


class TestDataGrouped:
    """Test grouped data loading functionality"""

    def test_from_grouped_basic(self, tmp_path):
        """Test basic grouped data loading with sequential indices"""
        # Create test files
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300])
        for i in range(3):
            signal_counts = np.array([900 - i * 100, 800 - i * 100, 700 - i * 100])
            openbeam_counts = np.array([1000, 1000, 1000])

            pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        # Load grouped data
        data = Data.from_grouped(
            str(signal_dir / "pixel_*.csv"),
            str(openbeam_dir / "pixel_*.csv"),
            verbosity=0,
            n_jobs=1
        )

        assert data.is_grouped
        assert len(data.groups) == 3
        assert len(data.indices) == 3
        assert data.group_shape == (3,)

    def test_from_grouped_2d(self, tmp_path):
        """Test 2D grouped data loading"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300])

        # Create 2x2 grid
        for x in range(2):
            for y in range(2):
                signal_counts = np.array([900, 800, 700])
                openbeam_counts = np.array([1000, 1000, 1000])

                pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                    signal_dir / f"pixel_x{x}_y{y}.csv", index=False
                )
                pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                    openbeam_dir / f"pixel_x{x}_y{y}.csv", index=False
                )

        # Load grouped data
        data = Data.from_grouped(
            str(signal_dir / "pixel_*.csv"),
            str(openbeam_dir / "pixel_*.csv"),
            verbosity=0,
            n_jobs=1
        )

        assert data.is_grouped
        assert len(data.groups) == 4
        assert data.group_shape == (2, 2)

    def test_from_grouped_custom_indices(self, tmp_path):
        """Test grouped data with custom indices"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300])
        names = ["sample1", "sample2", "sample3"]

        for name in names:
            counts = np.array([900, 800, 700])
            pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
                signal_dir / f"{name}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
                openbeam_dir / f"{name}.csv", index=False
            )

        # Load with custom indices
        data = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            indices=names,
            verbosity=0,
            n_jobs=1
        )

        assert data.is_grouped
        assert len(data.groups) == 3
        assert set(data.indices) == set(names)

    def test_from_grouped_L0_t0(self, tmp_path):
        """Test that L0 and t0 are applied to grouped data"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300])
        for i in range(2):
            counts = np.array([900, 800, 700])
            pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        # Load with default L0, t0
        data_default = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            verbosity=0,
            n_jobs=1
        )

        # Load with L0=1.1
        data_L0 = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            L0=1.1,
            verbosity=0,
            n_jobs=1
        )

        # Check that energies are different
        for idx in data_default.indices:
            assert not np.allclose(
                data_default.groups[idx]["energy"],
                data_L0.groups[idx]["energy"]
            )


class TestDataPlot:
    """Test plot method with grouped data"""

    def test_plot_grouped_data(self, tmp_path):
        """Test plotting grouped data with index parameter"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300])
        for i in range(2):
            counts = np.array([900, 800, 700])
            pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        data = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            verbosity=0,
            n_jobs=1
        )

        # Should not raise error
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        ax = data.plot(index=0)
        assert ax is not None

    def test_plot_non_grouped_rejects_index(self, tmp_path):
        """Test that non-grouped data rejects index parameter"""
        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        tof = np.array([100, 200, 300])
        counts = np.array([900, 800, 700])

        pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
            signal_file, index=False
        )
        pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
            openbeam_file, index=False
        )

        data = Data.from_counts(str(signal_file), str(openbeam_file))

        import matplotlib
        matplotlib.use('Agg')

        # Should raise error when specifying index for non-grouped data
        with pytest.raises(ValueError, match="Cannot specify index for non-grouped data"):
            data.plot(index=0)

    def test_plot_map_1d(self, tmp_path):
        """Test plot_map for 1D grouped data"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300])
        for i in range(3):
            signal_counts = np.array([900 - i * 100, 800 - i * 100, 700 - i * 100])
            openbeam_counts = np.array([1000, 1000, 1000])

            pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        data = Data.from_grouped(
            str(signal_dir / "pixel_*.csv"),
            str(openbeam_dir / "pixel_*.csv"),
            verbosity=0,
            n_jobs=1
        )

        import matplotlib
        matplotlib.use('Agg')

        # Test plot_map works
        ax = data.plot_map(emin=1e5, emax=1e7)
        assert ax is not None

    def test_plot_map_2d(self, tmp_path):
        """Test plot_map for 2D grouped data"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300])
        for x in range(2):
            for y in range(2):
                signal_counts = np.array([900, 800, 700])
                openbeam_counts = np.array([1000, 1000, 1000])

                pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                    signal_dir / f"pixel_x{x}_y{y}.csv", index=False
                )
                pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                    openbeam_dir / f"pixel_x{x}_y{y}.csv", index=False
                )

        data = Data.from_grouped(
            str(signal_dir / "pixel_*.csv"),
            str(openbeam_dir / "pixel_*.csv"),
            verbosity=0,
            n_jobs=1
        )

        import matplotlib
        matplotlib.use('Agg')

        # Test plot_map works for 2D data
        ax = data.plot_map(emin=1e5, emax=1e7)
        assert ax is not None

    def test_plot_map_rejects_non_grouped(self, tmp_path):
        """Test that plot_map raises error for non-grouped data"""
        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        tof = np.array([100, 200, 300])
        counts = np.array([900, 800, 700])

        pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
            signal_file, index=False
        )
        pd.DataFrame({"tof": tof, "counts": counts, "err": np.sqrt(counts)}).to_csv(
            openbeam_file, index=False
        )

        data = Data.from_counts(str(signal_file), str(openbeam_file))

        # Should raise error for non-grouped data
        with pytest.raises(ValueError, match="plot_map only works for grouped data"):
            data.plot_map()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
