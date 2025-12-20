import pytest
import pandas as pd
import numpy as np
import tempfile
from nres.data import Data
from nres.models import TransmissionModel
from nres.cross_section import CrossSection
from nres.grouped_fit import GroupedFitResult


class TestGroupedFit:
    """Test grouped fitting functionality"""

    def test_grouped_fit_basic(self, tmp_path):
        """Test basic grouped fitting with 1D data"""
        # Create test files for 3 groups
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        # Create test data with different transmission values
        tof = np.array([100, 200, 300, 400, 500])
        for i in range(3):
            # Different transmission for each group
            trans_value = 0.9 - i * 0.1
            signal_counts = np.array([900, 800, 700, 600, 500]) * trans_value
            openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

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

        # Create a simple cross section and model
        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)

        # Fit grouped data
        result = model.fit(data, emin=1e5, emax=1e7, method="least-squares", n_jobs=1, progress_bar=False)

        # Check that result is GroupedFitResult
        assert isinstance(result, GroupedFitResult)
        assert len(result) == 3
        assert result.group_shape == (3,)

        # Check that we can access individual results
        for i in range(3):
            individual_result = result[str(i)]
            assert individual_result is not None
            assert hasattr(individual_result, 'params')
            assert hasattr(individual_result, 'redchi')

    def test_grouped_fit_2d(self, tmp_path):
        """Test grouped fitting with 2D grid data"""
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

        # Create model
        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)

        # Fit grouped data
        result = model.fit(data, emin=1e5, emax=1e7, method="least-squares", n_jobs=1, progress_bar=False)

        # Check results
        assert isinstance(result, GroupedFitResult)
        assert len(result) == 4
        assert result.group_shape == (2, 2)

        # Check tuple indexing
        individual_result = result[(0, 0)]
        assert individual_result is not None

    def test_grouped_fit_summary(self, tmp_path):
        """Test summary method for grouped results"""
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

        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)
        result = model.fit(data, emin=1e5, emax=1e7, method="least-squares", n_jobs=1, progress_bar=False)

        # Test summary
        summary_df = result.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 2
        assert 'redchi' in summary_df.columns
        assert 'success' in summary_df.columns

    def test_grouped_fit_rietveld(self, tmp_path):
        """Test grouped fitting with rietveld method"""
        # Skip this test for now - rietveld fits require more realistic data
        pytest.skip("Rietveld grouped fits require more realistic test data")

    def test_grouped_fit_plot(self, tmp_path):
        """Test plot method for grouped results"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300, 400, 500])
        for i in range(2):
            signal_counts = np.array([900, 800, 700, 600, 500])
            openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

            pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        data = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            verbosity=0,
            n_jobs=1
        )

        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)
        result = model.fit(data, emin=1e5, emax=1e7, method="least-squares", n_jobs=1, progress_bar=False)

        # Test that plot method works
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            result.plot(0)
            result.plot("1")
            assert True
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_grouped_fit_plot_total_xs(self, tmp_path):
        """Test plot_total_xs method for grouped results"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300, 400, 500])
        for i in range(2):
            signal_counts = np.array([900, 800, 700, 600, 500])
            openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

            pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        data = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            verbosity=0,
            n_jobs=1
        )

        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)
        result = model.fit(data, emin=1e5, emax=1e7, method="least-squares", n_jobs=1, progress_bar=False)

        # Test that plot_total_xs method works
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            result.plot_total_xs(0)
            result.plot_total_xs("1")
            assert True
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_grouped_fit_fit_report(self, tmp_path):
        """Test fit_report method for grouped results"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300, 400, 500])
        for i in range(2):
            signal_counts = np.array([900, 800, 700, 600, 500])
            openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

            pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        data = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            verbosity=0,
            n_jobs=1
        )

        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)
        result = model.fit(data, emin=1e5, emax=1e7, method="least-squares", n_jobs=1, progress_bar=False)

        # Test that fit_report method works and returns string
        report = result.fit_report(0)
        assert isinstance(report, str)
        assert "Fit Statistics" in report or "redchi" in report.lower()

        # Also test with string index
        report = result.fit_report("1")
        assert isinstance(report, str)

    def test_grouped_fit_html_representation(self, tmp_path):
        """Test HTML representation for grouped results"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300, 400, 500])
        for i in range(2):
            signal_counts = np.array([900, 800, 700, 600, 500])
            openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

            pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        data = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            verbosity=0,
            n_jobs=1
        )

        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)
        result = model.fit(data, emin=1e5, emax=1e7, method="least-squares", n_jobs=1, progress_bar=False)

        # Test HTML representation
        html = result._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html
        assert "Grouped Fit Results Summary" in html

    def test_model_plot_with_grouped_data(self, tmp_path):
        """Test model.plot() with index parameter for grouped data"""
        signal_dir = tmp_path / "signal"
        openbeam_dir = tmp_path / "openbeam"
        signal_dir.mkdir()
        openbeam_dir.mkdir()

        tof = np.array([100, 200, 300, 400, 500])
        for i in range(2):
            signal_counts = np.array([900, 800, 700, 600, 500])
            openbeam_counts = np.array([1000, 1000, 1000, 1000, 1000])

            pd.DataFrame({"tof": tof, "counts": signal_counts, "err": np.sqrt(signal_counts)}).to_csv(
                signal_dir / f"pixel_{i}.csv", index=False
            )
            pd.DataFrame({"tof": tof, "counts": openbeam_counts, "err": np.sqrt(openbeam_counts)}).to_csv(
                openbeam_dir / f"pixel_{i}.csv", index=False
            )

        data = Data.from_grouped(
            str(signal_dir / "*.csv"),
            str(openbeam_dir / "*.csv"),
            verbosity=0,
            n_jobs=1
        )

        xs = CrossSection(Ag="Ag", splitby="materials")
        model = TransmissionModel(xs, vary_background=True)

        # Test that plot requires index for grouped data
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend

            # Should raise error without index
            with pytest.raises(ValueError, match="Data is grouped"):
                model.plot(data)

            # Should work with index
            ax = model.plot(data, index=0)
            assert ax is not None

            # Should work with string index
            ax = model.plot(data, index="1")
            assert ax is not None

        except ImportError:
            pytest.skip("matplotlib not available")


class TestGroupedFitResultMethods:
    """Test GroupedFitResult methods"""

    def test_normalize_index(self):
        """Test index normalization"""
        gfr = GroupedFitResult()

        # Test tuple normalization
        assert gfr._normalize_index((0, 0)) == "(0,0)"
        assert gfr._normalize_index("(0, 0)") == "(0,0)"
        assert gfr._normalize_index("(0,0)") == "(0,0)"

        # Test int normalization
        assert gfr._normalize_index(5) == "5"
        assert gfr._normalize_index("5") == "5"

    def test_parse_string_index(self):
        """Test string index parsing"""
        gfr = GroupedFitResult()

        # Test tuple parsing
        assert gfr._parse_string_index("(0,0)") == (0, 0)
        assert gfr._parse_string_index("(10,20)") == (10, 20)

        # Test int parsing
        assert gfr._parse_string_index("5") == 5

        # Test string parsing
        assert gfr._parse_string_index("center") == "center"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
