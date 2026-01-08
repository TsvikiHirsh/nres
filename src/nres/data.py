from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nres import utils

warnings.filterwarnings("ignore")


class Data:
    """
    A class for handling neutron transmission data, including reading counts data,
    calculating transmission, and plotting the results.

    Attributes:
    -----------
    table : pandas.DataFrame or None
        A dataframe containing energy, transmission, and error values.
    tgrid : pandas.Series or None
        A time-of-flight grid corresponding to the time steps in the data.
    signal : pandas.DataFrame or None
        The signal counts data (tof, counts, err).
    openbeam : pandas.DataFrame or None
        The open beam counts data (tof, counts, err).
    L : float or None
        Distance (meters) used in the energy conversion from time-of-flight.
    tstep : float or None
        Time step (seconds) for converting time-of-flight to energy.
    is_grouped : bool
        Whether this Data object contains grouped data.
    groups : dict or None
        Dict mapping index -> table for grouped data.
    indices : list or None
        List of string indices for grouped data.
    group_shape : tuple or None
        Tuple (nx, ny) for 2D, (n,) for 1D, None for named groups.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Data object with optional keyword arguments.

        Parameters:
        -----------
        **kwargs : dict, optional
            Additional keyword arguments to set any instance-specific properties.
        """
        self.table = None
        self.tgrid = None
        self.signal = None
        self.openbeam = None
        self.L = None
        self.tstep = None

        # Grouped data attributes
        self.is_grouped = False
        self.groups = None  # Dict mapping index -> table
        self.indices = None  # List of string indices
        self.group_shape = None  # Tuple (nx, ny) for 2D, (n,) for 1D, None for named

    @classmethod
    def _read_counts(cls, filename="run2_graphite_00000/graphite.csv"):
        """
        Reads the counts data from a CSV file and calculates errors if not provided.

        Parameters:
        -----------
        filename : str, optional
            The path to the CSV file containing time-of-flight (tof) and counts data.
            Default is 'run2_graphite_00000/graphite.csv'.

        Returns:
        --------
        df : pandas.DataFrame
            A DataFrame containing columns: 'tof', 'counts', and 'err'. Errors are calculated
            as the square root of counts if not provided in the file.
        """
        df = pd.read_csv(
            filename, names=["tof", "counts", "err"], header=None, skiprows=1
        )

        # If no error values provided, calculate as sqrt of counts
        if all(df["err"].isnull()):
            df["err"] = np.sqrt(df["counts"])

        # Store label from filename (without path and extension)
        df.attrs["label"] = filename.split("/")[-1].rstrip(".csv")

        return df

    @classmethod
    def from_counts(
        cls,
        signal: str,
        openbeam: str,
        empty_signal: str = "",
        empty_openbeam: str = "",
        tstep: float = 1.56255e-9,
        L: float = 10.59,
        L0: float = 1.0,
        t0: float = 0.0,
        verbosity: int = 1,
    ):
        """
        Creates a Data object from signal and open beam counts data, calculates transmission, and converts tof to energy.

        Parameters:
        -----------
        signal : str
            Path to the CSV file containing the signal data (tof, counts, err).
        openbeam : str
            Path to the CSV file containing the open beam data (tof, counts, err).
        empty_signal : str, optional
            Path to the CSV file containing the empty signal data for background correction. Default is an empty string.
        empty_openbeam : str, optional
            Path to the CSV file containing the empty open beam data for background correction. Default is an empty string.
        tstep : float, optional
            Time step (seconds) for converting time-of-flight (tof) to energy. Default is 1.56255e-9.
        L : float, optional
            Distance (meters) used in the energy conversion from time-of-flight. Default is 10.59 m.
        L0 : float, optional
            Flight path scale factor from vary_tof optimization. Default is 1.0.
            Values > 1.0 indicate a longer path, < 1.0 a shorter path.
        t0 : float, optional
            Time offset correction in seconds from vary_tof optimization. Default is 0.0.
            Will be converted to TOF channel units internally using tstep.
        verbosity : int, optional
            Verbosity level. If 0, suppresses warnings from sqrt operations. Default is 1.

        Returns:
        --------
        Data
            A Data object containing transmission and energy data.
        """
        # Read signal and open beam counts
        signal = cls._read_counts(signal)
        openbeam = cls._read_counts(openbeam)

        # Apply L0 and t0 corrections
        # Note: signal["tof"] is in TOF channel units, t0 is in seconds
        # Convert t0 to TOF channel units by dividing by tstep
        dtof = (1.0 - L0) * signal["tof"] + t0 / tstep
        corrected_tof = signal["tof"] + dtof

        # Convert corrected tof to energy using provided time step and distance
        signal["energy"] = utils.time2energy(corrected_tof * tstep, L)

        # Calculate transmission and associated error
        # Suppress RuntimeWarnings from sqrt if verbosity is 0
        import warnings

        with warnings.catch_warnings():
            if verbosity == 0:
                warnings.simplefilter("ignore", RuntimeWarning)

            transmission = signal["counts"] / openbeam["counts"]
            err = transmission * np.sqrt(
                (signal["err"] / signal["counts"]) ** 2
                + (openbeam["err"] / openbeam["counts"]) ** 2
            )

            # If background (empty) data is provided, apply correction
            if empty_signal and empty_openbeam:
                empty_signal = cls._read_counts(empty_signal)
                empty_openbeam = cls._read_counts(empty_openbeam)

                transmission *= empty_openbeam["counts"] / empty_signal["counts"]
                err = transmission * np.sqrt(
                    (signal["err"] / signal["counts"]) ** 2
                    + (openbeam["err"] / openbeam["counts"]) ** 2
                    + (empty_signal["err"] / empty_signal["counts"]) ** 2
                    + (empty_openbeam["err"] / empty_openbeam["counts"]) ** 2
                )

        # Construct a dataframe for energy, transmission, and error
        df = pd.DataFrame(
            {"energy": signal["energy"], "trans": transmission, "err": err}
        )

        # Set the label attribute from the signal file
        df.attrs["label"] = signal.attrs["label"]

        # Create and return the Data object
        self_data = cls()
        self_data.table = df
        self_data.tgrid = signal["tof"]
        self_data.signal = signal
        self_data.openbeam = openbeam
        self_data.L = L
        self_data.tstep = tstep

        # Store L0 and t0 values to indicate TOF correction was applied
        self_data.L0 = L0
        self_data.t0 = t0

        # Store empty signal/openbeam if provided (for proper rebinning)
        # Note: At this point, if background correction was applied, empty_signal
        # and empty_openbeam have been reassigned to DataFrames (line 148-149)
        if (
            empty_signal is not None
            and empty_openbeam is not None
            and isinstance(empty_signal, pd.DataFrame)
        ):
            self_data.empty_signal = empty_signal
            self_data.empty_openbeam = empty_openbeam
        else:
            self_data.empty_signal = None
            self_data.empty_openbeam = None

        return self_data

    def _normalize_index(self, index):
        """
        Normalize index for group lookup.
        Converts tuples like (10, 20) to strings like "(10,20)" for consistent access.
        Accepts both "(10,20)" and "(10, 20)" string formats.

        Parameters:
        -----------
        index : int, tuple, or str
            The index to normalize

        Returns:
        --------
        str
            String representation of the index (tuples without spaces)
        """
        if isinstance(index, tuple):
            # (10, 20) -> "(10,20)" (no spaces)
            return str(index).replace(" ", "")
        if isinstance(index, str):
            # Remove spaces from string if it looks like a tuple: "(10, 20)" -> "(10,20)"
            return index.replace(" ", "")
        # 5 -> "5"
        return str(index)

    def _parse_string_index(self, string_idx):
        """
        Parse a string index back to its original form.
        "(10, 20)" -> (10, 20)
        "5" -> 5
        "center" -> "center"

        Parameters:
        -----------
        string_idx : str
            String representation of index

        Returns:
        --------
        tuple, int, or str
            Original index form
        """
        import ast

        try:
            # Try to parse as Python literal (for tuples and ints)
            parsed = ast.literal_eval(string_idx)
            return parsed
        except (ValueError, SyntaxError):
            # If parsing fails, it's a named string
            return string_idx

    @classmethod
    def _extract_indices_from_filenames(cls, filenames, pattern):
        """Extract indices from filenames based on pattern."""
        import os
        import re

        indices = []

        # Auto-detect pattern if needed
        if pattern == "auto":
            # Try common patterns
            test_name = os.path.basename(filenames[0])

            # Try 2D patterns - look for _x or _y to avoid matching dimension specs like 16x16
            match_2d = re.search(r"_x(\d+).*_y(\d+)", test_name, re.IGNORECASE)
            if not match_2d:
                # Try without underscores but with word boundaries
                match_2d = re.search(
                    r"\bx(\d+)[_\s].*\by(\d+)", test_name, re.IGNORECASE
                )

            if match_2d:
                pattern = "_x{x}_y{y}"
            else:
                # Try 1D patterns - look for trailing numbers or with keywords
                match_1d = re.search(
                    r"(?:idx|pixel|det)[_\s]*(\d+)", test_name, re.IGNORECASE
                )
                if not match_1d:
                    # Try just trailing number before extension
                    match_1d = re.search(r"_(\d+)\.", test_name)

                if match_1d:
                    pattern = "idx{i}"
                else:
                    # No pattern found - use filenames as indices (for named ROIs)
                    pattern = "{name}"

        # Extract based on pattern
        for fname in filenames:
            basename = os.path.basename(fname)

            if "{x}" in pattern and "{y}" in pattern:
                # 2D grid pattern - try multiple patterns
                # First try with underscores
                match = re.search(r"_x(\d+).*_y(\d+)", basename, re.IGNORECASE)
                if not match:
                    # Try with word boundaries
                    match = re.search(
                        r"\bx(\d+)[_\s].*\by(\d+)", basename, re.IGNORECASE
                    )
                if not match:
                    # Try simple pattern as last resort
                    match = re.search(
                        r"(?<![\dx])x(\d+).*(?<![\dx])y(\d+)", basename, re.IGNORECASE
                    )

                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    indices.append((x, y))
                else:
                    raise ValueError(
                        f"Could not extract x,y coordinates from: {basename}. "
                        f"Filename should contain _x<num> and _y<num> patterns."
                    )

            elif "{i}" in pattern:
                # 1D array pattern - try multiple approaches
                # First try with keywords
                match = re.search(
                    r"(?:idx|pixel|det)[_\s]*(\d+)", basename, re.IGNORECASE
                )
                if not match:
                    # Try trailing number before extension
                    match = re.search(r"_(\d+)\.", basename)
                if not match:
                    # Last resort - any number in the filename (rightmost)
                    matches = re.findall(r"(\d+)", basename)
                    if matches:
                        match = type(
                            "obj", (object,), {"group": lambda self, n: matches[-1]}
                        )()

                if match:
                    indices.append(int(match.group(1)))
                else:
                    raise ValueError(f"Could not extract index from: {basename}")

            elif "{name}" in pattern:
                # Named groups - use filename without extension
                name = os.path.splitext(basename)[0]
                indices.append(name)

            else:
                # Unknown pattern - use sequential
                indices.append(len(indices))

        return indices

    @classmethod
    def _determine_group_shape(cls, indices):
        """Determine group shape and dimensionality from indices."""
        # Check for empty indices (works with both lists and numpy arrays)
        if len(indices) == 0:
            return None, False, False

        first_idx = indices[0]

        # Check if 2D (tuples)
        if isinstance(first_idx, tuple) and len(first_idx) == 2:
            # 2D grid - use max coordinates + 1 to handle sparse grids
            xs = [idx[0] for idx in indices]
            ys = [idx[1] for idx in indices]
            return (max(ys) + 1, max(xs) + 1), True, False

        # Check if 1D (ints)
        if isinstance(first_idx, (int, int.__class__)):
            # 1D array
            return (len(indices),), False, True

        # Named indices (strings)
        return None, False, False

    @classmethod
    def from_transmission(cls, filename: str):
        """
        Creates a Data object directly from a transmission data file containing energy, transmission, and error values.

        Parameters:
        -----------
        filename : str
            Path to the file containing the transmission data (energy, transmission, error) separated by whitespace.

        Returns:
        --------
        Data
            A Data object with the transmission data loaded into a dataframe.
        """
        df = pd.read_csv(
            filename,
            names=["energy", "trans", "err"],
            header=None,
            skiprows=0,
            sep=r"\s+",
        )

        # Create Data object and assign the dataframe
        self_data = cls()
        self_data.table = df

        return self_data

    @classmethod
    def from_grouped_arrays(
        cls,
        tof,
        trans,
        err,
        L: float,
        tstep: float,
        L0: float = 1.0,
        t0: float = 0.0,
        indices: list = None,
    ):
        """
        Creates a Data object from grouped transmission arrays.

        This method is useful for creating grouped data from numpy arrays,
        such as data from imaging detectors or multi-sample measurements.

        Parameters:
        -----------
        tof : array-like
            Time-of-flight bins (1D array).
        trans : array-like
            Transmission values. Shape: (n_groups, n_energy_bins)
        err : array-like
            Transmission uncertainties. Shape: (n_groups, n_energy_bins)
        L : float
            Flight path length in meters.
        tstep : float
            Time step in seconds for TOF to energy conversion.
        L0 : float, optional
            Flight path scale factor. Default is 1.0.
        t0 : float, optional
            Time offset in seconds. Default is 0.0.
        indices : list, optional
            List of group indices. Can be:
            - List of ints for 1D array: [0, 1, 2, ...]
            - List of tuples for 2D grid: [(0,0), (0,1), (1,0), ...]
            - List of strings for named groups: ['sample1', 'sample2', ...]
            If not provided, will use sequential integers.

        Returns:
        --------
        Data
            A Data object with grouped data.

        Examples:
        ---------
        >>> # Create grouped data for 10 pixels with 100 energy bins each
        >>> tof = np.arange(1, 101)
        >>> trans_2d = np.random.rand(10, 100)
        >>> err_2d = trans_2d * 0.05
        >>> data = Data.from_grouped_arrays(tof, trans_2d, err_2d, L=10.0, tstep=1e-6)
        """
        tof = np.asarray(tof)
        trans = np.asarray(trans)
        err = np.asarray(err)

        # Validate shapes
        if trans.ndim != 2 or err.ndim != 2:
            raise ValueError(
                "trans and err must be 2D arrays (n_groups, n_energy_bins)"
            )
        if trans.shape != err.shape:
            raise ValueError(f"Shape mismatch: trans {trans.shape} vs err {err.shape}")
        if len(tof) != trans.shape[1]:
            raise ValueError(
                f"TOF length {len(tof)} doesn't match trans shape {trans.shape}"
            )

        n_groups, n_energy = trans.shape

        # Create indices if not provided
        if indices is None:
            indices = list(range(n_groups))
        elif len(indices) != n_groups:
            raise ValueError(
                f"Number of indices ({len(indices)}) doesn't match number of groups ({n_groups})"
            )

        # Determine group shape
        group_shape = cls._determine_group_shape(indices)

        # Create Data object
        self_data = cls()
        self_data.L = L
        self_data.tstep = tstep
        self_data.L0 = L0
        self_data.t0 = t0
        self_data.is_grouped = True
        self_data.indices = indices
        self_data.group_shape = group_shape
        self_data.grouped_trans = trans
        self_data.grouped_err = err

        # Convert TOF to energy
        energy = utils.time2energy(tof * tstep, L)

        # Create groups dictionary
        self_data.groups = {}
        for i, idx in enumerate(indices):
            table = pd.DataFrame(
                {"energy": energy, "trans": trans[i, :], "err": err[i, :]}
            )
            table.attrs["label"] = str(idx)
            self_data.groups[idx] = table

        # Set main table to first group
        self_data.table = self_data.groups[indices[0]]

        return self_data

    @classmethod
    def from_grouped(
        cls,
        signal,
        openbeam,
        empty_signal: str = "",
        empty_openbeam: str = "",
        tstep: float = 1.56255e-9,
        L: float = 10.59,
        L0: float = 1.0,
        t0: float = 0.0,
        pattern: str = "auto",
        indices: list = None,
        verbosity: int = 1,
        n_jobs: int = -1,
    ):
        """
        Creates a Data object from grouped counts data using glob patterns.

        Supports 1D arrays, 2D grids, and named indices for spatially-resolved analysis.

        Parameters:
        -----------
        signal : str
            Glob pattern for signal files (e.g., "archive/pixel_*.csv" or "data/grid_*_x*_y*.csv").
            Can also be a folder path - all .csv files in the folder will be loaded.
        openbeam : str
            Glob pattern for openbeam files. Can also be a folder path.
        empty_signal : str, optional
            Glob pattern for empty signal files for background correction.
        empty_openbeam : str, optional
            Glob pattern for empty openbeam files for background correction.
        tstep : float, optional
            Time step (seconds) for converting time-of-flight to energy. Default is 1.56255e-9.
        L : float, optional
            Distance (meters) used in the energy conversion from time-of-flight. Default is 10.59 m.
        L0 : float, optional
            Flight path scale factor from vary_tof optimization. Default is 1.0.
        t0 : float, optional
            Time offset correction in seconds from vary_tof optimization. Default is 0.0.
            Will be converted to TOF channel units internally using tstep.
        pattern : str, optional
            Coordinate extraction pattern. Default is "auto" which tries common patterns:
            - "x{x}_y{y}" for 2D grids (e.g., "grid_x10_y20.csv")
            - "idx{i}" or "pixel_{i}" for 1D arrays
            Custom patterns can use {x}, {y}, {i}, or {name}.
        indices : list, optional
            If provided, use these indices instead of extracting from filenames.
            Can be list of ints (1D), list of tuples (2D), or list of strings (named).
        verbosity : int, optional
            Verbosity level. If >= 1, shows progress bar. Default is 1.
        n_jobs : int, optional
            Number of parallel jobs for loading files. Default is -1 (use all CPUs).
            Set to 1 for sequential loading.

        Returns:
        --------
        Data
            A Data object with grouped data stored in self.groups.

        Examples:
        ---------
        # 2D grid from filenames like "pixel_x10_y20.csv"
        >>> data = Data.from_grouped("folder/pixel_*.csv", "folder_ob/pixel_*.csv")

        # 1D array with custom indices
        >>> data = Data.from_grouped(
        ...     "data/det_*.csv", "data_ob/det_*.csv", indices=[0, 1, 2, 3]
        ... )

        # Named groups
        >>> data = Data.from_grouped(
        ...     "samples/*.csv", "ref/*.csv", indices=["sample1", "sample2"]
        ... )
        """
        import glob
        import os

        # Find all matching files (support folder input)
        if os.path.isdir(signal):
            signal_files = sorted(glob.glob(os.path.join(signal, "*.csv")))
        else:
            signal_files = sorted(glob.glob(signal))

        if os.path.isdir(openbeam):
            openbeam_files = sorted(glob.glob(os.path.join(openbeam, "*.csv")))
        else:
            openbeam_files = sorted(glob.glob(openbeam))

        if not signal_files:
            raise ValueError(f"No files found matching pattern: {signal}")
        if not openbeam_files:
            raise ValueError(f"No files found matching pattern: {openbeam}")
        if len(signal_files) != len(openbeam_files):
            raise ValueError(
                f"Mismatch: {len(signal_files)} signal files vs {len(openbeam_files)} openbeam files"
            )

        # Handle empty beam files if provided
        empty_signal_files = []
        empty_openbeam_files = []
        use_single_empty = False  # Flag for single empty file reuse

        if empty_signal and empty_openbeam:
            empty_signal_files = sorted(glob.glob(empty_signal))
            empty_openbeam_files = sorted(glob.glob(empty_openbeam))

            # Allow single empty file to be reused for all groups
            if len(empty_signal_files) == 1 and len(empty_openbeam_files) == 1:
                use_single_empty = True
            elif len(empty_signal_files) != len(signal_files) or len(
                empty_openbeam_files
            ) != len(signal_files):
                raise ValueError(
                    f"Empty file count mismatch: {len(empty_signal_files)} empty signal, "
                    f"{len(empty_openbeam_files)} empty openbeam vs {len(signal_files)} signal files. "
                    f"Provide either 1 empty file (reused for all) or one per signal file."
                )

        # Extract or use provided indices
        if indices is not None:
            # Convert numpy arrays to list
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()

            # User-provided indices
            if len(indices) != len(signal_files):
                raise ValueError(
                    f"Number of indices ({len(indices)}) must match number of files ({len(signal_files)})"
                )
            extracted_indices = indices
        else:
            # Auto-extract from filenames
            extracted_indices = cls._extract_indices_from_filenames(
                signal_files, pattern
            )

        # Determine group dimensionality and shape BEFORE converting to strings
        group_shape, is_2d, is_1d = cls._determine_group_shape(extracted_indices)

        # Convert all indices to strings for consistent access
        # For 2D: (10, 20) -> "(10,20)" (no spaces)
        # For 1D: 5 -> "5"
        # For named: "center" -> "center"
        string_indices = []
        for idx in extracted_indices:
            if isinstance(idx, tuple):
                # Convert tuple to string without spaces: "(10,20)"
                string_indices.append(str(idx).replace(" ", ""))
            elif isinstance(idx, str):
                string_indices.append(idx)  # "center"
            else:
                string_indices.append(str(idx))  # "5"

        extracted_indices = string_indices

        # Create Data object
        self_data = cls()
        self_data.is_grouped = True
        self_data.indices = extracted_indices
        self_data.group_shape = group_shape
        self_data.groups = {}
        self_data.L = L
        self_data.tstep = tstep

        # Store L0 and t0 values to indicate TOF correction was applied
        self_data.L0 = L0
        self_data.t0 = t0

        # Helper function to load a single group
        def load_single_group(i, idx):
            """Load a single group's data files."""
            sig_file = signal_files[i]
            ob_file = openbeam_files[i]

            # Handle empty files - use single file if available, otherwise per-group
            if use_single_empty:
                es_file = empty_signal_files[0]
                eo_file = empty_openbeam_files[0]
            else:
                es_file = empty_signal_files[i] if empty_signal_files else ""
                eo_file = empty_openbeam_files[i] if empty_openbeam_files else ""

            # Create individual Data object for this group
            group_data = cls.from_counts(
                signal=sig_file,
                openbeam=ob_file,
                empty_signal=es_file,
                empty_openbeam=eo_file,
                tstep=tstep,
                L=L,
                L0=L0,
                t0=t0,
                verbosity=verbosity,
            )

            return idx, group_data.table

        # Load groups in parallel or sequentially
        if n_jobs == 1:
            # Sequential loading with progress bar
            if verbosity >= 1:
                try:
                    from tqdm.auto import tqdm

                    iterator = tqdm(
                        enumerate(extracted_indices),
                        total=len(extracted_indices),
                        desc=f"Loading {len(extracted_indices)} groups",
                    )
                except ImportError:
                    iterator = enumerate(extracted_indices)
            else:
                iterator = enumerate(extracted_indices)

            for i, idx in iterator:
                idx, table = load_single_group(i, idx)
                self_data.groups[idx] = table
        else:
            # Parallel loading
            from joblib import Parallel, delayed

            # Create progress bar if needed
            if verbosity >= 1:
                try:
                    from tqdm.auto import tqdm

                    pbar = tqdm(
                        total=len(extracted_indices),
                        desc=f"Loading {len(extracted_indices)} groups",
                    )
                except ImportError:
                    pbar = None
            else:
                pbar = None

            # Load groups in parallel using threading backend
            # (threading is appropriate for I/O-bound file loading and avoids serialization issues)
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(load_single_group)(i, idx)
                for i, idx in enumerate(extracted_indices)
            )

            # Store results
            for idx, table in results:
                self_data.groups[idx] = table
                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        # Set first group as default table for compatibility
        self_data.table = self_data.groups[extracted_indices[0]]

        return self_data

    def plot(self, index=None, **kwargs):
        """
        Plots the transmission data with error bars.

        Parameters:
        -----------
        index : int, tuple, or str, optional
            For grouped data, specify which group to plot:
            - int: 1D array index
            - tuple: (x, y) for 2D grid
            - str: named index
            If None and data is grouped, plots first group.
            If None and data is not grouped, plots the main table.
        **kwargs : dict, optional
            Additional plotting parameters:
            - xlim : tuple, optional
              Limits for the x-axis (default: (0.5e6, 1e7)).
            - ylim : tuple, optional
              Limits for the y-axis (default: (0., 1.)).
            - ecolor : str, optional
              Error bar color (default: "0.8").
            - xlabel : str, optional
              Label for the x-axis (default: "Energy [eV]").
            - ylabel : str, optional
              Label for the y-axis (default: "Transmission").
            - logx : bool, optional
              Whether to plot the x-axis on a logarithmic scale (default: True).

        Returns:
        --------
        matplotlib.Axes
            The axes of the plot containing the transmission data.
        """
        xlim = kwargs.pop("xlim", (0.5e6, 1e7))
        ylim = kwargs.pop("ylim", (0.0, 1.0))
        ecolor = kwargs.pop("ecolor", "0.8")
        xlabel = kwargs.pop("xlabel", "Energy [eV]")
        ylabel = kwargs.pop("ylabel", "Transmission")
        logx = kwargs.pop("logx", True)

        # Determine which table to plot and set label
        plot_label = kwargs.pop("label", None)
        if self.is_grouped:
            # For grouped data
            if index is None:
                # Default to first group
                index = self.indices[0]
            # Normalize index for lookup (supports tuple, int, or string access)
            normalized_index = self._normalize_index(index)
            if normalized_index not in self.groups:
                raise ValueError(
                    f"Index {index} not found in groups. Available indices: {self.indices}"
                )
            table_to_plot = self.groups[normalized_index]
            # Add index to label if not provided
            if plot_label is None:
                plot_label = f"Index {index}"
        else:
            # For non-grouped data
            if index is not None:
                raise ValueError("Cannot specify index for non-grouped data")
            table_to_plot = self.table

        # Plot the data with error bars
        ax = table_to_plot.dropna().plot(
            x="energy",
            y="trans",
            yerr="err",
            xlim=xlim,
            ylim=ylim,
            logx=logx,
            ecolor=ecolor,
            xlabel=xlabel,
            ylabel=ylabel,
            label=plot_label,
            **kwargs,
        )

        # Add legend if label was set
        if plot_label is not None:
            ax.legend()

        return ax

    def plot_map(
        self,
        emin=0.5e6,
        emax=20e6,
        emin2=None,
        emax2=None,
        logT=False,
        n_density=None,
        sigma=None,
        **kwargs,
    ):
        """
        Plot transmission map averaged over energy range for grouped data.

        Parameters:
        -----------
        emin : float, optional
            Minimum energy for averaging (default: 0.5e6 eV).
        emax : float, optional
            Maximum energy for averaging (default: 20e6 eV).
        emin2 : float, optional
            Minimum energy for second energy range. If provided with emax2,
            plots the ratio of transmissions: T(emin,emax) / T(emin2,emax2).
        emax2 : float, optional
            Maximum energy for second energy range.
        logT : bool, optional
            If True, plots -ln(T)/(n*sigma) as thickness estimate in cm (default: False).
            Requires n_density and sigma parameters. According to Beer's law T=exp(-n*sigma*d),
            this transformation gives an estimate of thickness d.
        n_density : float, optional
            Number density in atoms/cm^3 (required if logT=True).
        sigma : float, optional
            Average cross-section in barns (required if logT=True).
        **kwargs : dict, optional
            Additional plotting parameters:
            - cmap : str, optional
              Colormap for 2D maps (default: 'viridis').
            - title : str, optional
              Plot title (default: auto-generated).
            - vmin, vmax : float, optional
              Color scale limits for 2D maps.
            - figsize : tuple, optional
              Figure size (width, height) in inches.

        Returns:
        --------
        matplotlib.Axes
            The axes of the plot.

        Raises:
        -------
        ValueError
            If called on non-grouped data, or if logT=True but n_density or sigma not provided.

        Examples:
        ---------
        >>> # For 2D grid data
        >>> data = Data.from_grouped("pixel_x*_y*.csv", "ob_x*_y*.csv")
        >>> data.plot_map(emin=1e6, emax=10e6)

        >>> # For 1D array data
        >>> data = Data.from_grouped("pixel_*.csv", "ob_*.csv")
        >>> data.plot_map(emin=0.5e6, emax=5e6)

        >>> # Plot thickness estimate
        >>> data.plot_map(emin=1e6, emax=10e6, logT=True, n_density=8.5e22, sigma=10.0)

        >>> # Plot transmission ratio
        >>> data.plot_map(emin=1e6, emax=5e6, emin2=5e6, emax2=10e6)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.is_grouped:
            raise ValueError("plot_map only works for grouped data")

        # Validate parameters
        if logT and (n_density is None or sigma is None):
            raise ValueError("logT=True requires both n_density and sigma parameters")

        if (emin2 is not None or emax2 is not None) and not (
            emin2 is not None and emax2 is not None
        ):
            raise ValueError("Both emin2 and emax2 must be provided together")

        if logT and (emin2 is not None or emax2 is not None):
            raise ValueError(
                "Cannot use both logT and transmission ratio (emin2/emax2) simultaneously"
            )

        # Extract kwargs
        cmap = kwargs.pop("cmap", "viridis")
        title = kwargs.pop("title", None)
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        figsize = kwargs.pop("figsize", None)

        # Calculate average transmission for each group
        avg_trans = {}
        for idx in self.indices:
            table = self.groups[idx]
            mask = (table["energy"] >= emin) & (table["energy"] <= emax)
            avg_trans[idx] = table.loc[mask, "trans"].mean()

        # Calculate second transmission map if requested (for ratio)
        if emin2 is not None and emax2 is not None:
            avg_trans2 = {}
            for idx in self.indices:
                table = self.groups[idx]
                mask = (table["energy"] >= emin2) & (table["energy"] <= emax2)
                avg_trans2[idx] = table.loc[mask, "trans"].mean()

            # Calculate ratio
            for idx in self.indices:
                if avg_trans2[idx] != 0:
                    avg_trans[idx] = avg_trans[idx] / avg_trans2[idx]
                else:
                    avg_trans[idx] = np.nan

        # Apply logT transformation if requested
        if logT:
            # Convert sigma from barns to cm^2 (1 barn = 1e-24 cm^2)
            sigma_cm2 = sigma * 1e-24
            for idx in self.indices:
                trans_val = avg_trans[idx]
                if trans_val > 0 and not np.isnan(trans_val):
                    # -ln(T) / (n * sigma) = thickness in cm
                    avg_trans[idx] = -np.log(trans_val) / (n_density * sigma_cm2)
                else:
                    avg_trans[idx] = np.nan

        # Create visualization based on group_shape
        if self.group_shape and len(self.group_shape) == 2:
            # 2D pcolormesh for proper block sizing
            # Extract unique x and y coordinates by parsing string indices
            xs = []
            ys = []
            for idx_str in self.indices:
                idx = self._parse_string_index(idx_str)
                if isinstance(idx, tuple) and len(idx) == 2:
                    xs.append(idx[0])
                    ys.append(idx[1])
            xs = sorted(set(xs))
            ys = sorted(set(ys))

            # Calculate grid spacing (block size)
            x_spacing = xs[1] - xs[0] if len(xs) > 1 else 1
            y_spacing = ys[1] - ys[0] if len(ys) > 1 else 1

            # Create coordinate arrays including edges for pcolormesh
            # Add half-spacing to create cell edges
            x_edges = np.array(xs) - x_spacing / 2
            x_edges = np.append(x_edges, xs[-1] + x_spacing / 2)
            y_edges = np.array(ys) - y_spacing / 2
            y_edges = np.append(y_edges, ys[-1] + y_spacing / 2)

            # Create 2D array for values
            trans_array = np.full((len(ys), len(xs)), np.nan)

            # Map indices to array positions
            x_map = {x: i for i, x in enumerate(xs)}
            y_map = {y: i for i, y in enumerate(ys)}

            for idx_str in self.indices:
                idx = self._parse_string_index(idx_str)
                if isinstance(idx, tuple) and len(idx) == 2:
                    x, y = idx
                    if x in x_map and y in y_map:
                        trans_array[y_map[y], x_map[x]] = avg_trans[idx_str]

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                trans_array,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="flat",
                **kwargs,
            )
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.set_aspect("equal")

            # Set appropriate title and colorbar label based on mode
            if title is None:
                if logT:
                    title = f"Thickness Estimate Map ({emin:.2g}-{emax:.2g} eV)"
                elif emin2 is not None:
                    title = f"Transmission Ratio Map ({emin:.2g}-{emax:.2g})/{emin2:.2g}-{emax2:.2g} eV)"
                else:
                    title = f"Average Transmission Map ({emin:.2g}-{emax:.2g} eV)"

            if logT:
                cbar_label = "Thickness [cm]"
            elif emin2 is not None:
                cbar_label = "Transmission Ratio"
            else:
                cbar_label = "Transmission"

            ax.set_title(title)
            plt.colorbar(im, ax=ax, label=cbar_label)
            return ax

        if self.group_shape and len(self.group_shape) == 1:
            # 1D line plot - parse string indices back to integers
            indices_array = np.array(
                [self._parse_string_index(idx) for idx in self.indices]
            )
            trans_values = np.array([avg_trans[idx] for idx in self.indices])

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(indices_array, trans_values, "o-", **kwargs)
            ax.set_xlabel("Pixel index")

            # Set appropriate ylabel and title based on mode
            if logT:
                ax.set_ylabel("Thickness [cm]")
                if title is None:
                    title = f"Thickness Estimate ({emin:.2g}-{emax:.2g} eV)"
            elif emin2 is not None:
                ax.set_ylabel("Transmission Ratio")
                if title is None:
                    title = f"Transmission Ratio ({emin:.2g}-{emax:.2g})/({emin2:.2g}-{emax2:.2g} eV)"
            else:
                ax.set_ylabel("Average Transmission")
                if title is None:
                    title = f"Average Transmission ({emin:.2g}-{emax:.2g} eV)"

            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            return ax

        # Bar chart for named indices
        fig, ax = plt.subplots(figsize=figsize)
        positions = np.arange(len(self.indices))
        trans_values = [avg_trans[idx] for idx in self.indices]

        ax.bar(positions, trans_values, **kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels(self.indices, rotation=45, ha="right")

        # Set appropriate ylabel and title based on mode
        if logT:
            ax.set_ylabel("Thickness [cm]")
            if title is None:
                title = f"Thickness Estimate ({emin:.2g}-{emax:.2g} eV)"
        elif emin2 is not None:
            ax.set_ylabel("Transmission Ratio")
            if title is None:
                title = f"Transmission Ratio ({emin:.2g}-{emax:.2g})/({emin2:.2g}-{emax2:.2g} eV)"
        else:
            ax.set_ylabel("Average Transmission")
            if title is None:
                title = f"Average Transmission ({emin:.2g}-{emax:.2g} eV)"

        ax.set_title(title)
        plt.tight_layout()
        return ax

    def rebin(self, n=None, tstep=None):
        """
        Rebin the time-of-flight data by combining bins or using a new time step.

        This method creates a new Data object with rebinned counts data, properly
        recalculated uncertainties, and updated transmission values. Works for both
        grouped and non-grouped data.

        Parameters:
        -----------
        n : int, optional
            Number of original bins to combine into one new bin.
            E.g., n=2 combines every 2 bins, n=4 combines every 4 bins.
            Mutually exclusive with tstep.
        tstep : float, optional
            New time step in seconds. Uses linear interpolation if the new bins
            don't align with the original bins.
            Mutually exclusive with n.
        linear_bins : bool, optional
            If True (default), creates linearly-spaced bins when using tstep parameter.
            If False, creates logarithmically-spaced bins in energy space, which is
            more appropriate for cross-section data that varies logarithmically with energy.
            Note: The cross-section C++ code assumes linear binning in time/energy,
            so use linear_bins=True for compatibility with the response function integration.
            Only affects tstep method, ignored for n method.

        Returns:
        --------
        Data
            A new Data object with rebinned data. All attributes (signal, openbeam,
            table, tgrid, etc.) are properly updated.

        Raises:
        -------
        ValueError
            If both n and tstep are provided, or if neither is provided.
            If the Data object doesn't have original counts data (signal/openbeam).
            If called on non-grouped data created from transmission files.

        Examples:
        ---------
        >>> # Combine every 4 bins
        >>> data_rebinned = data.rebin(n=4)

        >>> # Use a new time step (2x the original)
        >>> data_rebinned = data.rebin(tstep=2 * data.tstep)

        >>> # Works with grouped data too
        >>> grouped_data_rebinned = grouped_data.rebin(n=2)

        Notes:
        ------
        - Counts are summed in each new bin
        - Uncertainties are combined in quadrature: sqrt(sum(err^2))
        - Transmission and its error are recalculated from rebinned counts
        - Energy grid is recomputed from the new time-of-flight grid
        - For grouped data, rebinning is applied to all groups
        - **NaN handling**: Rows with NaN values in energy, transmission, or error
          columns are automatically removed before rebinning. This is safe because
          NaN values don't contribute to fits. If all data is NaN, an empty table
          is returned for that group.
        """
        # Validate input
        if n is None and tstep is None:
            raise ValueError(
                "Must specify either 'n' (number of bins to combine) or 'tstep' (new time step)"
            )
        if n is not None and tstep is not None:
            raise ValueError(
                "Cannot specify both 'n' and 'tstep'. Choose one rebinning method."
            )

        # Check that we have original counts data
        if not self.is_grouped and (self.signal is None or self.openbeam is None):
            raise ValueError(
                "Cannot rebin: original counts data (signal/openbeam) not available. "
                "This Data object was likely created from transmission files."
            )

        # Helper function to rebin a single counts DataFrame
        def rebin_counts_dataframe(
            df, n_bins=None, new_tstep=None, old_tstep=None, L=None
        ):
            """
            Rebin a counts DataFrame (tof, counts, err).

            Parameters:
            -----------
            df : pd.DataFrame
                Input DataFrame with 'tof', 'counts', 'err' columns
            n_bins : int, optional
                Number of bins to combine (simple binning)
            new_tstep : float, optional
                New time step (interpolation method)
            old_tstep : float, optional
                Original time step
            L : float, optional
                Flight path length for energy conversion

            Returns:
            --------
            pd.DataFrame
                Rebinned DataFrame
            """
            if n_bins is not None:
                # Simple binning: combine every n_bins
                if n_bins == 1:
                    # No rebinning needed, return copy of original
                    rebinned_df = df.copy()
                else:
                    n_original = len(df)
                    n_new = n_original // n_bins

                    # Truncate to make evenly divisible
                    df_truncated = df.iloc[: n_new * n_bins].copy()

                    # Reshape and sum
                    tof_reshaped = df_truncated["tof"].values.reshape(n_new, n_bins)
                    counts_reshaped = df_truncated["counts"].values.reshape(
                        n_new, n_bins
                    )
                    err_reshaped = df_truncated["err"].values.reshape(n_new, n_bins)

                    # New tof is the CENTER of each combined bin
                    # For bins [i, i+1, ..., i+n-1], the center is at (i + i+n-1 + 1) / 2
                    # This gives the correct energy for the rebinned data
                    new_tof = (tof_reshaped[:, 0] + tof_reshaped[:, -1] + 1) / 2
                    # Sum counts
                    new_counts = counts_reshaped.sum(axis=1)
                    # Combine errors in quadrature
                    new_err = np.sqrt((err_reshaped**2).sum(axis=1))

                    rebinned_df = pd.DataFrame(
                        {"tof": new_tof, "counts": new_counts, "err": new_err}
                    )

            else:
                # Interpolation method: new tstep
                # Check if we're using the same tstep
                if abs(new_tstep - old_tstep) / old_tstep < 1e-10:
                    # No rebinning needed, return copy of original
                    rebinned_df = df.copy()
                else:
                    from scipy.interpolate import interp1d

                    # Original tof grid in time units (seconds)
                    old_tof_time = df["tof"].values * old_tstep

                    # Create new tof grid with linear spacing
                    tof_min = old_tof_time.min()
                    tof_max = old_tof_time.max()
                    n_new_bins = int((tof_max - tof_min) / new_tstep)
                    new_tof_time = np.linspace(tof_min, tof_max, n_new_bins)

                    # Interpolate counts and errors
                    # For counts: linear interpolation (represents count rate)
                    counts_interp = interp1d(
                        old_tof_time,
                        df["counts"].values,
                        kind="linear",
                        fill_value=0,
                        bounds_error=False,
                    )
                    err_interp = interp1d(
                        old_tof_time,
                        df["err"].values,
                        kind="linear",
                        fill_value=0,
                        bounds_error=False,
                    )

                    new_counts = counts_interp(new_tof_time)
                    new_err = err_interp(new_tof_time)

                    # Scale by the ratio of bin widths to conserve total counts
                    bin_width_ratio = new_tstep / old_tstep
                    new_counts = new_counts * bin_width_ratio
                    new_err = new_err * bin_width_ratio

                    # Convert back to bin indices (bin centers)
                    new_tof = new_tof_time / new_tstep

                    rebinned_df = pd.DataFrame(
                        {"tof": new_tof, "counts": new_counts, "err": new_err}
                    )

            # Preserve label attribute if present
            if hasattr(df, "attrs") and "label" in df.attrs:
                rebinned_df.attrs["label"] = df.attrs["label"]

            return rebinned_df

        # Helper function to calculate transmission from rebinned counts
        def calculate_transmission(
            signal_df,
            openbeam_df,
            new_tstep_val,
            L_val,
            empty_signal_df=None,
            empty_openbeam_df=None,
        ):
            """Calculate transmission and energy from rebinned signal and openbeam."""
            # Convert tof to energy
            energy = utils.time2energy(signal_df["tof"].values * new_tstep_val, L_val)

            # Calculate transmission
            transmission = signal_df["counts"] / openbeam_df["counts"]

            # Calculate transmission error
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if empty_signal_df is not None and empty_openbeam_df is not None:
                    # Apply background correction
                    transmission *= (
                        empty_openbeam_df["counts"] / empty_signal_df["counts"]
                    )
                    trans_err = transmission * np.sqrt(
                        (signal_df["err"] / signal_df["counts"]) ** 2
                        + (openbeam_df["err"] / openbeam_df["counts"]) ** 2
                        + (empty_signal_df["err"] / empty_signal_df["counts"]) ** 2
                        + (empty_openbeam_df["err"] / empty_openbeam_df["counts"]) ** 2
                    )
                else:
                    trans_err = transmission * np.sqrt(
                        (signal_df["err"] / signal_df["counts"]) ** 2
                        + (openbeam_df["err"] / openbeam_df["counts"]) ** 2
                    )

            table_df = pd.DataFrame(
                {"energy": energy, "trans": transmission, "err": trans_err}
            )

            # Preserve label if present
            if hasattr(signal_df, "attrs") and "label" in signal_df.attrs:
                table_df.attrs["label"] = signal_df.attrs["label"]

            return table_df

        # Determine new tstep
        if tstep is not None:
            new_tstep = tstep
        else:
            # When using n-binning, keep tstep the same since TOF indices
            # are adjusted to bin centers
            new_tstep = self.tstep

        # Create new Data object
        new_data = Data()
        new_data.L = self.L
        new_data.tstep = new_tstep
        new_data.is_grouped = self.is_grouped

        # Copy L0 and t0 if they exist
        if hasattr(self, "L0"):
            new_data.L0 = self.L0
        if hasattr(self, "t0"):
            new_data.t0 = self.t0

        if self.is_grouped:
            # Rebin all groups
            new_data.groups = {}
            new_data.indices = self.indices
            new_data.group_shape = self.group_shape

            # For grouped data, we rebin the transmission tables directly
            # This is done by interpolating onto a new energy grid
            def rebin_transmission_table(
                table_df,
                n_bins=None,
                new_tstep_val=None,
                old_tstep_val=None,
                L_val=None,
            ):
                """
                Rebin a transmission table (energy, trans, err) for grouped data.

                Uses interpolation to map transmission values onto a new energy grid.
                Note: Uncertainties are interpolated, which is approximate. For best
                accuracy, rebin the original counts data before creating grouped data.
                """
                if n_bins is not None:
                    # Simple binning: combine every n_bins
                    if n_bins == 1:
                        return table_df.copy()

                    # Remove NaN values before binning
                    # NaN values don't contribute to fits, so it's safe to exclude them
                    clean_table = table_df.dropna(subset=["energy", "trans", "err"])

                    if len(clean_table) == 0:
                        # If all data is NaN, return empty table with same structure
                        return pd.DataFrame({"energy": [], "trans": [], "err": []})

                    n_original = len(clean_table)
                    n_new = n_original // n_bins

                    if n_new == 0:
                        # Not enough data points to bin, return as is
                        return clean_table.copy()

                    # Truncate to make evenly divisible
                    table_truncated = clean_table.iloc[: n_new * n_bins].copy()

                    # Reshape and average (for transmission, we average not sum)
                    energy_reshaped = table_truncated["energy"].values.reshape(
                        n_new, n_bins
                    )
                    trans_reshaped = table_truncated["trans"].values.reshape(
                        n_new, n_bins
                    )
                    err_reshaped = table_truncated["err"].values.reshape(n_new, n_bins)

                    # Use arithmetic mean for energy (centers of rebinned energy bins)
                    # nanmean to handle any remaining NaNs in the reshaped arrays
                    new_energy = np.nanmean(energy_reshaped, axis=1)
                    # Use arithmetic mean for transmission
                    new_trans = np.nanmean(trans_reshaped, axis=1)
                    # Combine errors: err_mean = sqrt(sum(err^2)) / n
                    # Use nansum to skip NaN values in error combination
                    new_err = np.sqrt(np.nansum(err_reshaped**2, axis=1)) / n_bins

                    rebinned_table = pd.DataFrame(
                        {"energy": new_energy, "trans": new_trans, "err": new_err}
                    )

                    # Remove any remaining NaN rows created by nanmean of all-NaN bins
                    rebinned_table = rebinned_table.dropna(
                        subset=["energy", "trans", "err"]
                    )

                else:
                    # Interpolation method: new tstep
                    from scipy.interpolate import interp1d

                    # Remove NaN values before rebinning
                    # NaN values don't contribute to fits, so it's safe to exclude them
                    clean_table = table_df.dropna(subset=["energy", "trans", "err"])

                    if len(clean_table) == 0:
                        # If all data is NaN, return empty table with same structure
                        return pd.DataFrame({"energy": [], "trans": [], "err": []})

                    # Current energy grid (without NaNs)
                    old_energy = clean_table["energy"].values

                    # Create new energy grid based on new tstep
                    # Convert energy back to TOF, apply new tstep, convert back
                    old_tof_time = utils.energy2time(old_energy, L_val)
                    tof_min = old_tof_time.min()
                    tof_max = old_tof_time.max()
                    n_new_bins = int((tof_max - tof_min) / new_tstep_val)

                    if n_new_bins <= 0:
                        # Handle edge case where time range is too small
                        n_new_bins = 1

                    new_tof_time = np.linspace(tof_min, tof_max, n_new_bins)
                    new_energy = utils.time2energy(new_tof_time, L_val)

                    # Interpolate transmission and error
                    # Use bounds_error=False with fill_value=nan to avoid extrapolating into NaN regions
                    trans_interp = interp1d(
                        old_energy,
                        clean_table["trans"].values,
                        kind="linear",
                        fill_value=np.nan,
                        bounds_error=False,
                    )
                    err_interp = interp1d(
                        old_energy,
                        clean_table["err"].values,
                        kind="linear",
                        fill_value=np.nan,
                        bounds_error=False,
                    )

                    new_trans = trans_interp(new_energy)
                    new_err = err_interp(new_energy)

                    # Create rebinned table
                    rebinned_table = pd.DataFrame(
                        {"energy": new_energy, "trans": new_trans, "err": new_err}
                    )

                    # Remove any NaN rows that may have been created during interpolation
                    rebinned_table = rebinned_table.dropna(
                        subset=["energy", "trans", "err"]
                    )

                # Preserve label if present
                if hasattr(table_df, "attrs") and "label" in table_df.attrs:
                    rebinned_table.attrs["label"] = table_df.attrs["label"]

                return rebinned_table

            # Rebin each group
            for idx in self.indices:
                group_table = self.groups[idx]
                rebinned_table = rebin_transmission_table(
                    group_table,
                    n_bins=n,
                    new_tstep_val=tstep,
                    old_tstep_val=self.tstep,
                    L_val=self.L,
                )
                new_data.groups[idx] = rebinned_table

            # Update the main table to be the first group
            new_data.table = new_data.groups[self.indices[0]]

            # Also update grouped_trans and grouped_err arrays
            # Handle case where groups may have different lengths due to NaN removal
            if len(new_data.groups[self.indices[0]]) > 0:
                n_groups = len(self.indices)
                n_energy = len(new_data.groups[self.indices[0]])
                new_trans_array = np.zeros((n_groups, n_energy))
                new_err_array = np.zeros((n_groups, n_energy))

                for i, idx in enumerate(self.indices):
                    group_len = len(new_data.groups[idx])
                    if group_len > 0:
                        # Handle case where this group might have different length
                        new_trans_array[i, : min(group_len, n_energy)] = (
                            new_data.groups[idx]["trans"].values[:n_energy]
                        )
                        new_err_array[i, : min(group_len, n_energy)] = new_data.groups[
                            idx
                        ]["err"].values[:n_energy]
                    else:
                        # Empty group - fill with NaN
                        new_trans_array[i, :] = np.nan
                        new_err_array[i, :] = np.nan

                new_data.grouped_trans = new_trans_array
                new_data.grouped_err = new_err_array
            else:
                # All groups are empty - create empty arrays
                new_data.grouped_trans = np.array([]).reshape(len(self.indices), 0)
                new_data.grouped_err = np.array([]).reshape(len(self.indices), 0)

        else:
            # Rebin non-grouped data
            rebinned_signal = rebin_counts_dataframe(
                self.signal, n_bins=n, new_tstep=tstep, old_tstep=self.tstep, L=self.L
            )
            rebinned_openbeam = rebin_counts_dataframe(
                self.openbeam, n_bins=n, new_tstep=tstep, old_tstep=self.tstep, L=self.L
            )

            # Rebin empty data if it exists (for background correction)
            rebinned_empty_signal = None
            rebinned_empty_openbeam = None
            if hasattr(self, "empty_signal") and self.empty_signal is not None:
                rebinned_empty_signal = rebin_counts_dataframe(
                    self.empty_signal,
                    n_bins=n,
                    new_tstep=tstep,
                    old_tstep=self.tstep,
                    L=self.L,
                )
            if hasattr(self, "empty_openbeam") and self.empty_openbeam is not None:
                rebinned_empty_openbeam = rebin_counts_dataframe(
                    self.empty_openbeam,
                    n_bins=n,
                    new_tstep=tstep,
                    old_tstep=self.tstep,
                    L=self.L,
                )

            # Calculate transmission table
            new_table = calculate_transmission(
                rebinned_signal,
                rebinned_openbeam,
                new_tstep,
                self.L,
                rebinned_empty_signal,
                rebinned_empty_openbeam,
            )

            new_data.signal = rebinned_signal
            new_data.openbeam = rebinned_openbeam
            new_data.table = new_table
            new_data.tgrid = rebinned_signal["tof"]

            # Store rebinned empty data
            new_data.empty_signal = rebinned_empty_signal
            new_data.empty_openbeam = rebinned_empty_openbeam

        return new_data
