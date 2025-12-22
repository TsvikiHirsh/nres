from nres import utils
import pandas as pd
import numpy as np
import warnings
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
        df = pd.read_csv(filename, names=["tof", "counts", "err"], header=None, skiprows=1)
        
        # If no error values provided, calculate as sqrt of counts
        if all(df["err"].isnull()):
            df["err"] = np.sqrt(df["counts"])
        
        # Store label from filename (without path and extension)
        df.attrs["label"] = filename.split("/")[-1].rstrip(".csv")
        
        return df
    
    @classmethod
    def from_counts(cls, signal: str, openbeam: str,
                    empty_signal: str = "", empty_openbeam: str = "",
                    tstep: float = 1.56255e-9, L: float = 10.59,
                    L0: float = 1.0, t0: float = 0.0, verbosity: int = 1):
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
            err = transmission * np.sqrt((signal["err"] / signal["counts"])**2 +
                                         (openbeam["err"] / openbeam["counts"])**2)

            # If background (empty) data is provided, apply correction
            if empty_signal and empty_openbeam:
                empty_signal = cls._read_counts(empty_signal)
                empty_openbeam = cls._read_counts(empty_openbeam)

                transmission *= empty_openbeam["counts"] / empty_signal["counts"]
                err = transmission * np.sqrt(
                    (signal["err"] / signal["counts"])**2 +
                    (openbeam["err"] / openbeam["counts"])**2 +
                    (empty_signal["err"] / empty_signal["counts"])**2 +
                    (empty_openbeam["err"] / empty_openbeam["counts"])**2
                )
        
        # Construct a dataframe for energy, transmission, and error
        df = pd.DataFrame({
            "energy": signal["energy"],
            "trans": transmission,
            "err": err
        })
        
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
        elif isinstance(index, str):
            # Remove spaces from string if it looks like a tuple: "(10, 20)" -> "(10,20)"
            return index.replace(" ", "")
        else:
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
        import re
        import os

        indices = []

        # Auto-detect pattern if needed
        if pattern == "auto":
            # Try common patterns
            test_name = os.path.basename(filenames[0])

            # Try 2D patterns - look for _x or _y to avoid matching dimension specs like 16x16
            match_2d = re.search(r'_x(\d+).*_y(\d+)', test_name, re.IGNORECASE)
            if not match_2d:
                # Try without underscores but with word boundaries
                match_2d = re.search(r'\bx(\d+)[_\s].*\by(\d+)', test_name, re.IGNORECASE)

            if match_2d:
                pattern = "_x{x}_y{y}"
            else:
                # Try 1D patterns - look for trailing numbers or with keywords
                match_1d = re.search(r'(?:idx|pixel|det)[_\s]*(\d+)', test_name, re.IGNORECASE)
                if not match_1d:
                    # Try just trailing number before extension
                    match_1d = re.search(r'_(\d+)\.', test_name)

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
                match = re.search(r'_x(\d+).*_y(\d+)', basename, re.IGNORECASE)
                if not match:
                    # Try with word boundaries
                    match = re.search(r'\bx(\d+)[_\s].*\by(\d+)', basename, re.IGNORECASE)
                if not match:
                    # Try simple pattern as last resort
                    match = re.search(r'(?<![\dx])x(\d+).*(?<![\dx])y(\d+)', basename, re.IGNORECASE)

                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    indices.append((x, y))
                else:
                    raise ValueError(f"Could not extract x,y coordinates from: {basename}. "
                                   f"Filename should contain _x<num> and _y<num> patterns.")

            elif "{i}" in pattern:
                # 1D array pattern - try multiple approaches
                # First try with keywords
                match = re.search(r'(?:idx|pixel|det)[_\s]*(\d+)', basename, re.IGNORECASE)
                if not match:
                    # Try trailing number before extension
                    match = re.search(r'_(\d+)\.', basename)
                if not match:
                    # Last resort - any number in the filename (rightmost)
                    matches = re.findall(r'(\d+)', basename)
                    if matches:
                        match = type('obj', (object,), {'group': lambda self, n: matches[-1]})()

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
        elif isinstance(first_idx, (int, int.__class__)):
            # 1D array
            return (len(indices),), False, True

        # Named indices (strings)
        else:
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
        df = pd.read_csv(filename, names=["energy", "trans", "err"], header=None,
                         skiprows=0, sep=r"\s+")
        
        # Create Data object and assign the dataframe
        self_data = cls()
        self_data.table = df
        
        return self_data

    @classmethod
    def from_grouped(cls, signal, openbeam,
                     empty_signal: str = "", empty_openbeam: str = "",
                     tstep: float = 1.56255e-9, L: float = 10.59,
                     L0: float = 1.0, t0: float = 0.0,
                     pattern: str = "auto", indices: list = None, verbosity: int = 1,
                     n_jobs: int = -1):
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
        >>> data = Data.from_grouped("data/det_*.csv", "data_ob/det_*.csv", indices=[0, 1, 2, 3])

        # Named groups
        >>> data = Data.from_grouped("samples/*.csv", "ref/*.csv", indices=["sample1", "sample2"])
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
            raise ValueError(f"Mismatch: {len(signal_files)} signal files vs {len(openbeam_files)} openbeam files")

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
            elif len(empty_signal_files) != len(signal_files) or len(empty_openbeam_files) != len(signal_files):
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
                raise ValueError(f"Number of indices ({len(indices)}) must match number of files ({len(signal_files)})")
            extracted_indices = indices
        else:
            # Auto-extract from filenames
            extracted_indices = cls._extract_indices_from_filenames(signal_files, pattern)

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
                verbosity=verbosity
            )

            return idx, group_data.table

        # Load groups in parallel or sequentially
        if n_jobs == 1:
            # Sequential loading with progress bar
            if verbosity >= 1:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(enumerate(extracted_indices), total=len(extracted_indices),
                                   desc=f"Loading {len(extracted_indices)} groups")
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
                    pbar = tqdm(total=len(extracted_indices), desc=f"Loading {len(extracted_indices)} groups")
                except ImportError:
                    pbar = None
            else:
                pbar = None

            # Load groups in parallel using threading backend
            # (threading is appropriate for I/O-bound file loading and avoids serialization issues)
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
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
        ylim = kwargs.pop("ylim", (0., 1.))
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
                raise ValueError(f"Index {index} not found in groups. Available indices: {self.indices}")
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
        ax = table_to_plot.dropna().plot(x="energy", y="trans", yerr="err",
                                        xlim=xlim, ylim=ylim, logx=logx, ecolor=ecolor,
                                        xlabel=xlabel, ylabel=ylabel, label=plot_label, **kwargs)

        # Add legend if label was set
        if plot_label is not None:
            ax.legend()

        return ax

    def plot_map(self, emin=0.5e6, emax=20e6, **kwargs):
        """
        Plot transmission map averaged over energy range for grouped data.

        Parameters:
        -----------
        emin : float, optional
            Minimum energy for averaging (default: 0.5e6 eV).
        emax : float, optional
            Maximum energy for averaging (default: 20e6 eV).
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
            If called on non-grouped data.

        Examples:
        ---------
        >>> # For 2D grid data
        >>> data = Data.from_grouped("pixel_x*_y*.csv", "ob_x*_y*.csv")
        >>> data.plot_map(emin=1e6, emax=10e6)

        >>> # For 1D array data
        >>> data = Data.from_grouped("pixel_*.csv", "ob_*.csv")
        >>> data.plot_map(emin=0.5e6, emax=5e6)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.is_grouped:
            raise ValueError("plot_map only works for grouped data")

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
            mask = (table['energy'] >= emin) & (table['energy'] <= emax)
            avg_trans[idx] = table.loc[mask, 'trans'].mean()

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
            im = ax.pcolormesh(x_edges, y_edges, trans_array, cmap=cmap, vmin=vmin, vmax=vmax,
                              shading='flat', **kwargs)
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.set_aspect('equal')
            if title is None:
                title = f"Average Transmission Map ({emin:.2g}-{emax:.2g} eV)"
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="Transmission")
            return ax

        elif self.group_shape and len(self.group_shape) == 1:
            # 1D line plot - parse string indices back to integers
            indices_array = np.array([self._parse_string_index(idx) for idx in self.indices])
            trans_values = np.array([avg_trans[idx] for idx in self.indices])

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(indices_array, trans_values, 'o-', **kwargs)
            ax.set_xlabel("Pixel index")
            ax.set_ylabel("Average Transmission")
            if title is None:
                title = f"Average Transmission ({emin:.2g}-{emax:.2g} eV)"
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            return ax

        else:
            # Bar chart for named indices
            fig, ax = plt.subplots(figsize=figsize)
            positions = np.arange(len(self.indices))
            trans_values = [avg_trans[idx] for idx in self.indices]

            ax.bar(positions, trans_values, **kwargs)
            ax.set_xticks(positions)
            ax.set_xticklabels(self.indices, rotation=45, ha='right')
            ax.set_ylabel("Average Transmission")
            if title is None:
                title = f"Average Transmission ({emin:.2g}-{emax:.2g} eV)"
            ax.set_title(title)
            plt.tight_layout()
            return ax