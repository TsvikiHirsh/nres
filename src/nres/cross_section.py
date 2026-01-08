from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

import nres
import nres.utils as utils
from nres._integrate_xs import integrate_cross_section


class CrossSection:
    """
    A class representing neutron cross-sections for single or multiple isotopes.

    This class handles the loading, manipulation, and analysis of neutron cross-section data.
    It supports operations on individual isotopes as well as combinations of isotopes with
    different weights. The class provides functionality for interpolation and calculation
    of total cross-sections based on weighted sums.

    Attributes:
        isotopes (Dict[Union[str, 'CrossSection'], float]): Dictionary mapping isotope names
            or CrossSection objects to their respective weights.
        name (str): Identifier for this cross-section combination.
        weights (pd.Series): Normalized weights for each isotope, ensuring they sum to 1.
        table (pd.DataFrame): DataFrame containing the interpolated cross-section data.
            Includes columns for each isotope and a 'total' column.
        L (float): Flight path length in meters.
        tstep (float): Time step for the simulation in seconds.
        tbins (int): Number of time bins for the simulation.
        first_tbin (int): Index of the first time bin (typically 1).
        n (float): Number density calculated based on isotope weights.
        materials (Dict): Dictionary containing material information and properties.
    """

    def __init__(
        self,
        isotopes: Dict[Union[str, CrossSection], float] = None,
        name: str = "",
        total_weight: float = 1.0,
        L: float = 10.59,
        tstep: float = 1.56255e-9,
        tbins: int = 640,
        first_tbin: int = 1,
        splitby: str = "elements",
        **materials,
    ):
        """Initialize a new CrossSection instance.

        Args:
            isotopes: Dictionary mapping isotope names/CrossSection objects to weights,
                     or a CrossSection object to copy
            name: Identifier for this cross-section combination
            total_weight: Overall scaling factor for the cross-section
            L: Flight path length in meters
            tstep: Time step for simulation in seconds
            tbins: Number of time bins
            first_tbin: Index of the first time bin
            splitby: How to split cross sections ("isotopes", "elements", "materials")
            **materials: Additional materials to initialize with
        """
        # Initialize basic attributes
        self.name = name
        self.L = L
        self.tstep = tstep
        self.tbins = tbins
        self.first_tbin = first_tbin
        self.tgrid = np.arange(self.first_tbin, self.tbins + 1, 1) * self.tstep

        # Store the original materials and their properties
        self.materials = {}
        self.__xsdata__ = None
        self._load_xsdata()

        # Initialize empty table and weights
        self.table = pd.DataFrame()
        self.weights = pd.Series(dtype=float)
        self.isotopes = {}
        self.n = 0.0

        if isinstance(isotopes, CrossSection):
            self._init_from_cross_section(isotopes, name)
        elif isotopes:
            material_data = self._get_material_data(isotopes)
            self.add_material(
                name or "material_1", material_data, splitby, total_weight
            )

        # Handle additional materials passed as keyword arguments
        for mat_name, mat_data in materials.items():
            material_data = self._get_material_data(mat_data)
            self.add_material(mat_name, material_data, splitby, total_weight)

    def _load_xsdata(self):
        """Load cross-section data from file.

        First tries to use trinidi_data package if installed.
        If not found, downloads xsdata.npy to a local cache directory.
        Only downloads if the file doesn't already exist in the cache.
        """
        if self.__xsdata__ is None:
            # Try to import trinidi_data package
            try:
                import trinidi_data

                data_path = os.path.join(
                    os.path.dirname(trinidi_data.__file__), "xsdata.npy"
                )
            except ImportError:
                # trinidi_data not installed, use local cache
                import platformdirs

                cache_dir = platformdirs.user_cache_dir("nres", "nres")
                os.makedirs(cache_dir, exist_ok=True)
                data_path = os.path.join(cache_dir, "xsdata.npy")

                # Download if file doesn't exist
                if not os.path.exists(data_path):
                    import requests

                    url = "https://github.com/TsvikiHirsh/trinidi-data/raw/main/trinidi_data/xsdata.npy"
                    print(f"Downloading cross-section data to {data_path}...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()

                    with open(data_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print("Download complete!")
                # else: File already exists, skip download

            xsdata = np.load(data_path, allow_pickle=True)[()]
            self.__xsdata__ = {
                isotope.replace("-", ""): pd.Series(
                    xsdata["cross_sections"][i],
                    index=xsdata["energies"][i],
                    name=isotope.replace("-", ""),
                )
                for i, isotope in enumerate(xsdata["isotopes"])
            }

    def _get_material_data(self, material: Union[str, Dict]) -> Dict:
        """Convert material string or dict to full material data structure."""
        if isinstance(material, str):
            formulas = {
                nres.materials[element]["formula"]: nres.materials[element]["name"]
                for element in nres.materials
            }
            try:  # Try materials database
                material = nres.materials[formulas.get(material, material)]
            except KeyError:  # Try elements database
                try:
                    formulas = {
                        nres.elements[element]["formula"]: nres.elements[element][
                            "name"
                        ]
                        for element in nres.elements
                    }
                    material = nres.elements[
                        formulas.get(material.capitalize(), material.capitalize())
                    ]
                except KeyError:  # Try isotopes database
                    material = nres.isotopes[material.capitalize().replace("-", "")]
        return material

    def _set_weights(self, weights: Optional[List[float]] = None):
        """Set and normalize weights for all isotopes.

        Args:
            weights: Optional list of new weights. If provided, must match
                    the number of isotopes.

        Raises:
            ValueError: If the number of provided weights doesn't match
                       the number of isotopes.
        """
        if weights is not None:
            if len(weights) != len(self.isotopes):
                raise ValueError("Number of weights must match number of isotopes")

            self.weights = pd.Series(weights, index=self.isotopes.keys())
        else:
            self.weights = pd.Series(self.isotopes)

        # Remove zero-weight isotopes and normalize
        self.weights = self.weights[self.weights > 0]
        self.weights /= self.weights.sum()

        # Update total cross-section with new weights
        self.table["total"] = (
            (self.table.drop(columns="total", errors="ignore") * self.weights)
            .sum(axis=1)
            .astype(float)
        )

        # Update total atomic density
        self.n = self._update_atomic_density()

    def add_material(
        self,
        name: str,
        material_data: Dict,
        splitby: str = "elements",
        total_weight: float = 1.0,
    ):
        """
        Add a new material with complete information.

        Args:
            name: str, name of the material
            material_data: Dict, material composition data
            splitby: str, how to split the material ('elements', 'isotopes', or 'materials')
            total_weight: float, total weight of the material
        """
        # Deep copy the material data to prevent modifications to the original
        self.materials[name] = deepcopy(material_data)
        self.materials[name]["splitby"] = splitby
        self.materials[name]["total_weight"] = total_weight

        # Collect all existing energy grids
        energy_grids = []

        # Add current table's energy grid if it exists
        if hasattr(self, "table") and len(self.table) > 0:
            energy_grids.append(self.table.index)

        # Add energy grids from the new material's cross sections
        for element_info in material_data["elements"].values():
            for isotope in element_info["isotopes"]:
                isotope_clean = isotope.replace("-", "")
                if isotope_clean in self.__xsdata__:
                    energy_grids.append(self.__xsdata__[isotope_clean].index)

        # If we have any energy grids, merge them
        if energy_grids:
            # Create a merged grid that includes all unique energy points
            merged_grid = pd.Index(sorted(set().union(*energy_grids)))
            self._energy_grid = merged_grid

        # Recalculate cross sections with the updated energy grid
        self._recalculate_cross_sections()

    def _update_atomic_density(self) -> float:
        """Calculate and update the total atomic density."""
        new_n = 0

        for material_name, material_info in self.materials.items():
            material_weight = material_info["total_weight"]
            splitby = material_info["splitby"]

            if splitby == "isotopes":
                for element_info in material_info["elements"].values():
                    for isotope, weight in element_info["isotopes"].items():
                        isotope_clean = isotope.replace("-", "")
                        if isotope_clean in self.weights.index:
                            new_n += (
                                material_info["n"]
                                * self.weights[isotope_clean]
                                * material_weight
                            )

            elif splitby == "elements":
                for element, element_info in material_info["elements"].items():
                    if element in self.weights.index:
                        new_n += (
                            material_info["n"] * self.weights[element] * material_weight
                        )

            elif splitby == "materials":
                if material_name in self.weights.index:
                    new_n += (
                        material_info["n"]
                        * self.weights[material_name]
                        * material_weight
                    )

        return new_n

    def _recalculate_cross_sections(self):
        """Calculate cross sections based on material information."""
        if not self.materials:
            return

        cross_sections = {}
        combined_weights = {}

        material_xs = {}
        material_weights = {}

        for material_name, material_info in self.materials.items():
            splitby = material_info["splitby"]
            total_weight = material_info["total_weight"]

            if splitby == "isotopes":
                for element_info in material_info["elements"].values():
                    for isotope, weight in element_info["isotopes"].items():
                        isotope_clean = isotope.replace("-", "")
                        if isotope_clean in self.__xsdata__:
                            cross_sections[isotope_clean] = self.__xsdata__[
                                isotope_clean
                            ]
                            combined_weights[isotope_clean] = weight * total_weight

            elif splitby == "elements":
                for element, element_info in material_info["elements"].items():
                    element_xs_dict = {}
                    element_weights = {}
                    for isotope, weight in element_info["isotopes"].items():
                        isotope_clean = isotope.replace("-", "")
                        if isotope_clean in self.__xsdata__:
                            # Store the raw cross section data instead of multiplying by weight here
                            element_xs_dict[isotope_clean] = self.__xsdata__[
                                isotope_clean
                            ]
                            element_weights[isotope_clean] = weight * total_weight

                    if len(element_xs_dict) > 0:
                        # Convert dictionary of Series to list of DataFrames for interleave_xs_energies
                        element_xs_dfs = [
                            pd.DataFrame({name: xs})
                            for name, xs in element_xs_dict.items()
                        ]
                        # Use the new interleave_xs_energies function
                        element_xs = self._interleave_xs_energies(element_xs_dfs)

                        # Now apply the weights after interpolation
                        element_weights = pd.Series(element_weights)
                        element_weights /= element_weights.sum()

                        # Calculate weighted sum for the element
                        total = pd.Series(0.0, index=element_xs.index)
                        for col in element_xs.columns:
                            isotope = col.split("_iso_")[
                                0
                            ]  # Get original isotope name from column
                            if isotope in element_weights:
                                total += element_xs[col] * element_weights[isotope]

                        cross_sections[element] = total
                        combined_weights[element] = (
                            element_info["weight"] * total_weight
                        )

            elif splitby == "materials":
                material_xs_dict = {}
                material_weights = {}
                for element_info in material_info["elements"].values():
                    for isotope, weight in element_info["isotopes"].items():
                        isotope_clean = isotope.replace("-", "")
                        if isotope_clean in self.__xsdata__:
                            # Store raw cross section data
                            material_xs_dict[isotope_clean] = self.__xsdata__[
                                isotope_clean
                            ]
                            material_weights[isotope_clean] = weight * total_weight

                if len(material_xs_dict) > 0:
                    # Convert dictionary of Series to list of DataFrames for interleave_xs_energies
                    material_xs_dfs = [
                        pd.DataFrame({name: xs})
                        for name, xs in material_xs_dict.items()
                    ]
                    # Use the new interleave_xs_energies function
                    material_xs = self._interleave_xs_energies(material_xs_dfs)

                    # Now apply the weights after interpolation
                    material_weights = pd.Series(material_weights)
                    material_weights /= material_weights.sum()

                    # Calculate weighted sum for the material
                    total = pd.Series(0.0, index=material_xs.index)
                    for col in material_xs.columns:
                        isotope = col.split("_iso_")[
                            0
                        ]  # Get original isotope name from column
                        if isotope in material_weights:
                            total += material_xs[col] * material_weights[isotope]

                    cross_sections[material_name] = total
                    combined_weights[material_name] = total_weight

        if cross_sections:
            # Convert all cross sections to DataFrames for final interpolation
            xs_dfs = [pd.DataFrame({name: xs}) for name, xs in cross_sections.items()]
            combined_table = self._interleave_xs_energies(xs_dfs)

            # If we have a stored energy grid, reindex and interpolate
            if hasattr(self, "_energy_grid"):
                combined_table = combined_table.reindex(self._energy_grid)
                combined_table = self._interleave_xs_energies([combined_table])

            total_weight = sum(combined_weights.values())
            combined_weights = {
                k: v / total_weight for k, v in combined_weights.items()
            }

            self.table = combined_table
            self.table.index.name = "energy"

            self.weights = pd.Series(combined_weights)

            weight_series = pd.Series(0.0, index=self.table.columns)
            for col in self.table.columns:
                base_col = col.split("_iso_")[
                    0
                ]  # Handle new column naming from interleave_xs_energies
                if base_col in self.weights:
                    weight_series[col] = self.weights[base_col]

            # Calculate total weighted cross section
            self.table["total"] = (self.table * weight_series).sum(axis=1).astype(float)
            self.isotopes = self.weights.to_dict()

            self.n = self._update_atomic_density()

    def __add__(self, other: CrossSection) -> CrossSection:
        """Add two CrossSection objects together."""
        new_self = deepcopy(self)

        # Store current energy grids
        energy_grids = []
        if hasattr(self, "table") and self.table is not None:
            energy_grids.append(self.table.index)
        if hasattr(other, "table") and other.table is not None:
            energy_grids.append(other.table.index)

        # Combine materials
        for mat_name, mat_info in other.materials.items():
            new_mat = deepcopy(mat_info)
            new_mat["total_weight"] = mat_info["total_weight"]
            new_self.add_material(
                name=mat_name,
                material_data=new_mat,
                splitby=mat_info["splitby"],
                total_weight=new_mat["total_weight"],
            )

        # Store merged grid if available
        if energy_grids:
            merged_grid = pd.Index(sorted(set().union(*energy_grids)))
            new_self._energy_grid = merged_grid

        new_self._recalculate_cross_sections()
        return new_self

    def __mul__(self, total_weight: float = 1.0) -> CrossSection:
        """Scale the CrossSection by a total weight factor."""
        new_self = deepcopy(self)
        new_self.total_weight = total_weight

        for material_name in new_self.materials:
            new_self.materials[material_name]["total_weight"] *= total_weight

        new_self._recalculate_cross_sections()

        return new_self

    def __rmul__(self, total_weight: float = 1.0) -> CrossSection:
        """Right multiplication to support scalar * CrossSection."""
        return self.__mul__(total_weight)

    def __call__(
        self,
        E: np.ndarray,
        weights: Optional[np.ndarray] = None,
        response: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the weighted cross-section for given energy values."""
        if weights is not None:
            self._set_weights(weights=weights)
        if response is None:
            response = [1.0]
        return np.array(
            integrate_cross_section(
                self.table["total"].index.values,
                self.table["total"].values,
                E,
                response,
            )
        )

    def _interleave_xs_energies(self, xs_data):
        """
        Interleave cross section data from different isotopes by interpolating across their
        combined energy grid points.

        Args:
            xs_data: DataFrame or Series containing cross section data with energy index,
                    or a list of such DataFrames/Series
        Returns:
            DataFrame with combined energy grid and interpolated cross section values
        """
        if isinstance(xs_data, pd.Series):
            return self._interleave_xs_energies(pd.DataFrame(xs_data)).iloc[:, 0]

        # If input is a single DataFrame, wrap it in a list
        if not isinstance(xs_data, list):
            xs_data = [xs_data]

        # Combine all unique energy points from all cross section data
        all_energies = sorted(set().union(*[df.index for df in xs_data]))

        # Create a new DataFrame with the combined energy grid
        result = pd.DataFrame(index=all_energies)

        # Add data from each cross section dataset
        for i, xs_df in enumerate(xs_data):
            if isinstance(xs_df, pd.Series):
                xs_df = xs_df.to_frame(f"xs_{i}")

            # Reindex to include all energy points
            xs_reindexed = xs_df.reindex(all_energies)

            # Interpolate missing values for each cross section column
            for col in xs_reindexed.columns:
                # Keep original column name but add isotope identifier if needed
                result[f"{col}" if len(xs_data) > 1 else col] = xs_reindexed[
                    col
                ].interpolate(method="linear")

        return result

    def plot(self, **kwargs):
        """
        Create a matplotlib plot of the cross-section data.

        Generates a plot showing the total cross-section and individual
        isotope contributions, with weights displayed in the legend.

        Args:
            **kwargs: Additional keyword arguments for plot customization:
                - title: Plot title (default: self.name)
                - ylabel: Y-axis label (default: "σ [barn]")
                - xlabel: X-axis label (default: "Energy [eV]")
                - lw: Line width (default: 1.0)
                - Additional arguments passed to pandas.DataFrame.plot

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot

        Note:
            The total cross-section is plotted in black with increased line width
            for emphasis. Individual isotope contributions are plotted in different
            colors with their weights shown as percentages in the legend.
        """
        import matplotlib.pyplot as plt

        title = kwargs.pop("title", self.name)
        ylabel = kwargs.pop("ylabel", r"$\sigma$ [barn]")
        xlabel = kwargs.pop("xlabel", "Energy [eV]")
        lw = kwargs.pop("lw", 1.0)

        # Apply weights and format column labels with percentage contributions
        table = self.table.mul(np.r_[self.weights, 1.0], axis=1)
        table.columns = [
            f"{column}: {weight*100:>6.2f}%" for column, weight in self.weights.items()
        ] + ["total"]

        fig, ax = plt.subplots()
        # Plot total cross-section with emphasis
        table.plot(y="total", linewidth=1.5, ax=ax, color="0.2", zorder=100, **kwargs)
        # Plot individual contributions
        table.drop("total", axis=1).plot(
            ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, linewidth=lw, **kwargs
        )
        return ax

    def iplot(self, **kwargs):
        """
        Create an interactive plotly plot of the cross-section data.

        Generates an interactive plot showing the total cross-section and
        individual isotope contributions, with customizable axes scales
        and energy range.

        Args:
            **kwargs: Additional keyword arguments for plot customization:
                - title: Plot title (default: self.name)
                - ylabel: Y-axis label (default: "σ [barn]")
                - xlabel: X-axis label (default: "Energy [eV]")
                - emin: Minimum energy to plot (default: 0.1 eV)
                - emax: Maximum energy to plot (default: 2e7 eV)
                - scalex: X-axis scale ("log" or "linear", default: "log")
                - scaley: Y-axis scale ("log" or "linear", default: "log")
                - Additional arguments passed to plotly

        Returns:
            plotly.graph_objects.Figure: Interactive figure object

        Note:
            This method uses plotly as the backend for interactive visualization,
            allowing for features like zooming, panning, and hover tooltips.
        """
        pd.options.plotting.backend = "plotly"

        title = kwargs.pop("title", self.name)
        ylabel = kwargs.pop("ylabel", "σ [barn]")
        xlabel = kwargs.pop("xlabel", "Energy [eV]")
        emin = kwargs.pop("emin", 0.1)
        emax = kwargs.pop("emax", 2e7)
        scalex = kwargs.pop("scalex", "log")
        scaley = kwargs.pop("scaley", "log")

        # Filter data to specified energy range
        filtered_table = self.table.query("@emin <= energy <= @emax")

        # Apply weights and format column labels
        table = filtered_table.mul(np.r_[self.weights, 1.0], axis=1)
        table.columns = [
            f"{column}: {weight*100:>6.2f}%" for column, weight in self.weights.items()
        ] + ["total"]

        fig = table.plot(**kwargs)

        # Configure layout
        fig.update_layout(
            xaxis_type=scalex,
            yaxis_type=scaley,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            title_text=title,
        )

        return fig
