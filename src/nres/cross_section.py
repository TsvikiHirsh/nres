import os
import pandas as pd
import numpy as np
import nres.utils as utils
from typing import Dict, Union, List, Optional
from copy import deepcopy
import nres
from nres._integrate_xs import integrate_cross_section

class CrossSection:
    """
    A class representing neutron cross-sections for single or multiple isotopes.

    This class handles the loading, manipulation, and analysis of neutron cross-section data.
    It supports operations on individual isotopes as well as combinations of isotopes with
    different weights. The class provides functionality for interpolation, rebinning,
    and calculation of total cross-sections based on weighted sums.

    Key Features:
    - Load and manage cross-section data for multiple isotopes
    - Perform weighted combinations of cross-sections
    - Interpolate and rebin cross-section data
    - Calculate total cross-sections
    - Support for mathematical operations (addition, multiplication)
    - Plotting capabilities

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
        total_weight (float): Total weight factor for the cross-section.
    """

    def __init__(self, isotopes: Dict[Union[str, 'CrossSection'], float] | 'CrossSection' = None, 
                 name: str = "", 
                 total_weight: float = 1.,
                 L: float = 10.59,
                 tstep: float = 1.56255e-9,
                 tbins: int = 640,
                 first_tbin: int = 1):
        """
        Initialize a new CrossSection instance.

        The initialization can occur in two ways:
        1. From scratch using a dictionary of isotopes and their weights
        2. By copying an existing CrossSection object

        Args:
            isotopes: Either a dictionary mapping isotope names/CrossSection objects to their
                weights, or an existing CrossSection object to copy from.
            name: Identifier for this cross-section combination. If copying from an existing
                CrossSection and no name is provided, uses the original name.
            total_weight: Overall scaling factor for the cross-section (default: 1.0).
            L: Flight path length in meters (default: 10.59).
            tstep: Time step for the simulation in seconds (default: 1.56255e-9).
            tbins: Number of time bins for the simulation (default: 640).
            first_tbin: Index of the first time bin (default: 1).
        """
        if isinstance(isotopes, CrossSection):
            self._init_from_cross_section(isotopes, name)
        else:
            self._init_new(isotopes, name, total_weight, L, tstep, tbins, first_tbin)

    def _init_from_cross_section(self, other: 'CrossSection', name: str = ""):
        """
        Initialize this CrossSection by copying from an existing instance.

        Creates a new CrossSection with the same properties as the source object
        but maintains independence through proper copying of mutable attributes.

        Args:
            other: Source CrossSection object to copy from.
            name: New name for this cross-section. If empty, uses the original name.
        """
        self.isotopes = other.isotopes.copy()
        self.name = name if name else f"{other.name}"
        self.total_weight = other.total_weight
        self.L = other.L
        self.tstep = other.tstep
        self.tbins = other.tbins
        self.first_tbin = other.first_tbin
        self.tgrid = other.tgrid.copy()
        self.n = other.n
        self.__xsdata__ = other.__xsdata__
        
        # Deep copy mutable attributes to ensure independence
        self.table = other.table.copy()
        self.weights = other.weights.copy()

    def _init_new(self, isotopes: Dict[Union[str, 'CrossSection'], float] = None,
                  name: str = "", 
                  total_weight: float = 1.,
                  L: float = 10.59,
                  tstep: float = 1.56255e-9,
                  tbins: int = 640,
                  first_tbin: int = 1):
        """
        Initialize a new CrossSection from scratch with given parameters.

        This method handles the core initialization logic when creating a new
        CrossSection instance from isotope data rather than copying.

        Args:
            isotopes: Dictionary mapping isotope names/CrossSection objects to weights.
            name: Identifier for this cross-section combination.
            total_weight: Overall scaling factor for the cross-section.
            L: Flight path length in meters.
            tstep: Time step for the simulation in seconds.
            tbins: Number of time bins.
            first_tbin: Index of the first time bin.
        """
        self.isotopes = isotopes or {}
        self.name = name
        self.total_weight = total_weight if self.isotopes else 0.
        self.L = L
        self.tstep = tstep
        self.tbins = tbins
        self.first_tbin = first_tbin
        self.tgrid = np.arange(self.first_tbin, self.tbins+1, 1) * self.tstep

        self.__xsdata__ = None
        self._load_xsdata()
        self._populate_isotope_data()
        self._set_weights()

    def _load_xsdata(self):
        """
        Load cross-section data from file or download if not present.

        Checks for cached cross-section data in the local cache directory.
        If not found, downloads the data from the remote source. The data
        is stored in a numpy file containing energies and cross-sections
        for various isotopes.

        The loaded data is stored in self.__xsdata__ as a dictionary mapping
        isotope names to their cross-section data as pandas Series.
        """
        if self.__xsdata__ is None:
            cache_path = utils.get_cache_path() / "xsdata.npy"
            if not os.path.exists(cache_path):
                print(f"File not found at {cache_path}, downloading...")
                utils.download_xsdata()

            xsdata = np.load(cache_path, allow_pickle=True)[()]
            self.__xsdata__ = {
                isotope.replace("-",""): pd.Series(
                    xsdata["cross_sections"][i], 
                    index=xsdata["energies"][i], 
                    name=isotope.replace("-","")
                )
                for i, isotope in enumerate(xsdata["isotopes"])
            }

    def _populate_isotope_data(self):
        """
        Process and combine cross-section data for all isotopes.

        This method handles both string-based isotope names (loading from __xsdata__)
        and CrossSection objects. For each isotope:
        - If it's a string, loads data from __xsdata__
        - If it's a CrossSection object, uses its total cross-section
        
        The method also:
        - Updates isotope weights
        - Calculates the combined number density (n)
        - Creates an interpolated table of all cross-sections
        """
        xs = {}
        updated_isotopes = {}
        self.n = 0.

        for isotope, weight in self.isotopes.items():
            if isinstance(isotope, str):
                # Handle string-based isotope names
                isotope = isotope.replace("-","")
                if weight > 0:
                    xs[isotope] = self.__xsdata__[isotope]
                    updated_isotopes[isotope] = weight
                    self.n = np.nan
            elif isinstance(isotope, CrossSection):
                # Handle CrossSection objects
                xs[isotope.name] = isotope.table["total"].rename(isotope.name)
                updated_isotopes[isotope.name] = weight
                self.n += isotope.n * weight
                isotope = isotope.name

        self.isotopes = updated_isotopes
        self.table = utils.interpolate(pd.DataFrame(xs))
        self.table.index.name = "energy"

    def _set_weights(self, weights: Optional[List[float]] = None):
        """
        Set and normalize weights for all isotopes.

        This method handles the normalization of isotope weights and updates
        the total cross-section accordingly. It ensures that:
        - Weights sum to 1.0
        - Isotopes with zero weight are removed
        - The total cross-section is recalculated using the new weights

        Args:
            weights: Optional list of new weights. If provided, must match
                the number of isotopes. If None, uses weights from self.isotopes.

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
            self.table.drop(columns="total", errors="ignore") * self.weights
        ).sum(axis=1).astype(float)
        
    def _set_energy_range(self, emin: float = 0.5e6, emax: float = 2.0e7):
        """
        Set the energy range for cross-section calculations.

        This method filters the cross-section data to a specific energy range
        and updates the corresponding time grid for time-of-flight calculations.

        Args:
            emin: Minimum energy in eV (default: 0.5e6)
            emax: Maximum energy in eV (default: 2.0e7)

        Note:
            The time grid is calculated using the flight path length (L)
            and the energy-to-time conversion from utils.time2energy.
        """
        self.total = self.table["total"].loc[emin:emax].fillna(0.).values
        self.egrid = self.table["total"].loc[emin:emax].fillna(0.).index.values
        self.tgrid = utils.time2energy(self.egrid, self.L)

    def __add__(self, other: 'CrossSection') -> 'CrossSection':
        """
        Add two CrossSection objects together.

        Performs a weighted combination of two CrossSection objects, handling:
        - Interpolation of cross-sections to a common energy grid
        - Combination of weights based on total_weight values
        - Preservation of number density calculations
        
        The resulting CrossSection maintains consistency in all attributes
        and represents the physical combination of the two cross-sections.

        Args:
            other: Another CrossSection object to add to this one

        Returns:
            CrossSection: A new CrossSection object representing the weighted sum
                of the two input cross-sections

        Note:
            The resulting weights are normalized such that they sum to 1.0,
            and the total_weight of the result is set to 1.0.
        """
        all_energies = self.table.index.union(other.table.index)
        self_interpolated = utils.interpolate(self.table.reindex(all_energies)).drop(columns='total')
        other_interpolated = utils.interpolate(other.table.reindex(all_energies)).drop(columns='total')

        # Combine weights considering total_weight of each CrossSection
        combined_weights = (self.weights * self.total_weight).add(
            other.weights * other.total_weight, fill_value=0
        )
        combined_weights /= combined_weights.sum()

        # Stack and combine the interpolated cross-sections
        interpolated = pd.concat([
            self_interpolated,
            other_interpolated
        ], keys=['self', 'other'], axis=1)

        new_self = deepcopy(self)
        try:
            # Try new pandas style first
            new_self.table = (
                interpolated.mul(pd.concat([self.weights, other.weights], keys=['self', 'other']))
                .stack(0, future_stack=True).groupby(level=0).sum()
            )
        except TypeError:  # Older pandas versions won't recognize future_stack parameter
            new_self.table = (
                interpolated.mul(pd.concat([self.weights, other.weights], keys=['self', 'other']))
                .stack(0).groupby(level=0).sum()
            )

        # Update attributes for the combined CrossSection
        new_self.weights = combined_weights
        new_self.total_weight = 1.
        new_self.table["total"] = (new_self.table * new_self.weights).sum(axis=1).astype(float)
        new_self.n = self.total_weight * self.n + other.total_weight * other.n
        new_self.isotopes = new_self.weights.to_dict()

        return new_self

    def __mul__(self, total_weight: float = 1.) -> 'CrossSection':
        """
        Scale the CrossSection by a total weight factor.

        Creates a new CrossSection object with the same cross-section data
        but scaled by the specified total_weight. This is useful for
        combining cross-sections with different relative abundances.

        Args:
            total_weight: Scaling factor to apply (default: 1.0)

        Returns:
            CrossSection: A new CrossSection object with updated total_weight

        Note:
            This operation does not modify the internal weights between isotopes,
            only the overall scaling factor total_weight.
        """
        new_self = deepcopy(self)
        new_self.total_weight = total_weight
        return new_self

    def __call__(self, E: np.ndarray, weights: Optional[np.ndarray] = None, 
                 response: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the weighted cross-section for given energy values.

        This method enables the CrossSection object to be called as a function,
        returning interpolated cross-section values for the specified energies.
        It can optionally apply new weights or a response function.

        Args:
            E: Array of energy values in eV at which to evaluate the cross-section
            weights: Optional array of weights to temporarily override the current weights
            response: Optional response function to convolve with the cross-section
                (default: [1.0], i.e., no modification)

        Returns:
            np.ndarray: Array of cross-section values at the specified energies

        Note:
            The cross-section values are calculated using the integrate_cross_section
            function, which handles the interpolation and optional convolution
            with the response function.
        """
        if weights is not None:
            self._set_weights(weights=weights)
        if response is None:
            response = [1.]
        return np.array(integrate_cross_section(
            self.table["total"].index.values, 
            self.table["total"].values, 
            E, 
            response
        ))

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
        ylabel = kwargs.pop("ylabel", "$\sigma$ [barn]")
        xlabel = kwargs.pop("xlabel", "Energy [eV]")
        lw = kwargs.pop("lw", 1.)

        # Apply weights and format column labels with percentage contributions
        table = self.table.mul(np.r_[self.weights, 1.], axis=1)
        table.columns = [f"{column}: {weight*100:>6.2f}%" 
                        for column, weight in self.weights.items()] + ["total"]
        
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
        table = filtered_table.mul(np.r_[self.weights, 1.], axis=1)
        table.columns = [f"{column}: {weight*100:>6.2f}%" 
                        for column, weight in self.weights.items()] + ["total"]
        
        fig = table.plot(**kwargs)
        
        # Configure layout
        fig.update_layout(
            xaxis_type=scalex,
            yaxis_type=scaley,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            title_text=title
        )

        return fig

    @classmethod
    def from_material(cls, mat: Union[str, Dict], short_name: str = "", 
                     total_weight: float = 1., splitby: str = "elements") -> 'CrossSection':
        """
        Create a CrossSection instance from a material specification.

        This class method provides a convenient way to create CrossSection
        objects from material definitions, supporting various ways to split
        the contributions from different components.

        Args:
            mat: Material specification, either as:
                - String: Material/element name or chemical formula
                - Dict: Material information dictionary
            short_name: Brief identifier for the material (default: material name)
            total_weight: Overall scaling factor (default: 1.0)
            splitby: How to split the cross-section contributions:
                - "elements": Split by chemical elements
                - "isotopes": Split by individual isotopes
                - "materials": Combine all into a single material

        Returns:
            CrossSection: A new CrossSection object representing the material

        Raises:
            ValueError: If splitby is not one of the allowed values
            KeyError: If the material name/formula is not found in the database

        Note:
            The method attempts to find the material in three databases:
            1. Materials database
            2. Elements database
            3. Isotopes database
        """
        if isinstance(mat, str):
            formulas = {nres.materials[element]["formula"]: nres.materials[element]["name"] 
                       for element in nres.materials}
            try: # Try materials database
                mat = nres.materials[formulas.get(mat, mat)]
            except KeyError: # Try elements database
                try:
                    formulas = {nres.elements[element]["formula"]: nres.elements[element]["name"] 
                              for element in nres.elements}
                    mat = nres.elements[formulas.get(mat.capitalize(), mat.capitalize())]
                except KeyError: # Try isotopes database
                    mat = nres.isotopes[mat.capitalize().replace("-","")]

        short_name = short_name or mat["name"]

        if splitby == "isotopes":
            # Create separate cross-sections for each isotope
            xs = sum((cls(mat["elements"][element]["isotopes"], name=element, 
                        total_weight=data["weight"])
                     for element, data in mat["elements"].items()), start=cls())
            xs.isotopes = xs.weights.to_dict()
            xs.name = short_name
        elif splitby == "elements":
            # Group isotopes by element
            xs_elements = {cls(data["isotopes"], name=element): data["weight"]
                         for element, data in mat["elements"].items()}
            xs = cls(xs_elements, name=short_name, total_weight=total_weight)
            xs.isotopes = xs.weights.to_dict()
        elif splitby == "materials":
            # Combine all into a single material
            xs_elements = {cls(data["isotopes"], name=element): data["weight"]
                         for element, data in mat["elements"].items()}
            xs = cls(xs_elements, name=short_name, 
                    total_weight=total_weight).group(name=short_name)
            xs.isotopes = xs.weights.to_dict()
        else:
            raise ValueError("splitby must be 'isotopes', 'elements', or 'materials'")
            
        xs.n = mat["n"]
        return xs
    
    def group(self, name: str) -> 'CrossSection':
        """
        Group all components under a single name.

        Combines all cross-section components into a single column while
        preserving the total cross-section. This is useful for simplifying
        the representation of complex materials.

        Args:
            name: Name to use for the grouped cross-section

        Returns:
            CrossSection: A new CrossSection object with all components
                grouped under the specified name

        Note:
            The resulting CrossSection will have only two columns:
            the specified name and 'total', both containing the same values.
        """
        new_self = deepcopy(self)
        new_self.table[name] = new_self.table["total"]
        new_self.table = new_self.table[[name, "total"]]
        new_self.weights = pd.Series([1.], index=[name])
        return new_self
    
    def _is_isotope(self, isotope: str) -> bool:
        """
        Check if a given key represents an isotope name.

        Args:
            isotope: String to check against the isotope database

        Returns:
            bool: True if the key exists in the cross-section data
        """
        return isotope in self.__xsdata__
    
    def groupby_isotopes(self) -> 'CrossSection':
        """
        Group cross-sections by chemical element, combining isotopes.

        Creates a new CrossSection where isotopes of the same element
        are combined into a single component. Non-isotope components
        are preserved as-is.

        Returns:
            CrossSection: A new CrossSection object with isotopes grouped
                by element

        Note:
            The weights of isotopes belonging to the same element are
            combined, and their cross-sections are weighted accordingly
            in the combination.
        """
        new_weights = {}
        new_table = {}
        new_self = deepcopy(self)

        for isotope, weight in new_self.weights.items():
            if self._is_isotope(isotope):
                # Group isotopes by element
                element, mass = isotope.split("-")
                if element in new_weights:
                    new_weights[element] += weight
                    new_table[element] += weight * new_self.table[isotope]
                else:
                    new_weights[element] = weight
                    new_table[element] = weight * new_self.table[isotope]
            else:
                # Preserve non-isotope components
                new_weights[isotope] = weight
                new_table[isotope] = weight * new_self.table[isotope]

        new_self.table = pd.DataFrame(new_table)
        new_self.weights = pd.Series(new_weights)
        new_self.table["total"] = (new_self.table * new_self.weights).sum(1).astype(float)
        return new_self