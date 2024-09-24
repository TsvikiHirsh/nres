import os
import pandas as pd
import numpy as np
import nres.utils as utils
from typing import Dict, Union, List
from copy import copy
import nres


class CrossSection:
    """
    Represents a combination of cross-sections for different isotopes.

    This class loads isotope cross-section data, performs interpolation, rebinning,
    and allows calculation of total cross-section based on weighted sums.

    Attributes:
        isotopes (Dict[Union[str, 'CrossSection'], float]): Isotopes and their weights.
        name (str): Name of the combined cross-section.
        weights (pd.Series): Normalized weights for the isotopes.
        table (pd.DataFrame): Interpolated cross-section data for all isotopes and their weighted sum.
        L (float): Flight path length [m].
        tstep (float): Time step for the simulation.
        tbins (int): Number of time bins.
        first_tbin (int): Index of the first time bin.
        n (float): Calculated n value based on isotope weights.
        total_weight (float): Total weight of the cross-section.
    """

    def __init__(self, isotopes: Dict[Union[str, 'CrossSection'], float] = None, 
                 name: str = "", 
                 total_weight: float = 1.,
                 L: float = 10.59,
                 tstep: float = 1.56255e-9,
                 tbins: int = 640,
                 first_tbin: int = 1):
        """
        Initialize the CrossSection class.

        Args:
            isotopes: Dictionary of isotope names or CrossSection objects and their weights.
            name: Name of the combined cross-section.
            total_weight: Total weight of the cross-section.
            L: Flight path length [m].
            tstep: Time step for the simulation.
            tbins: Number of time bins.
            first_tbin: Index of the first time bin.
        """
        self.isotopes = isotopes or {}
        self.name = name
        self.total_weight = total_weight
        self.L = L
        self.tstep = tstep
        self.tbins = tbins
        self.first_tbin = first_tbin

        self.__xsdata__ = None
        self._load_xsdata()
        self._populate_isotope_data()
        self._set_weights()

    def _load_xsdata(self):
        """Load the cross-section data from file or download if not present."""
        if self.__xsdata__ is None:
            cache_path = utils.get_cache_path() / "xsdata.npy"
            if not os.path.exists(cache_path):
                print(f"File not found at {cache_path}, downloading...")
                utils.download_xsdata()

            xsdata = np.load(cache_path, allow_pickle=True)[()]
            self.__xsdata__ = {
                isotope: pd.Series(xsdata["cross_sections"][i], index=xsdata["energies"][i], name=isotope)
                for i, isotope in enumerate(xsdata["isotopes"])
            }

    def _populate_isotope_data(self):
        """Populate cross-section data for the isotopes and compute weighted total."""
        xs = {}
        updated_isotopes = {}
        self.n = 0.

        for isotope, weight in self.isotopes.items():
            if isinstance(isotope, str):
                if weight > 0:
                    xs[isotope] = self.__xsdata__[isotope]
                    updated_isotopes[isotope] = weight
                    self.n = np.NaN
            elif isinstance(isotope, CrossSection):
                xs[isotope.name] = isotope.table["total"].rename(isotope.name)
                updated_isotopes[isotope.name] = weight
                self.n += isotope.n * weight
                isotope = isotope.name

        self.isotopes = updated_isotopes
        self.table = pd.DataFrame(xs).interpolate()
        self.table.index.name = "energy"

    def _set_weights(self, weights: List[float] = None):
        """
        Set and normalize the weights for the isotopes.

        Args:
            weights: Optional list of new weights. If provided, must match the number of isotopes.
        """
        if weights is not None:
            if len(weights) != len(self.isotopes):
                raise ValueError("Number of weights must match number of isotopes")
            self.weights = pd.Series(weights, index=self.isotopes.keys())
        else:
            self.weights = pd.Series(self.isotopes)

        # Remove isotopes with zero weight
        self.weights = self.weights[self.weights > 0]

        # Normalize weights
        self.weights /= self.weights.sum()

        # Update the total cross-section
        self.table["total"] = (self.table.drop(columns="total", errors="ignore") * self.weights).sum(axis=1)


    def _set_energy_range(self, emin: float = 0.5e6, emax: float = 2.0e7):
        """Set the energy range for the cross-section data."""
        self.total = self.table["total"].loc[emin:emax].fillna(0.).values
        self.egrid = self.table["total"].loc[emin:emax].fillna(0.).index.values

    def __add__(self, other: 'CrossSection') -> 'CrossSection':
        """
        Add two CrossSection objects.

        Args:
            other: Another CrossSection object to add to the current one.

        Returns:
            A new CrossSection object representing the sum of the two.
        """
        all_energies = self.table.index.union(other.table.index)
        self_interpolated = self.table.reindex(all_energies).interpolate(method='index').drop(columns='total')
        other_interpolated = other.table.reindex(all_energies).interpolate(method='index').drop(columns='total')

        combined_weights = (self.weights * self.total_weight).add(
            other.weights * other.total_weight, fill_value=0
        )
        combined_weights /= combined_weights.sum()

        interpolated = pd.concat([
            self_interpolated,
            other_interpolated
        ], keys=['self', 'other'], axis=1)

        new_self = copy(self)
        new_self.table = (
            interpolated.mul(pd.concat([self.weights, other.weights], keys=['self', 'other']))
            .stack(0).groupby(level=0).sum()
        )

        new_self.weights = combined_weights
        new_self.total_weight = 1.
        new_self.table["total"] = (new_self.table * new_self.weights).sum(axis=1)
        new_self.n = self.total_weight * self.n + other.total_weight * other.n

        return new_self

    def __mul__(self, total_weight: float = 1.) -> 'CrossSection':
        """
        Multiply the CrossSection by a total weight.

        Args:
            total_weight: The weight to multiply by.

        Returns:
            A new CrossSection object with updated total_weight.
        """
        new_self = copy(self)
        new_self.total_weight = total_weight
        return new_self

    def __call__(self, E: np.ndarray, weights: np.ndarray = None, response: np.ndarray = None) -> np.ndarray:
        """
        Calculate the weighted cross-section for a given set of energies.

        Args:
            E: Array of energy values.
            weights: Optional array of new weights.
            response: Optional response function (default: [1.]).

        Returns:
            Array of weighted cross-section values.
        """
        from nres._integrate_xs import integrate_cross_section

        if weights is not None:
            self._set_weights(weights=weights)
        if response is None:
            response = [1.]
        return np.array(integrate_cross_section(self.table["total"].index.values, self.table["total"].values, E, response, self.L))

    def plot(self, **kwargs):
        """
        Plot the cross-section data.

        Args:
            **kwargs: Optional plotting parameters.
        """
        import matplotlib.pyplot as plt

        title = kwargs.get("title", self.name)
        ylabel = kwargs.get("ylabel", "$\sigma$ [barn]")
        xlabel = kwargs.get("xlabel", "Energy [eV]")
        lw = kwargs.get("lw", 1.)

        table = self.table.mul(np.r_[self.weights, 1.], axis=1)
        table.columns = [f"{column}: {weight*100:>6.2f}%" for column, weight in self.weights.items()] + ["total"]
        
        fig, ax = plt.subplots()
        table.plot(y="total", lw=1.5, ax=ax, color="0.2", zorder=100,**kwargs)
        table.drop("total", axis=1).plot(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, lw=lw, **kwargs)
        # return ax
    
    @classmethod
    def from_material(cls, mat: Union[str, Dict], short_name: str = "", 
                      total_weight: float = 1., splitby:str ="elements") -> 'CrossSection':
        """
        Create a CrossSection instance from a material.

        Args:
            mat: Material or Element name/formula or dictionary containing material information.
            short_name: Short name for the material (optional).
            total_weight: Total weight of the material (default is 1.0).
            splitby: split the cross section for "elements", "isotopes" or "materials" contributions

        Returns:
            CrossSection instance representing the material.
        """
        if isinstance(mat, str):
            formulas = {nres.materials[element]["formula"]: nres.materials[element]["name"] for element in nres.materials}
            try: # try finding the material name in the materials database
                mat = nres.materials[formulas.get(mat, mat)]
            except KeyError: # otherwise try the elements database
                formulas = {nres.elements[element]["formula"]: nres.elements[element]["name"] for element in nres.elements}
                mat = nres.elements[formulas.get(mat.capitalize(), mat.capitalize())]
        
        short_name = short_name or mat["name"]

        if splitby=="isotopes":
            for i, (element, data) in enumerate(mat["elements"].items()):
                if i==0:
                    xs = cls(mat["elements"][element]["isotopes"], name=element, total_weight=data["weight"])
                else:
                    xs += cls(mat["elements"][element]["isotopes"], name=element, total_weight=data["weight"])
            xs.name = short_name
        elif splitby=="elements":
                    
            xs_elements = {}
            for element, data in mat["elements"].items():
                xs = cls(data["isotopes"], name=element)
                xs_elements[xs] = data["weight"]
            xs = cls(xs_elements, name=short_name, total_weight=total_weight)

        elif splitby=="materials":

            xs_elements = {}
            for element, data in mat["elements"].items():
                xs = cls(data["isotopes"], name=element)
                xs_elements[xs] = data["weight"]
            xs = cls(xs_elements, name=short_name, total_weight=total_weight).group(name=short_name)

        else:
            raise ValueError("you can splitby 'isotopes','elements' or 'materials'")
            
        xs.n = mat["n"]
        return xs
    
    def group(self,name:str):
        # group the CrossSection objects under the same name
        new_self = copy(self)
        new_self.table[name] = new_self.table["total"]
        new_self.table = new_self.table[[name,"total"]]
        new_self.weights = pd.Series([1.],index=[name])
        return new_self
    
    def _is_isotope(self,isotope:str):
        # checks if a key is an isotope name:
        return True if isotope in self.__xsdata__ else False
    
    def groupby_isotopes(self):
        # groupby isotopes
        new_weights = {}
        new_table = {}
        new_self = copy(self)
        for isotope,weight in new_self.weights.items():
            if self._is_isotope(isotope):
                element,mass = isotope.split("-")
                if element in new_weights:
                    new_weights[element] += weight
                    new_table[element]+=weight*new_self.table[isotope]
                else:
                    new_weights[element] = weight
                    new_table[element]=weight*new_self.table[isotope]
            else:
                new_weights[isotope] = weight
                new_table[isotope]=weight*new_self.table[isotope]

        new_self.table = pd.DataFrame(new_table)
        
        new_self.weights = pd.Series(new_weights)
        new_self.table["total"] = (new_self.table*new_self.weights).sum(1)
        return new_self

        
