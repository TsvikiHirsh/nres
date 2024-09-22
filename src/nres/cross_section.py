import os
import pandas as pd
import numpy as np
from nres.response import Response
import nres.utils as utils
from scipy.signal import convolve
import site
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import warnings
from nres._integrate_xs import integrate_cross_section
import nres
from copy import copy

def from_material(mat, short_name: str = "",total_weight=1.):
    if isinstance(mat,str):
        formulas = {nres.materials[element]["formula"]:nres.materials[element]["name"] for element in nres.materials}
        if mat in formulas:
            mat = nres.materials[formulas[mat]]
        else:
            mat = nres.materials[mat]

    xs_elements = {}
    for element in mat["elements"]:
        xs = CrossSection(mat["elements"][element]["isotopes"],name=element)
        xs_elements[xs] = mat["elements"][element]["weight"]
    short_name = short_name if short_name else mat["name"]
    xs = CrossSection(xs_elements,name=short_name)
    xs.n = mat["n"]
    return xs

def from_element(mat,short_name: str = "",total_weight=1.):

    if isinstance(mat,str):
        formulas = {nres.elements[element]["formula"]:nres.elements[element]["name"] for element in nres.elements}
        if mat in formulas:
            mat = nres.elements[formulas[mat]]
        else:
            mat = nres.elements[mat]
    element = list(mat["elements"].keys())[0]
    short_name = short_name if short_name else mat["name"]
    xs = CrossSection(mat["elements"][element]["isotopes"],name=short_name,total_weight=total_weight)
    xs.n = mat["n"]
    return xs

class CrossSection:
    """
    Class representing a combination of cross-sections for different isotopes.

    This class loads isotope cross-section data, performs interpolation, rebinning, and
    allows calculation of total cross-section based on weighted sums. Cross-section data 
    is either loaded from local files or downloaded from a remote GitHub repository.

    Attributes:
    ----------
    isotopes : dict
        A dictionary with isotope names or `CrossSection` objects as keys and their respective weights as values.
    name : str
        The name of the combined cross-section.
    weights : np.ndarray
        Normalized weights for the isotopes.
    table : pd.DataFrame
        Interpolated cross-section data for all isotopes and their weighted sum.
    ufuncs : list
        List of interpolation functions for each isotope.
    L : float
        Flight path length [m].
    tstep : float
        Time step for the simulation.
    tbins : int
        Number of time bins.
    first_tbin : int
        The index of the first time bin.
    """

    def __init__(self, isotopes: dict = {}, 
                 name: str = "", 
                 total_weight: float = 1.,
                 L: float = 10.59,
                 tstep: float = 1.56255e-9,
                 tbins: int = 640,
                 first_tbin: int = 1):
        """ 
        Initialize the CrossSection class with isotopes, weights, and other parameters.

        Parameters:
        ----------
        isotopes : dict
            Dictionary with isotope names or `CrossSection` objects as keys and their respective weights as values.
        name : str
            Name of the combined cross-section.
        L : float
            Flight path length [m].
        tstep : float
            Time step for the simulation.
        tbins : int
            Number of time bins.
        first_tbin : int
            The index of the first time bin.
        """
        self.isotopes = isotopes
        
        # names = []
        # for isotope in self.isotopes:
        #     if isinstance(isotope, CrossSection):
        #         names.append(isotope.name)
        #     else:
        #         names.append(isotope)

        self.L = L
        self.name = name
        self.ufuncs = []
        self.total_weight = total_weight

        self.__xsdata__ = None
        self._load_xsdata()

        self.first_tbin = first_tbin
        self.tstep = tstep
        self.tbins = tbins

        # Populate cross-section data for isotopes
        self._populate_isotope_data()
        self._set_weights()


    def _load_xsdata(self):
        """Load the xsdata.npy file into the object."""
        if self.__xsdata__ is None:
            cache_path = utils.get_cache_path() / "xsdata.npy"
            if not os.path.exists(cache_path):
                print(f"File not found at {cache_path}, downloading...")
                utils.download_xsdata()

            xsdata = np.load(cache_path, allow_pickle=True)[()]
            self.__xsdata__ = {
                isotope: pd.Series(xsdata["cross_sections"][i], index=xsdata["energies"][i],name=isotope)
                for i, isotope in enumerate(xsdata["isotopes"])
            }

    def _populate_isotope_data(self):
        """Populate cross-section data for the isotopes and compute weighted total."""
        xs = {}
        updated_isotopes = {}
        self.n = 0.
        for isotope, weight in self.isotopes.items():
            if isinstance(isotope, str):
                if weight>0:
                    xs[isotope] = self.__xsdata__[isotope]
                    updated_isotopes[isotope] = weight
                    self.n = np.NaN
            elif isinstance(isotope, CrossSection):
                xs[isotope.name] = isotope.table["total"].rename(isotope.name)
                updated_isotopes[isotope.name] = weight
                self.n+=isotope.n*weight
                isotope = isotope.name

        self.isotopes = updated_isotopes
        table = pd.DataFrame(xs).interpolate()
        table.index.name = "energy"
        self.table = table

    def _set_energy_range(self,emin=0.5e6,emax=2.0e7):
        self.total = self.table["total"].loc[emin:emax].fillna(0.).values
        self.egrid = self.table["total"].loc[emin:emax].fillna(0.).index.values

    def _set_weights(self,weights=[]):
        self.weights = pd.Series(self.isotopes)
        if len(weights):
            self.weights.values = weights
        self.weights = self.weights[self.weights>0] # remove weight=0 isotopes
        self.weights /= self.weights.sum()  # Normalize weights
        self.table["total"] =  (self.table * self.weights).sum(axis=1)


    def __add__(self,other):
        """Add a cross section object

        Args:
            other (CrossSection object): Another CrossSection Object to add to the current one
        """
        self.weights = self.weights.mul(self.total_weight).add(other.weights.mul(other.total_weight), fill_value=0.)
        self.weights /= self.weights.sum()
        self.isotopes = self.weights.to_dict()
        total_weight = self.total_weight + other.total_weight
        self.total_weight/=total_weight
        other.total_weight/=total_weight

        # Combine energy grids
        all_energies = self.table.index.union(other.table.index)

        # Interpolate both tables to the combined energy grid
        self_interpolated = self.table.reindex(all_energies).interpolate(method='index').drop(columns='total')
        other_interpolated = other.table.reindex(all_energies).interpolate(method='index').drop(columns='total')

        # Add the weighted tables
        combined_weights = (self.weights * self.total_weight).add(
            other.weights * other.total_weight, fill_value=0
        )
        combined_weights /= combined_weights.sum()

        interpolated = pd.concat([
            self_interpolated,
            other_interpolated
        ], keys=['self', 'other'],axis=1)

        new_self = copy(self)
        # Calculate new cross-sections
        new_self.table = (
            interpolated.mul(pd.concat([self.weights, other.weights], keys=['self', 'other']))
            .stack(0).groupby(level=0).sum()
        )

        new_self.weights = combined_weights

        new_self.table["total"] = (new_self.table * new_self.weights).sum(axis=1)
        new_self.n = self.total_weight*self.n + other.total_weight*other.n

        return new_self
    
    def __mul__(self,total_weight=1.):
        new_self = copy(self)
        new_self.total_weight = total_weight
        return new_self


    def __call__(self, E, weights = None,response=[0.,1.,0.]):
        """
        Calculate the weighted cross-section for a given set of energies.

        Parameters:
        ----------
        weights : np.ndarray, optional
            Optional array of new weights.

        Returns:
        -------
        np.ndarray
            Array of weighted cross-section values.
        """
        if weights==None or weights==[]:
            pass
        else:
            self._set_weights(weights=weights)
        response = response if len(response) else [0.]
        return np.array(integrate_cross_section(self.total.index.values, self.total.values, E, response, self.L))

    def plot(self,**kwargs):
        """Plot the cross-section data with optional plotting parameters."""
        title = kwargs.get("title",self.name)
        ylabel = kwargs.get("ylabel","$\sigma$ [barn]")
        xlabel = kwargs.get("xlabel","Energy [eV]")
        lw = kwargs.get("lw",1.)
        # rename columns

        table = self.table.mul(np.r_[self.weights,1.], axis=1)
        table.columns = [f"{column}: {weight*100:>6.2f}%" for column,weight in self.weights.items()] + ["total"]
        ax = table.drop("total",axis=1).plot(title=title,
                                                xlabel=xlabel,
                                                ylabel=ylabel,
                                                lw = lw,
                                                **kwargs)
        table.plot(y="total",lw=1.5,ax=ax,color="0.2")

