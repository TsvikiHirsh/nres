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

def from_material(mat, short_name: str = ""):
    xs_elements = {}
    for element in mat["elements"]:
        xs = CrossSection(mat["elements"][element]["isotopes"],name=element)
        xs_elements[xs] = mat["elements"][element]["weight"]
    short_name = short_name if short_name else mat["name"]
    xs = CrossSection(xs_elements,name=short_name)
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
        self.weights = np.array(list(self.isotopes.values()), float)
        self.weights = self.weights[self.weights>0] # remove weight=0 isotopes
        self.weights /= self.weights.sum()  # Normalize weights
        self.L = L
        self.name = name
        self.ufuncs = []

        # Define the filename and location in site-packages
        self.file_name = 'xsdata.npy'
        self.package_dir = os.path.join(site.getsitepackages()[0], 'cross_section_data')
        self.file_path = os.path.join(self.package_dir, self.file_name)

        self.__xsdata__ = None
        self._load_xsdata()

        self.first_tbin = first_tbin
        self.tstep = tstep
        self.tbins = tbins



        # Populate cross-section data for isotopes
        self._populate_isotope_data()


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
        table["total"] = (table * self.weights).sum(axis=1)
        table.index.name = "energy"
        self.table = table

    def set_energy_range(self,emin=0.5e6,emax=2.0e7):
        self.total = self.table["total"].loc[emin:emax].fillna(0.).values
        self.egrid = self.table["total"].loc[emin:emax].fillna(0.).index.values

    def set_weights(self,weights):
        self.total = self.table.drop(["total"],axis=1).mul(weights, axis=1).sum(axis=1).fillna(0.)

    def set_response(self, kind="gauss_norm", **kwargs):
        """
        Set the response function for the cross-section and apply it to the total cross-section.

        Parameters:
        ----------
        kind : str
            Type of response function to use.
        kwargs : dict
            Additional parameters for the response function.
        """
        self.response = Response(kind=kind, L=self.L,tbin=self.tstep,nbins=self.nbins)
        tof = utils.energy2time(self.table.index.values, self.L)
        self.table["response_total"] = convolve(self.table["total"], self.response(tof, **kwargs), "same")

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
            self.set_weights(weights=weights)
        response = response if len(response) else [0.]
        return np.array(integrate_cross_section(self.total.index.values, self.total.values, E, response, self.L))

    def plot(self,**kwargs):
        """Plot the cross-section data with optional plotting parameters."""
        title = kwargs.get("title","Cross section")
        ylabel = kwargs.get("ylabel","$\sigma$ [barn]")
        xlabel = kwargs.get("xlabel","Energy [eV]")
        self.table.mul(np.r_[self.weights, 1], axis=1).plot(title=title,
                                                            xlabel=xlabel,
                                                            ylabel=ylabel,
                                                            **kwargs)

