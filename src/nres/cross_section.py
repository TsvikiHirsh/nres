import os
import requests
import pandas as pd
import numpy as np
from nres.response import Response
import nres.utils as utils
from scipy.signal import convolve
from tqdm import tqdm
import site
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import warnings
from nres._integrate_xs import integrate_cross_section


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
        self.set_weights(self.weights)
        # self.set_energy_range()

    def _download_xsdata(self):
        """Download the xsdata.npy file from GitHub and save it to the package directory."""
        url = 'https://github.com/lanl/trinidi-data/blob/main/xsdata.npy?raw=true'

        # Create the folder if it doesn't exist
        if not os.path.exists(self.package_dir):
            os.makedirs(self.package_dir)

        # Download with progress bar
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(self.file_path, 'wb') as f, tqdm(
                desc=f"Downloading {self.file_path}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ncols=80
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
            print(f"File downloaded and saved to {self.file_path}")
        else:
            raise Exception(f"Failed to download the file. Status code: {response.status_code}")

    def _load_xsdata(self):
        """Load the xsdata.npy file into the object."""
        if self.__xsdata__ is None:
            if not os.path.exists(self.file_path):
                print(f"File not found at {self.file_path}, downloading...")
                self._download_xsdata()

            xsdata = np.load(self.file_path, allow_pickle=True)[()]
            self.__xsdata__ = {
                isotope: pd.Series(xsdata["cross_sections"][i], index=xsdata["energies"][i],name=isotope)
                for i, isotope in enumerate(xsdata["isotopes"])
            }

    def _populate_isotope_data(self):
        """Populate cross-section data for the isotopes and compute weighted total."""
        xs = {}
        updated_isotopes = {}
        
        for isotope, weight in self.isotopes.items():
            if isinstance(isotope, str):
                xs[isotope] = self.__xsdata__[isotope]
                updated_isotopes[isotope] = weight
            elif isinstance(isotope, CrossSection):
                xs[isotope.name] = isotope.table["total"].rename(isotope.name)
                updated_isotopes[isotope.name] = weight
                isotope = isotope.name
        
        self.isotopes = updated_isotopes
        table = pd.DataFrame(xs).interpolate()
        table["total"] = (table * self.weights).sum(axis=1)
        table.index.name = "energy"
        self.table = table

    def get_xs(self, isotope: str = "C-12") -> pd.Series:
        """
        Retrieve and interpolate the cross-section data for a given isotope.

        Parameters:
        ----------
        isotope : str
            The isotope for which to retrieve the cross-section.

        Returns:
        -------
        pd.Series
            Interpolated cross-section values for the isotope.
        """
        warnings.filterwarnings("ignore")

        # Get the cross-section data
        xs = self.__xsdata__[isotope]

        # Prepare the input data for the C++ function
        xs_energies = xs.index.values # Convert index to list
        xs_values = xs.values         # Convert values to list

        # Define the grid for integration
        grid = np.arange(self.first_tbin, self.tbins + 1 + self.first_tbin, 1)
        energy_grid = [utils.time2energy(g * self.tstep, self.L) for g in grid]

        # Call the C++ integration function
        results = integrate_cross_section(xs_energies, xs_values, energy_grid)

        # Convert the results back to a pandas Series
        integral = {g: result if result >= 0 else 0. for g, result in zip(grid[:-1], results)}
        integral = pd.Series(integral, name=isotope)
        self.tgrid = grid[:-1]
        self.egrid = np.array(energy_grid[:-1])

        return integral


    # def get_xs(self, isotope: str = "C-12") -> pd.Series:
    #     """
    #     Retrieve and interpolate the cross-section data for a given isotope.

    #     Parameters:
    #     ----------
    #     isotope : str
    #         The isotope for which to retrieve the cross-section.

    #     Returns:
    #     -------
    #     pd.Series
    #         Interpolated cross-section values for the isotope.
    #     """
    #     warnings.filterwarnings("ignore")

    #     # Get the cross-section data and create a linear interpolation function
    #     xs = self.__xsdata__[isotope]
    #     ufunc = UnivariateSpline(xs.index.values, xs.values, k=1, s=0)

    #     integral = {}
    #     grid = np.arange(self.first_tbin, self.tbins + 1 + self.first_tbin, 1)
    #     for i, g in enumerate(grid[:-1]):
    #         emin = utils.time2energy(grid[i + 1] * self.tstep, self.L)
    #         emax = utils.time2energy(grid[i] * self.tstep, self.L)
    #         if emin > 0 and emax > 0:
    #             result = quad(ufunc, emin, emax)[0] / (emax - emin)
    #             integral[g] = result if result >= 0 else 0.
    #         else:
    #             integral[g] = 0
    #     integral = pd.Series(integral, name=isotope)
    #     self.tgrid = grid[:-1]
    #     self.egrid = utils.time2energy(self.tgrid*self.tstep, self.L)

    #     return integral
    
    def set_energy_range(self,emin=0.5e6,emax=2.0e7):
        self.total = self.table.query(f"{emin}<=energy<={emax}")["total"].fillna(0.).values
        self.egrid = self.table.query(f"{emin}<=energy<={emax}")["energy"].values

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
        return integrate_cross_section(self.total.index.values, self.total.values, E, response, self.L)

    def plot(self,**kwargs):
        """Plot the cross-section data with optional plotting parameters."""
        self.table.mul(np.r_[self.weights, 1], axis=1).plot(**kwargs)

