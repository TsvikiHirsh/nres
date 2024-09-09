import os
import requests
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline


def format_isotope_name(isotope: str) -> str:
    """
    Format the isotope name to match the GitHub repository format.
    
    Parameters:
    isotope (str): The isotope name in formats like 'Al-27' or 'Al27'.
    
    Returns:
    str: Formatted isotope name, e.g., 'Al027'.
    """
    isotope = isotope.replace('-', '').replace(' ', '')
    element = ''.join([char for char in isotope if not char.isdigit()])
    atomic_number = ''.join([char for char in isotope if char.isdigit()])
    return f"{element}{atomic_number.zfill(3)}"


def grab_from_endf(isotope: str, evaluated_data_dir: str = "evaluated_data") -> pd.Series:
    """
    Download and save the cross-section data for the specified isotope if it doesn't exist locally.
    
    Parameters:
    isotope (str): The name of the isotope, e.g., 'Al-27'.
    evaluated_data_dir (str): Directory to save the downloaded data.
    
    Returns:
    pd.Series: The cross-section data with energy as the index and cross-section as the values.
    """
    formatted_isotope = format_isotope_name(isotope)
    url = (
        f"https://raw.githubusercontent.com/pedrojrv/ML_Nuclear_Data/master/"
        f"Evaluated_Data/neutrons/{formatted_isotope}/endfb8.0/tables/xs/"
        f"n-{formatted_isotope}-MT001.endfb8.0"
    )
    file_path = os.path.join(evaluated_data_dir, f"{formatted_isotope}.endf")

    # Download and save the file if it doesn't exist locally
    if not os.path.exists(file_path):
        print(f"Downloading data for {isotope}...")
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        os.makedirs(evaluated_data_dir, exist_ok=True)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Data saved to {file_path}")

    # Read and process the cross-section data
    xs = pd.read_csv(file_path, comment="#", delim_whitespace=True, header=None, names=["E", "xs"], usecols=[0, 1], index_col=0)
    xs = xs[~xs.index.duplicated()]
    xs["xs"] *= 0.001  # Convert to barns
    xs.index *= 1e6  # Convert to eV
    return xs["xs"].rename(isotope)


class CrossSection:
    """
    Class representing a combination of cross-sections for different isotopes.

    Attributes:
    isotopes (dict): A dictionary with isotope names as keys and their respective weights as values.
    name (str): The name of the combined cross-section.
    weights (np.ndarray): Normalized weights for the isotopes.
    table (pd.DataFrame): Interpolated cross-section data for all isotopes and their weighted sum.
    ufuncs (list): List of interpolation functions for each isotope.
    """

    def __init__(self, isotopes: dict = {}, name: str = "", vary_weights: bool = False):
        """
        Initialize the CrossSection class.

        Parameters:
        isotopes (dict): A dictionary with isotope names or CrossSection objects as keys and their respective weights as values.
        name (str): The name of the combined cross-section.
        vary_weights (bool): Whether to allow varying weights in calculations (default: False).
        """
        self.isotopes = isotopes
        self.weights = np.array(list(self.isotopes.values()))
        self.weights /= self.weights.sum()  # Normalize weights to 1
        self.name = name
        self.ufuncs = []

        # Populate the cross-section data
        xs = {}
        for isotope in self.isotopes:
            if isinstance(isotope, str):
                xs[isotope] = grab_from_endf(isotope)
            elif isinstance(isotope, CrossSection):
                xs[isotope.name] = isotope.table["total"].rename(isotope.name)
                isotope = isotope.name

            self.ufuncs.append(UnivariateSpline(xs[isotope].index.values, xs[isotope].values, k=1, s=0))

        # Update isotope names
        updated_isotopes = {}
        for isotope in self.isotopes:
            if isinstance(isotope, str):
                updated_isotopes[isotope] = self.isotopes[isotope]
            elif isinstance(isotope, CrossSection):
                updated_isotopes[isotope.name] = self.isotopes[isotope]

        self.isotopes = updated_isotopes

        # Create an interpolated table with the weighted sum of cross-sections
        table = pd.DataFrame(xs).interpolate()
        table["total"] = (table * self.weights).sum(axis=1)
        self.table = table

    def __call__(self, energies: np.ndarray,weights: np.ndarray=np.array([])) -> np.ndarray:
        """
        Calculate the weighted cross-section for given energies.
        
        Parameters:
        energies (np.ndarray): Array of energy values.
        weights (np.ndarray) (optional): Array of new weights
        
        Returns:
        np.ndarray: Array of weighted cross-section values.
        """
        if len(weights):
            return (np.array([ufunc(energies) for ufunc in self.ufuncs]).T * weights).sum(axis=1)
        else:
            return (np.array([ufunc(energies) for ufunc in self.ufuncs]).T * self.weights).sum(axis=1)

    def plot(self, **kwargs):
        """
        Plot the cross-section data.
        
        Parameters:
        kwargs: Additional arguments to pass to the plotting function.
        """
        self.table.mul(np.r_[self.weights, 1], axis=1).plot(**kwargs)
