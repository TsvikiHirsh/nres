import os
import requests
from bs4 import BeautifulSoup
import zipfile
import pandas as pd
import endf
import glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import periodictable

# Constants
BASE_URL = "https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/"
SPEED_OF_LIGHT = 299792458  # m/s
MASS_OF_NEUTRON = 939.56542052 * 1e6 / (SPEED_OF_LIGHT ** 2)  # [eV s²/m²]

def _find_isotope_url(isotope):
    """
    Find the URL for the isotope's ENDF file by parsing the IAEA webpage.

    Parameters:
    isotope (str): The isotope designation, e.g., "Mn-55".

    Returns:
    str: The full URL of the isotope's ENDF file or None if not found.
    """
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and isotope in href:
            return BASE_URL + href
    
    return None

def _check_endf_file_exists(isotope, directory="endf_database"):
    """
    Check if the ENDF file for a given isotope exists in the specified directory.

    Parameters:
    isotope (str): The isotope designation, e.g., "Mn-55".
    directory (str): The directory to search for the ENDF file.

    Returns:
    str or None: The path to the ENDF file if it exists, otherwise None.
    """
    expected_file_path = os.path.join(directory, f"{isotope}.dat")
    if os.path.exists(expected_file_path):
        return expected_file_path
    return None

def _download_isotope_data(isotope, archive_folder="endf_database"):
    """
    Download the isotope data if it doesn't already exist in the archive.

    Parameters:
    isotope (str): The isotope designation, e.g., "Mn-55".
    archive_folder (str): The directory to save the downloaded zip files.

    Returns:
    str: The path to the downloaded zip file or None if download failed.
    """
    os.makedirs(archive_folder, exist_ok=True)
    
    isotope_url = _find_isotope_url(isotope)
    if not isotope_url:
        print(f"Isotope {isotope} not found on the website.")
        return None
    
    zip_file_path = os.path.join(archive_folder, os.path.basename(isotope_url))
    
    if os.path.exists(zip_file_path):
        print(f"{os.path.basename(isotope_url)} already exists in the archive.")
        return zip_file_path
    
    response = requests.get(isotope_url)
    if response.status_code == 200:
        with open(zip_file_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {os.path.basename(isotope_url)}")
    else:
        print(f"Failed to download {os.path.basename(isotope_url)}.")
        return None
    
    return zip_file_path

def _extract_endf_file(zip_file_path, extract_folder="endf_database"):
    """
    Extract the ENDF file from the downloaded zip file.

    Parameters:
    zip_file_path (str): The path to the zip file.
    extract_folder (str): The directory to extract the ENDF files.

    Returns:
    str: The path to the extracted ENDF file.
    """
    os.makedirs(extract_folder, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        extracted_files = zip_ref.namelist()
    
    # Remove the zip file after extraction
    os.remove(zip_file_path)
    
    return os.path.join(extract_folder, extracted_files[0])

def _read_total_cross_section(endf_file_path, z, a):
    """
    Read the total cross section (MT=1) from the ENDF file.

    Parameters:
    endf_file_path (str): The path to the ENDF file.
    z (int): The atomic number of the element.
    a (int): The mass number of the isotope.

    Returns:
    pandas.Series: A series with energy (MeV) as index and cross-section (barns) as values.
    """
    neutron_data = endf.IncidentNeutron(z, int(a), 0)
    reactions = neutron_data.from_endf(endf_file_path)
    total_xs = reactions.reactions[1]  # MT=1 corresponds to total cross-section
    
    xs_table = total_xs.xs["0K"]  # Assuming 0K temperature
    return pd.Series(xs_table.y, index=xs_table.x, name="Total Cross Section (barns)")

def total_xs(isotope="Mn-55", energies=None):
    """
    Get the total cross section of a specified isotope from ENDF.

    Parameters:
    isotope (str): Isotope name, e.g., "U-238".
    energies (list, optional): List of energies for interpolation. Defaults to None.

    Returns:
    pandas.Series: Series with energy as the index and cross-section as the value.
    """
    endf_file_path = _check_endf_file_exists(isotope)
    
    if not endf_file_path:
        # Download the isotope data
        zip_file_path = _download_isotope_data(isotope)
        if zip_file_path:
            # Extract the ENDF file
            endf_file_path = _extract_endf_file(zip_file_path)
    
    name, a = isotope.split("-")
    z = periodictable.elements.isotope(name).number

    cross_section_series = _read_total_cross_section(endf_file_path, z, a)

    cross_section_series = cross_section_series[~cross_section_series.index.duplicated()]
    

    
    if energies:
        # interpolation object
        uxs = RegularGridInterpolator((cross_section_series.index.values,), cross_section_series.values)
        cross_section_series = pd.Series(uxs(energies), index=energies)

    return cross_section_series

class Cross_section:

    def __init__(self,
                 isotopes:dict={},
                 name:str="",
                      vary_weights:bool=False):
        
        self.isotopes = isotopes

        self.weights = list(self.isotopes.values())
        self.name = name
        self._get_xs()

    def _get_xs(self):
        xs = {}
        self.xs_funcs = {}
        for isotope in self.isotopes:
            if type(isotope)==str:
                xs[isotope] = total_xs(isotope)
            elif type(isotope)==Cross_section:
                xs[isotope.name] = isotope.xs_table["total"].rename(isotope.name)
                
                isotope = isotope.name

            self.xs_funcs[isotope] = RegularGridInterpolator((xs[isotope].index.values,), xs[isotope].values)

        isotopes = {}
        for isotope in self.isotopes:
            if type(isotope)==str:
                isotopes[isotope] = self.isotopes[isotope]
            elif type(isotope)==Cross_section:
                isotopes[isotope.name] = self.isotopes[isotope]
        self.isotopes = isotopes

        table = pd.DataFrame(xs)
        table["total"] = (table.interpolate()*self.weights).sum(1)
        self.xs_table = table

    
        
        