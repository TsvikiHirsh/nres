import numpy as np
import site
import os
from pathlib import Path
import shelve
from tqdm import tqdm
import requests

# Constants
SPEED_OF_LIGHT = 299792458  # m/s
MASS_OF_NEUTRON = 939.56542052 * 1e6 / (SPEED_OF_LIGHT ** 2)  # [eV s²/m²]


def time2energy(time, flight_path_length):
    """
    Convert time-of-flight to energy of the neutron.

    Parameters:
    time (float): Time-of-flight in seconds.
    flight_path_length (float): Flight path length in meters.

    Returns:
    float: Energy of the neutron in electronvolts (eV).
    """
    γ = 1 / np.sqrt(1 - (flight_path_length / time) ** 2 / SPEED_OF_LIGHT ** 2)
    return (γ - 1) * MASS_OF_NEUTRON * SPEED_OF_LIGHT ** 2  # eV

def energy2time(energy, flight_path_length):
    """
    Convert energy to time-of-flight of the neutron.

    Parameters:
    energy (float): Energy of the neutron in electronvolts (eV).
    flight_path_length (float): Flight path length in meters.

    Returns:
    float: Time-of-flight in seconds.
    """
    γ = 1 + energy / (MASS_OF_NEUTRON * SPEED_OF_LIGHT ** 2)
    return flight_path_length / SPEED_OF_LIGHT * np.sqrt(γ ** 2 / (γ ** 2 - 1))

def materials_dict():
    import materials_compendium as mc
    materials = {}
    for material in mc.MaterialsCompendium:
        mat_name = material.Name
        density = material.Density
        n = material.MaterialAtomDensity
        formula = material.Formula
        elements = {}
        for element in material.Elements:
            name = element.Element
            weight = element.WeightFraction_whole
            elements[name] = {"weight":weight}
            isotopes = {}
            for isotope in element.Isotopes:
                iso_name = format_isotope(isotope.Isotope)
                iso_weight = isotope.WeightFraction
                if iso_weight>0:
                    isotopes[iso_name] = iso_weight
            elements[name]["isotopes"] = isotopes
        materials[mat_name] = {"name":mat_name,"density":density,"n":n,"formula":formula,"elements":elements}
    return materials

def elements_and_isotopes_dict():
    elements = {}
    import mendeleev
    all_elements = mendeleev.get_all_elements()
    for element in all_elements:
        name = element.name
        elements[name] = {}
    
        elements[name]["name"] = name
        elements[name]["n"] = element.density/element.mass*0.602214076 # atoms/barn
        elements[name]["formula"] = element.symbol
        elements[name]["density"] = element.density
        elements[name]["elements"] = {element.symbol:{"weight":1}}
        elements[name]["elements"][element.symbol]["isotopes"] = {f"{iso.element.symbol}{iso.mass_number}":iso.abundance*0.01 for iso in element.isotopes if iso.abundance}

    isotopes = {}
    for element in all_elements:
        for iso in element.isotopes:
            if iso.abundance:
                name = f"{iso.element.symbol}{iso.mass_number}"
                isotopes[name] = {}
                isotopes[name]["name"] = name
                isotopes[name]["n"] = element.density/iso.mass*0.602214076 # atoms/barn
                isotopes[name]["formula"] = name
                isotopes[name]["density"] = element.density
                isotopes[name]["elements"] = {element.symbol:{"weight":1}}
                isotopes[name]["elements"][element.symbol]["isotopes"] = {name:1.}
    return elements, isotopes


def format_isotope(isotope_string):
    # Find where the digits start in the string
    for i, char in enumerate(isotope_string):
        if char.isdigit():
            # Split the string into element and mass number parts
            element = isotope_string[:i]
            mass_number = isotope_string[i:]
            # Return the formatted string
            return f"{element}-{mass_number}"
    # Return the original string if no digits are found
    return isotope_string

def get_cache_path():
    # Get the user's site-packages directory
    user_site = site.getusersitepackages()
    # Create a subdirectory for our cache
    cache_dir = Path(user_site) / "nres"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def load_or_create_materials_cache():
    cache_path = get_cache_path() / "materials.db"
    
    if cache_path.exists():
        with shelve.open(str(cache_path.with_suffix(""))) as fid:
            materials = fid.get("materials")
            elements = fid.get("elements")
            isotopes = fid.get("isotopes")
        
        # If the cache is incomplete, regenerate it
        if materials is None or elements is None:
            return create_and_save_materials_cache()
        
        return materials, elements, isotopes
    else:
        return create_and_save_materials_cache()

def create_and_save_materials_cache():
    materials = materials_dict()
    elements, isotopes = elements_and_isotopes_dict()
    
    cache_path = get_cache_path() / "materials"
    with shelve.open(str(cache_path)) as fid:
        fid["materials"] = materials
        fid["elements"] = elements
        fid["isotopes"] = isotopes
    
    return materials, elements, isotopes

def download_xsdata():
    """Download the xsdata.npy file from GitHub and save it to the package directory."""
    url = 'https://github.com/lanl/trinidi-data/blob/main/xsdata.npy?raw=true'

    cache_path = get_cache_path() / "xsdata.npy"

    # Create the folder if it doesn't exist
    if not os.path.exists(cache_path.parent):
        os.makedirs(cache_path.parent)

    # Download with progress bar
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(str(cache_path), 'wb') as f, tqdm(
            desc=f"Downloading {cache_path}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ncols=80
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))
        print(f"File downloaded and saved to {cache_path}")
    else:
        raise Exception(f"Failed to download the file. Status code: {response.status_code}")