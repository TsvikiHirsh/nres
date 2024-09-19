import numpy as np

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
    import shelve
    with shelve.open("evaluated_data/materials") as fid:
        fid["materials"] = materials
    return materials

def elements_dict():
    mat = {}
    import mendeleev
    for element in mendeleev.get_all_elements():
        name = element.name
        if not element.is_radioactive:
            mat[name] = {}
        
            mat[name]["name"] = name
            mat[name]["n"] = element.density/element.mass*0.602214076 # atoms/barn
            mat[name]["formula"] = element.symbol
            mat[name]["density"] = element.density
            mat[name]["elements"] = {element.symbol:{"weight":1}}
            mat[name]["elements"][element.symbol]["isotopes"] = {f"{iso.element.symbol}-{iso.mass_number}":iso.abundance*0.01 if iso.abundance else 0 for iso in element.isotopes if iso.is_stable}
    import shelve
    with shelve.open("evaluated_data/materials") as fid:
        fid["elements"] = mat
    return mat

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