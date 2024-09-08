import os
import requests
from bs4 import BeautifulSoup
import zipfile
import pandas as pd
import endf
import glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline


def format_isotope_name(isotope):
    """Format the isotope name to match the GitHub repository format."""
    isotope = isotope.replace('-', '').replace(' ', '')
    element = ''.join([i for i in isotope if not i.isdigit()])
    atomic_number = ''.join([i for i in isotope if i.isdigit()])
    return f"{element}{atomic_number.zfill(3)}"

def grab_from_endf(isotope, evaluated_data_dir="evaluated_data"):
    """Download and save the cross-section data for the specified isotope if it doesn't exist locally."""
    formatted_isotope = format_isotope_name(isotope)
    url = (f"https://raw.githubusercontent.com/pedrojrv/ML_Nuclear_Data/master/"
           f"Evaluated_Data/neutrons/{formatted_isotope}/endfb8.0/tables/xs/"
           f"n-{formatted_isotope}-MT001.endfb8.0")
    file_path = os.path.join(evaluated_data_dir, f"n-{formatted_isotope}-MT001.endfb8.0")
    
    # Check if the file already exists
    if not os.path.exists(file_path):
        # If not, download and save the file
        print(f"Downloading data for {isotope}...")
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        
        os.makedirs(evaluated_data_dir, exist_ok=True)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Data saved to {file_path}")
    
    # Read the file using pandas
    xs = pd.read_csv(file_path, comment="#", delim_whitespace=True, header=None, names=["E", "xs"], usecols=[0, 1], index_col=0)
    xs = xs[~xs.index.duplicated()]
    xs["xs"]*=0.001 # convert to barns
    return xs["xs"].rename(isotope)


class Cross_section:

    def __init__(self,
                 isotopes:dict={},
                 name:str="",
                 vary_weights:bool=False):
        
        self.isotopes = isotopes
        self.weights = np.array(list(self.isotopes.values()))
        self.weights/=self.weights.sum() # normalize weights to 1
        self.name = name

        # populate the cross section data
        xs = {}
        self.ufuncs = []
        for isotope in self.isotopes:
            if type(isotope)==str:
                xs[isotope] = grab_from_endf(isotope)
            elif type(isotope)==Cross_section:
                xs[isotope.name] = isotope.table["total"].rename(isotope.name)
                
                isotope = isotope.name

            self.ufuncs.append(UnivariateSpline((xs[isotope].index.values,), xs[isotope].values,k=1,s=0))

        # update names in the isotopes object
        isotopes = {}
        for isotope in self.isotopes:
            if type(isotope)==str:
                isotopes[isotope] = self.isotopes[isotope]
            elif type(isotope)==Cross_section:
                isotopes[isotope.name] = self.isotopes[isotope]

        self.isotopes = isotopes


        table = pd.DataFrame(xs).interpolate()
        table["total"] = (table.interpolate()*self.weights).sum(1)
        self.table = table

    def __call__(self,energies):
        return (np.array([u(energies) for u in self.ufuncs]).T*self.weights).sum(1)
    
    def plot(self,**kwargs):
        self.table.mul(np.r_[self.weights,1],axis=1).plot(**kwargs)

    
        
        