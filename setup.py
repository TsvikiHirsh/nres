from __future__ import annotations

from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup

# Define the extension module using Pybind11Extension
ext_modules = [
    Pybind11Extension(
        "nres._integrate_xs",
        sources=["src/bindings.cpp", "src/integrate_xs.cpp"],
        include_dirs=["include"],
        cxx_std=17,  # Use cxx_std instead of extra_compile_args
    )
]

# Use setup to build the extension module
setup(
    name="nres",
    version="0.4.0",
    author="Tsviki Y. Hirsh",
    author_email="tsviki.hirsh@gmail.com",
    description="Simple yet powerful package for neutron resonance fitting",
    packages=find_packages(where="src"),  # Look for packages in src/
    package_dir={"": "src"},  # Tell setuptools that packages are in src/
    include_package_data=True,  # To include non-python files
    package_data={
        "nres": ["data/nres_materials.json"],  # Specify the JSON file as package data
    },
    python_requires=">=3.8",
    ext_modules=ext_modules,
    zip_safe=False,
)


