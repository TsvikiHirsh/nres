from setuptools import setup, Extension, find_packages
import pybind11

# Define the extension module
ext_modules = [
    Extension(
        "nres._integrate_xs",
        sources=["src/bindings.cpp", "src/integrate_xs.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "include"
        ],
        extra_compile_args=["-std=c++17"]  # Add C++17 requirement
    )
]

# Use setup to build the extension module
setup(
    name="nres",
    version="0.3.0",
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


