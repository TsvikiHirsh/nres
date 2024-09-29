Welcome to nres's documentation!
================================

nres (pronounced N-res) is a simple yet powerful package for fitting neutron resonances. It allows quick and easy quantitative fitting of total cross-section for neutron absorption resonances transmission spectrum in the epithermal and fast energy regions.

Features
--------
- Flexible and simple way to generate and combine cross-sections from different isotopic materials.
- Built-in database of many useful materials and elements.
- Cross-sections taken from ENDF8.0.
- Built on `lmfit` for intuitive and powerful fit exploration.
- Python API leveraging popular libraries like `numpy` and `pandas`.
- Methods to define response functions and background functions.
- Plotting utilities for concise result visualization.
- Fast cross-section integration and convolution with response function using C++ core code.

Installation
------------
To install from source, you can clone the repository and install it using pip:

.. code-block:: bash

    git clone https://github.com/TsvikiHirsh/nres
    cd nres
    pip install .

Basic Usage
-----------
Here's a quick example of how to use `nres`:

.. code-block:: python

    # Import nres
    import nres

    # Define material
    Si = nres.CrossSection.from_material("Silicon")

    # Load data
    data = nres.Data.from_transmission("silicon.dat")

    # Define model
    model = nres.TransmissionModel(Si, vary_background=True)

    # Fit using lmfit
    result = model.fit(data, emin=0.4e6, emax=1.7e6)

    # Plot fit results
    result.plot()

.. image:: https://github.com/tsvikihirsh/nres/raw/main/docs/images/silicon_fit.png
   :alt: Fit results
   :align: center
   :width: 600px

For more detailed examples and advanced usage, please refer to our Jupyter `notebook demo <nhttps://github.com/tsvikihirsh/nres/otebooks/nres_demo.ipynb>`_.

Contributing
------------
See `CONTRIBUTING.md <https://github.com/tsvikihirsh/nres/CONTRIBUTING.md>`_ for instructions on how to contribute.

License
-------
Distributed under the terms of the `MIT license <https://github.com/tsvikihirsh/nres/LICENSE>`_.

Contact
-------
For questions, issues, or contributions, please visit the `GitHub repository <https://github.com/tsvikihirsh/nres>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules  # Add your modules.rst file here

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
