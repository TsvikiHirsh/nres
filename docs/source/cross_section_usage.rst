Using nres.CrossSection for Material Definition and Combination
===============================================================

This guide demonstrates various ways to define materials and combine them using the ``nres.CrossSection.from_material`` method. We'll also cover how to view cross-section weights, access the cross-section table, and plot the results.

Defining Materials
------------------

From Elements
^^^^^^^^^^^^^

You can create a ``CrossSection`` object from a single element:

.. code-block:: python

    import nres

    # Create a CrossSection for iron
    iron_xs = nres.CrossSection.from_material("Fe")

    # View the weights of isotopes in iron
    print(iron_xs.weights)

    # Plot the cross-section
    iron_xs.plot()

From Compounds
^^^^^^^^^^^^^^

For compounds, you can use their chemical formula:

.. code-block:: python

    # Create a CrossSection for water
    water_xs = nres.CrossSection.from_material("H2O")

    # View the cross-section table
    print(water_xs.table)

From Custom Materials
^^^^^^^^^^^^^^^^^^^^^

You can also define custom materials using dictionaries:

.. code-block:: python

    custom_material = {
        "name": "Custom Alloy",
        "formula": "Cu70Ni30",
        "elements": {
            "Cu": {"weight": 0.7},
            "Ni": {"weight": 0.3}
        },
        "n": 8.94  # Number density in units of 1e28 m^-3
    }

    custom_xs = nres.CrossSection.from_material(custom_material)

Combining Materials
-------------------

You can combine different materials using the ``__add__`` method:

.. code-block:: python

    # Combine iron and nickel
    iron_nickel_xs = iron_xs + nres.CrossSection.from_material("Ni")

    # Plot the combined cross-section
    iron_nickel_xs.plot(title="Iron-Nickel Alloy")

Specifying Split Options
------------------------

The ``from_material`` method allows you to specify how to split the cross-sections:

.. code-block:: python

    # Split by isotopes
    water_isotopes = nres.CrossSection.from_material("H2O", splitby="isotopes")

    # Split by elements
    water_elements = nres.CrossSection.from_material("H2O", splitby="elements")

    # Split by materials (useful for complex mixtures)
    water_material = nres.CrossSection.from_material("H2O", splitby="materials")

Working with nres Dictionaries
------------------------------

You can access predefined materials, elements, and isotopes from nres dictionaries:

.. code-block:: python

    # Using nres.materials
    steel_xs = nres.CrossSection.from_material(nres.materials["Steel"])

    # Using nres.elements
    carbon_xs = nres.CrossSection.from_material(nres.elements["C"])

    # Using nres.isotopes
    u235_xs = nres.CrossSection.from_material(nres.isotopes["U-235"])

Viewing and Analyzing Cross-Sections
------------------------------------

Accessing Weights and Table Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # View weights of components
    print(steel_xs.weights)

    # Access the cross-section table
    print(steel_xs.table.head())

Plotting Cross-Sections
^^^^^^^^^^^^^^^^^^^^^^^

The ``plot`` method allows for customization:

.. code-block:: python

    steel_xs.plot(
        title="Steel Cross-Section",
        xlabel="Energy (eV)",
        ylabel="Cross-Section (barn)",
        lw=2,
        logx=True,
        logy=True
    )

Advanced Usage
--------------

Grouping by Isotopes
^^^^^^^^^^^^^^^^^^^^

You can group cross-sections by isotopes:

.. code-block:: python

    grouped_steel = steel_xs.groupby_isotopes()
    grouped_steel.plot(title="Steel Cross-Section (Grouped by Elements)")


This guide provides a comprehensive overview of using ``nres.CrossSection`` for defining and combining materials, as well as analyzing and visualizing cross-section data. Experiment with different materials and combinations to explore their neutron interaction properties!