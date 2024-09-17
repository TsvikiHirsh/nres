#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic conversion between Python lists and std::vector
#include "integrate_xs.h"

namespace py = pybind11;

PYBIND11_MODULE(_integrate_xs, m) {
    m.def("integrate_cross_section", &integrate_cross_section, "A function for cross-section integration.",
          py::arg("xs_energies"), py::arg("xs_values"), py::arg("energy_grid"));  // Add argument names for clarity
}