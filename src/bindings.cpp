#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "integrate_xs.h"

namespace py = pybind11;

PYBIND11_MODULE(_integrate_xs, m) {
    // Bind the original function for backward compatibility
    m.def("integrate_cross_section", &integrate_cross_section,
          "Integrates cross-section data with optional response and flight path length.",
          py::arg("xs_energies"),
          py::arg("xs_values"),
          py::arg("energy_grid"),
          py::arg("response") = std::vector<double>{0., 1., 0.}
    );

    // Bind the new CrossSectionCalculator class
    py::class_<CrossSectionCalculator>(m, "CrossSectionCalculator")
        .def(py::init<>())
        .def("initialize", &CrossSectionCalculator::initialize,
             "Initialize with energy grid and optional response kernel",
             py::arg("grid"),
             py::arg("kernel") = std::vector<double>{0., 1., 0.})
        .def("add_isotope", &CrossSectionCalculator::add_isotope,
             "Add isotope cross-section data",
             py::arg("name"),
             py::arg("energies"),
             py::arg("xs_values"))
        .def("calculate_xs", &CrossSectionCalculator::calculate_xs,
             "Calculate weighted cross-sections using provided fractions",
             py::arg("fractions"))
        .def("get_isotope_names", &CrossSectionCalculator::get_isotope_names,
             "Get list of loaded isotope names")
        .def("get_energy_grid", &CrossSectionCalculator::get_energy_grid,
             "Get current energy grid")
        .def("get_response_kernel", &CrossSectionCalculator::get_response_kernel,
             "Get current response kernel");
}
