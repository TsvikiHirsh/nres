// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "integrate_xs.h"

namespace py = pybind11;

PYBIND11_MODULE(_integrate_xs, m) {

    // Bind the new CrossSectionCalculator class
    py::class_<CrossSectionCalculator>(m, "CrossSectionCalculator")
        .def(py::init<>())
        .def("initialize", &CrossSectionCalculator::initialize,
             "Initialize with energy grid and flight path length",
             py::arg("grid"),
             py::arg("flight_path_length") = 10.59,
             py::arg("tstep") = 1.5625e-9,
             py::arg("tau") = 1.0,
             py::arg("sigma") = 1.0,
             py::arg("x0") = 0.0)
        .def("add_xs_data", &CrossSectionCalculator::add_xs_data,
             "Add cross-section data from DataFrame-like structure",
             py::arg("energies"),
             py::arg("xs_data"))
        .def("calculate_xs", &CrossSectionCalculator::calculate_xs,
             "Calculate weighted cross-sections with time delay and response parameters",
             py::arg("user_energy_grid"),
             py::arg("fractions"),
             py::arg("t0") = 0.0,
             py::arg("L0") = 1.0,
             py::arg("tau") = 1e-9,
             py::arg("sigma") = 1e-9,
             py::arg("x0") = 0.0)
        .def("get_isotope_names", &CrossSectionCalculator::get_isotope_names)
        .def("get_energy_grid", &CrossSectionCalculator::get_energy_grid)
        .def("get_flight_path", &CrossSectionCalculator::get_flight_path)
        .def("get_response", &CrossSectionCalculator::get_response,
               py::arg("t0"), py::arg("L0"),  py::arg("tau") = 1e-9, py::arg("sigma") = 1e-9, py::arg("x0") = 0.0);


}