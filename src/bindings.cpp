#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "integrate_xs.h"

namespace py = pybind11;

PYBIND11_MODULE(_integrate_xs, m) {
    py::class_<KernelParams>(m, "KernelParams")
        .def(py::init<>())
        .def_readwrite("shift_offset", &KernelParams::shift_offset)
        .def_readwrite("shift_slope", &KernelParams::shift_slope)
        .def_readwrite("stretch_offset", &KernelParams::stretch_offset)
        .def_readwrite("stretch_slope", &KernelParams::stretch_slope);

    m.def("integrate_cross_section", &integrate_cross_section,
        "Integrates cross-section data with optional kernel transformation parameters",
        py::arg("xs_energies"),
        py::arg("xs_values"),
        py::arg("energy_grid"),
        py::arg("kernel") = std::vector<double>{0.,1.,0.},
        py::arg("kernel_params") = KernelParams()
    );
}
