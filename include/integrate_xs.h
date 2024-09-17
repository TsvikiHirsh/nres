#ifndef INTEGRATE_XS_H
#define INTEGRATE_XS_H

#include <vector>

// Declare the function with the correct signature
std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,  // Energy grid of cross-section data
    const std::vector<double>& xs_values,    // Cross-section values corresponding to xs_energies
    const std::vector<double>& energy_grid);  // User-defined energy grid for integration

#endif // INTEGRATE_XS_H
