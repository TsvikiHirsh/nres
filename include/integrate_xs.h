#ifndef INTEGRATE_XS_H
#define INTEGRATE_XS_H
#include <vector>
#include <string>

struct KernelParams {
    double shift_offset = 0.0;    // s0: base shift
    double shift_slope = 0.0;     // s1: energy-dependent shift coefficient
    double stretch_offset = 1.0;  // c0: base stretch (default 1.0 means no stretch)
    double stretch_slope = 0.0;   // c1: energy-dependent stretch coefficient
};

std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,    // Cross-section energies
    const std::vector<double>& xs_values,      // Cross-section values
    const std::vector<double>& energy_grid,    // User energy grid
    const std::vector<double>& kernel = std::vector<double>{0.,1.,0.}, // Default kernel
    const KernelParams& kernel_params = KernelParams()  // Default kernel parameters
);
#endif // INTEGRATE_XS_H