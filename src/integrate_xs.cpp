#include <vector>
#include <cmath>
#include <algorithm>

// Perform linear interpolation of cross-section values
double linear_interp(const std::vector<double>& x, const std::vector<double>& y, double xi) {
    // Find the interval containing xi
    auto it = std::lower_bound(x.begin(), x.end(), xi);
    
    // If xi is outside the known grid, return 0 (or extrapolate if needed)
    if (it == x.begin()) return y.front();
    if (it == x.end()) return y.back();
    
    // Get the two bounding points
    int i = std::distance(x.begin(), it) - 1;
    double x0 = x[i], x1 = x[i + 1];
    double y0 = y[i], y1 = y[i + 1];
    
    // Linear interpolation formula
    return y0 + (xi - x0) * (y1 - y0) / (x1 - x0);
}

// Generate equally spaced energy points within the bin
std::vector<double> generate_energy_points(double emin, double emax, int num_points) {
    std::vector<double> points(num_points);
    double step = (emax - emin) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        points[i] = emin + i * step;
    }
    return points;
}

// Trapezoidal integration of cross sections with at least 10 points per bin
std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,  // Energy grid of cross-section data
    const std::vector<double>& xs_values,    // Cross-section values corresponding to xs_energies
    const std::vector<double>& energy_grid)  // User-defined energy grid for integration
{
    std::vector<double> integrated_values;
    integrated_values.reserve(energy_grid.size() - 1);

    int min_points = 10;  // Ensure at least 10 points in each bin
    
    for (size_t i = 0; i < energy_grid.size() - 1; ++i) {
        double emin = energy_grid[i];
        double emax = energy_grid[i + 1];
        
        // Check if the energy range is valid
        if (emin <= 0 || emax <= 0) {
            integrated_values.push_back(0.0);
            continue;
        }
        
        // Generate at least 10 points in the current bin
        std::vector<double> e_points = generate_energy_points(emin, emax, min_points);
        double integral = 0.0;
        
        // Trapezoidal integration using interpolated cross-section values
        for (size_t j = 0; j < e_points.size() - 1; ++j) {
            double e1 = e_points[j];
            double e2 = e_points[j + 1];
            
            double xs1 = linear_interp(xs_energies, xs_values, e1);
            double xs2 = linear_interp(xs_energies, xs_values, e2);
            
            // Trapezoidal rule: 0.5 * (f(x1) + f(x2)) * (x2 - x1)
            integral += 0.5 * (xs1 + xs2) * (e2 - e1);
        }
        
        integrated_values.push_back(integral);
    }
    
    return integrated_values;
}
