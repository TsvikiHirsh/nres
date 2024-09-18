#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric> 
#include <iostream>  // For debug output

// Constants
const double SPEED_OF_LIGHT = 299792458;  // m/s
const double MASS_OF_NEUTRON = 939.56542052 * 1e6 / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);  // [eV s²/m²]

// Perform linear interpolation of cross-section values
double linear_interp(const std::vector<double>& x, const std::vector<double>& y, double xi) {
    auto it = std::lower_bound(x.begin(), x.end(), xi);
    if (it == x.begin()) return y.front();
    if (it == x.end()) return y.back();
    int i = std::distance(x.begin(), it) - 1;
    double x0 = x[i], x1 = x[i + 1];
    double y0 = y[i], y1 = y[i + 1];
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

// Calculate the bin spacing in 1/sqrt(E) and add a new bin accordingly
std::vector<double> add_prefix_bin(const std::vector<double>& energy_grid) {
    if (energy_grid.size() < 2) return energy_grid;

    // Calculate 1/sqrt(E) for the first two energy bins
    double inv_sqrt_E1 = 1.0 / std::sqrt(energy_grid[0]);
    double inv_sqrt_E2 = 1.0 / std::sqrt(energy_grid[1]);

    // The difference in 1/sqrt(E) is approximately constant
    double spacing = inv_sqrt_E2 - inv_sqrt_E1;

    // Calculate the new bin by subtracting the spacing in 1/sqrt(E) and converting back to energy
    double new_inv_sqrt_E = inv_sqrt_E1 - spacing;
    double new_bin = 1.0 / (new_inv_sqrt_E * new_inv_sqrt_E);  // Convert back to energy

    // Prepend the new bin to the energy grid
    std::vector<double> new_grid = { new_bin };
    new_grid.insert(new_grid.end(), energy_grid.begin(), energy_grid.end());
    return new_grid;
}

// Convert time-of-flight (tof) to neutron energy (in eV) using relativistic formula
double time2energy(double tof, double flight_path_length) {
    double v = flight_path_length / tof;
    double gamma = 1.0 / std::sqrt(1.0 - (v * v) / (SPEED_OF_LIGHT * SPEED_OF_LIGHT));
    return (gamma - 1.0) * MASS_OF_NEUTRON * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
}

// Convert neutron energy (in eV) to time-of-flight (tof) using relativistic formula
double energy2time(double energy, double flight_path_length) {
    double gamma = 1.0 + energy / (MASS_OF_NEUTRON * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
    double v = SPEED_OF_LIGHT * std::sqrt(1.0 - 1.0 / (gamma * gamma));
    return flight_path_length / v;
}

// Perform convolution with kernel in the time domain
std::vector<double> convolve_with_kernel(const std::vector<double>& values, const std::vector<double>& kernel) {
    std::vector<double> result(values.size(), 0.0);
    int kernel_half_size = kernel.size() / 2;
    
    for (size_t i = 0; i < values.size(); ++i) {
        double conv_sum = 0.0;
        for (int j = -kernel_half_size; j <= kernel_half_size; ++j) {
            int idx = i + j;
            if (idx >= 0 && idx < values.size()) {
                conv_sum += values[idx] * kernel[kernel_half_size + j];
            }
        }
        result[i] = conv_sum;
    }
    return result;
}

// Trapezoidal integration of cross sections with at least 10 points per bin, optional convolution
std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,      // Energy grid of cross-section data
    const std::vector<double>& xs_values,        // Cross-section values corresponding to xs_energies
    const std::vector<double>& energy_grid,      // User-defined energy grid for integration
    const std::vector<double>& kernel,      // Optional kernel for convolution
    double flight_path_length)             // Flight path length (in meters), default is 1 meter
{
    std::vector<double> integrated_values;
    
    // Add a new bin based on the spacing pattern and update the energy grid
    std::vector<double> updated_grid = add_prefix_bin(energy_grid);
    
    int min_points = 10;  // Ensure at least 10 points in each bin

    // Create a vector to store time values for the corresponding energy points
    std::vector<double> times;
    for (size_t i = 0; i < updated_grid.size() - 1; ++i) {
        double emin = updated_grid[i];
        double emax = updated_grid[i + 1];
        
        if (emin <= 0 || emax <= 0) {
            integrated_values.push_back(0.0);
            continue;
        }
        
        // Generate at least 10 points in the current bin
        std::vector<double> e_points = generate_energy_points(emin, emax, min_points);
        double integral = 0.0;
        
        for (size_t j = 0; j < e_points.size() - 1; ++j) {
            double e1 = e_points[j];
            double e2 = e_points[j + 1];
            
            double xs1 = linear_interp(xs_energies, xs_values, e1);
            double xs2 = linear_interp(xs_energies, xs_values, e2);
            
            integral += 0.5 * (xs1 + xs2) * (e2 - e1);
            
            // Convert energy points to time-of-flight
            double t1 = energy2time(e1, flight_path_length);
            double t2 = energy2time(e2, flight_path_length);
            times.push_back(t1);
            times.push_back(t2);
        }
        
        integrated_values.push_back(integral);
    }
    
    // If a kernel is provided, convolve the result in the time domain
    if (!kernel.empty()) {
        integrated_values = convolve_with_kernel(integrated_values, kernel);
    }

    return integrated_values;
}
