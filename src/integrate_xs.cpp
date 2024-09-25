#include <vector>
#include <cmath>
#include <algorithm>

// Constants
const double SPEED_OF_LIGHT = 299792458;  // m/s
const double MASS_OF_NEUTRON = 939.56542052 * 1e6 / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);  // [eV s²/m²]

// Linear interpolation function
double linear_interp(const std::vector<double>& xs_energies, const std::vector<double>& xs_values, double energy) {
    size_t n = xs_energies.size();

    // If the energy is out of bounds, return 0 (or handle as needed)
    if (energy <= xs_energies.front()) return xs_values.front();
    if (energy >= xs_energies.back()) return xs_values.back();

    // Find the two points in the grid between which the energy lies
    for (size_t i = 0; i < n - 1; ++i) {
        if (energy >= xs_energies[i] && energy <= xs_energies[i + 1]) {
            double slope = (xs_values[i + 1] - xs_values[i]) / (xs_energies[i + 1] - xs_energies[i]);
            return xs_values[i] + slope * (energy - xs_energies[i]);
        }
    }

    return 0.0; // Default return in case of error
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
            if (static_cast<size_t>(idx) >= 0 && idx < values.size()) {
                conv_sum += values[idx] * kernel[kernel_half_size + j];
            }
        }
        result[i] = conv_sum;
    }
    return result;
}

std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,      // Energy grid of cross-section data
    const std::vector<double>& xs_values,        // Cross-section values corresponding to xs_energies
    const std::vector<double>& energy_grid,      // User-defined energy grid for integration
    const std::vector<double>& kernel,           // Optional kernel for convolution
    double flight_path_length = 1.0)             // Flight path length (in meters), default is 1 meter
{
    // Calculate the number of bins to add based on the kernel length
    int num_bins_to_add = kernel.empty() ? 0 : kernel.size() / 2;

    // Extend the energy grid
    std::vector<double> extended_energy_grid = energy_grid;
    for (int i = 0; i < num_bins_to_add; ++i) {
        // Add prefix bin
        double new_prefix = energy_grid.front() * std::pow(energy_grid.front() / energy_grid[1], i + 1);
        extended_energy_grid.insert(extended_energy_grid.begin(), new_prefix);
        
        // Add suffix bin
        double new_suffix = energy_grid.back() * std::pow(energy_grid.back() / energy_grid[energy_grid.size() - 2], i + 1);
        extended_energy_grid.push_back(new_suffix);
    }

    std::vector<double> integrated_values;

    // Perform integration over the extended energy grid
    for (size_t i = 0; i < extended_energy_grid.size() - 1; ++i) {
        double emin = extended_energy_grid[i];
        double emax = extended_energy_grid[i + 1];

        if (emin <= 0 || emax <= 0) {
            integrated_values.push_back(0.0);
            continue;
        }

        // Create vectors to store energy points and cross-sections within the current bin
        std::vector<double> energy_points;
        std::vector<double> xs_points;

        // Add the lower edge of the bin
        energy_points.push_back(emin);
        xs_points.push_back(linear_interp(xs_energies, xs_values, emin));

        // Add points inside the bin (from xs_energies) if they lie between emin and emax
        for (size_t j = 0; j < xs_energies.size(); ++j) {
            if (xs_energies[j] > emin && xs_energies[j] < emax) {
                energy_points.push_back(xs_energies[j]);
                xs_points.push_back(xs_values[j]);
            }
        }

        // Add the upper edge of the bin
        energy_points.push_back(emax);
        xs_points.push_back(linear_interp(xs_energies, xs_values, emax));

        // Perform trapezoidal integration over the points in this bin
        double integral = 0.0;
        for (size_t j = 0; j < energy_points.size() - 1; ++j) {
            double x1 = energy_points[j];
            double x2 = energy_points[j + 1];
            double y1 = xs_points[j];
            double y2 = xs_points[j + 1];
            integral += 0.5 * (y1 + y2) * (x2 - x1);
        }

        // Calculate the average cross-section for this bin
        double avg_xs = integral / (emax - emin);
        integrated_values.push_back(avg_xs);
    }

    // If a kernel is provided and its size is greater than 1, convolve the result
    if (kernel.size() > 1) {
        integrated_values = convolve_with_kernel(integrated_values, kernel);
    }

    // Trim the result to match the original energy grid size
    if (num_bins_to_add > 0) {
        integrated_values.erase(integrated_values.begin(), integrated_values.begin() + num_bins_to_add - 1);
        integrated_values.erase(integrated_values.end() - num_bins_to_add, integrated_values.end());
    }

    // Replace NaN values with zero
    std::replace_if(integrated_values.begin(), integrated_values.end(), 
                    [](double val) { return std::isnan(val); }, 0.0);

    return integrated_values;
}
