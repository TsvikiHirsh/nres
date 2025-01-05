#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "integrate_xs.h"

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

// Extend and normalize the kernel
std::vector<double> extend_and_normalize_kernel(const std::vector<double>& kernel, size_t pad_size) {
    size_t original_size = kernel.size();
    std::vector<double> extended_kernel(original_size + 2 * pad_size, 0.0);

    // Fill the center with the original kernel
    std::copy(kernel.begin(), kernel.end(), extended_kernel.begin() + pad_size);

    // Mirror padding on both sides
    for (size_t i = 0; i < pad_size; ++i) {
        extended_kernel[pad_size - 1 - i] = kernel[i];
        extended_kernel[pad_size + original_size + i] = kernel[original_size - 1 - i];
    }

    // Normalize the extended kernel
    double sum = std::accumulate(extended_kernel.begin(), extended_kernel.end(), 0.0);
    if (sum > 0) {
        for (double& val : extended_kernel) {
            val /= sum;
        }
    }

    return extended_kernel;
}

// Transform kernel based on energy
std::vector<double> apply_kernel_transformation(
    const std::vector<double>& kernel,
    double energy,
    const KernelParams& params) 
{
    // Calculate energy-dependent shift and stretch
    double shift = params.shift_slope * (energy - params.shift_offset);
    double stretch = params.stretch_slope * (energy - params.stretch_offset);
    
    std::vector<double> transformed_kernel(kernel.size());
    int middle = kernel.size() / 2;
    
    for (size_t i = 0; i < kernel.size(); ++i) {
        // Apply stretch relative to the center of the kernel
        int relative_pos = i - middle;
        double stretched_pos = relative_pos * stretch;
        
        // Find the position after shifting
        double final_pos = stretched_pos + shift + middle;
        
        // Linear interpolation between kernel points
        if (final_pos >= 0 && final_pos < kernel.size() - 1) {
            int lower_idx = static_cast<int>(final_pos);
            double fraction = final_pos - lower_idx;
            transformed_kernel[i] = kernel[lower_idx] * (1 - fraction) + 
                                  kernel[lower_idx + 1] * fraction;
        } else {
            transformed_kernel[i] = 0.0;  // Out of bounds
        }
    }
    
    // Normalize the transformed kernel
    double sum = std::accumulate(transformed_kernel.begin(), 
                               transformed_kernel.end(), 0.0);
    if (sum > 0) {
        for (double& val : transformed_kernel) {
            val /= sum;
        }
    }
    
    return transformed_kernel;
}

// Convolve with energy-dependent kernel
std::vector<double> convolve_with_kernel(
    const std::vector<double>& values,
    const std::vector<double>& kernel,
    const std::vector<double>& energy_grid,
    const KernelParams& kernel_params) 
{
    int values_size = values.size();
    int kernel_size = kernel.size();
    int pad_size = kernel_size / 2;

    // Extend and normalize the kernel
    std::vector<double> extended_kernel = extend_and_normalize_kernel(kernel, pad_size);

    // Create a padded version of the input values
    std::vector<double> padded_values(values_size + 2 * pad_size, 0.0);
    std::copy(values.begin(), values.end(), padded_values.begin() + pad_size);

    // Result vector will have the same size as the original values
    std::vector<double> result(values_size, 0.0);

    // Perform convolution with energy-dependent kernel
    for (int i = 0; i < values_size; ++i) {
        // Get transformed kernel for current energy
        std::vector<double> transformed_kernel = 
            apply_kernel_transformation(extended_kernel, energy_grid[i], kernel_params);
            
        double conv_sum = 0.0;
        for (int j = 0; j < kernel_size; ++j) {
            conv_sum += padded_values[i + j] * transformed_kernel[j];
        }
        result[i] = conv_sum;
    }

    return result;
}

std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,
    const std::vector<double>& xs_values,
    const std::vector<double>& energy_grid,
    const std::vector<double>& kernel,
    const KernelParams& kernel_params)
{
    // Handle the case when xs_values has only one element
    if (xs_values.size() == 1) {
        return std::vector<double>(energy_grid.size(), xs_values[0]);
    }

    // Calculate the number of bins to add based on the kernel length
    int num_bins_to_add = kernel.empty() ? 0 : kernel.size() - 1;

    // Extend the energy grid
    std::vector<double> extended_energy_grid = energy_grid;
    for (int i = 0; i < num_bins_to_add; ++i) {
        // Add prefix bin
        double new_prefix = extended_energy_grid.front() * 
            std::pow(extended_energy_grid.front() / extended_energy_grid[1], 1);
        extended_energy_grid.insert(extended_energy_grid.begin(), new_prefix);
        
        // Add suffix bin
        double new_suffix = extended_energy_grid.back() * 
            std::pow(extended_energy_grid.back() / extended_energy_grid[extended_energy_grid.size() - 2], 1);
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

        // Perform trapezoidal integration
        double integral = 0.5 * (linear_interp(xs_energies, xs_values, emin) + 
                               linear_interp(xs_energies, xs_values, emax)) * (emax - emin);

        // Calculate the average cross-section for this bin
        double avg_xs = integral / (emax - emin);
        integrated_values.push_back(avg_xs);
    }

    // If a kernel is provided and its size is greater than 1, convolve the result
    if (kernel.size() > 1) {
        integrated_values = convolve_with_kernel(integrated_values, kernel, 
                                               extended_energy_grid, kernel_params);
    }

    // Trim the result to match the original energy grid size
    size_t expected_size = energy_grid.size();
    if (integrated_values.size() > expected_size) {
        size_t extra = integrated_values.size() - expected_size;
        size_t to_remove_start = extra / 2;
        size_t to_remove_end = extra - to_remove_start;
        integrated_values.erase(integrated_values.begin(), 
                              integrated_values.begin() + to_remove_start);
        integrated_values.erase(integrated_values.end() - to_remove_end, 
                              integrated_values.end());
    }

    // Ensure the result matches the original energy grid size
    if (integrated_values.size() != expected_size) {
        integrated_values.resize(expected_size, integrated_values.back());
    }

    // Replace NaN values with zero
    std::replace_if(integrated_values.begin(), integrated_values.end(), 
                    [](double val) { return std::isnan(val); }, 0.0);

    return integrated_values;
}
