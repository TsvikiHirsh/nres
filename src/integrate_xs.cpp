// integrate_xs.cpp
#include "integrate_xs.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Implementation of the backward compatibility function
std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,
    const std::vector<double>& xs_values,
    const std::vector<double>& energy_grid,
    const std::vector<double>& response)
{
    // Create a temporary calculator instance
    CrossSectionCalculator calc;
    
    // Initialize with the provided grid and response
    calc.initialize(energy_grid, response);
    
    // Add the single isotope data
    calc.add_isotope("temp", xs_energies, xs_values);
    
    // Calculate with weight 1.0 (100% of this isotope)
    return calc.calculate_xs({1.0});
}

// Class method implementations
void CrossSectionCalculator::initialize(const std::vector<double>& grid, 
                                      const std::vector<double>& kernel) {
    energy_grid = grid;
    response_kernel = kernel;
}

void CrossSectionCalculator::add_isotope(const std::string& name,
                                       const std::vector<double>& energies,
                                       const std::vector<double>& xs_values) {
    if (energies.size() != xs_values.size()) {
        throw std::invalid_argument("Energy and cross-section arrays must have same size");
    }
    isotope_xs_data[name] = {energies, xs_values};
    // Only add to names if it's not already there
    if (std::find(isotope_names.begin(), isotope_names.end(), name) == isotope_names.end()) {
        isotope_names.push_back(name);
    }
}

double CrossSectionCalculator::linear_interp(const std::vector<double>& xs_energies, 
                                           const std::vector<double>& xs_values, 
                                           double energy) const {
    if (energy <= xs_energies.front()) return xs_values.front();
    if (energy >= xs_energies.back()) return xs_values.back();

    for (size_t i = 0; i < xs_energies.size() - 1; ++i) {
        if (energy >= xs_energies[i] && energy <= xs_energies[i + 1]) {
            double slope = (xs_values[i + 1] - xs_values[i]) / 
                         (xs_energies[i + 1] - xs_energies[i]);
            return xs_values[i] + slope * (energy - xs_energies[i]);
        }
    }
    return 0.0;
}

std::vector<double> CrossSectionCalculator::convolve_with_kernel(
    const std::vector<double>& values) const {
    int values_size = values.size();
    int kernel_size = response_kernel.size();
    int pad_size = kernel_size / 2;

    std::vector<double> padded_values(values_size + 2 * pad_size, 0.0);
    std::copy(values.begin(), values.end(), padded_values.begin() + pad_size);

    std::vector<double> result(values_size, 0.0);
    for (int i = 0; i < values_size; ++i) {
        double conv_sum = 0.0;
        for (int j = 0; j < kernel_size; ++j) {
            conv_sum += padded_values[i + j] * response_kernel[j];
        }
        result[i] = conv_sum;
    }
    return result;
}

std::vector<double> CrossSectionCalculator::calculate_xs(
    const std::vector<double>& fractions) const {
    if (fractions.size() != isotope_names.size()) {
        throw std::invalid_argument("Number of fractions must match number of isotopes");
    }

    std::vector<double> total_xs(energy_grid.size(), 0.0);
    
    for (size_t i = 0; i < isotope_names.size(); ++i) {
        const auto& isotope = isotope_xs_data.at(isotope_names[i]);
        std::vector<double> integrated = integrate_isotope_xs(isotope);
        
        for (size_t j = 0; j < total_xs.size(); ++j) {
            total_xs[j] += integrated[j] * fractions[i];
        }
    }

    if (!response_kernel.empty()) {
        total_xs = convolve_with_kernel(total_xs);
    }

    return total_xs;
}

std::vector<double> CrossSectionCalculator::integrate_isotope_xs(
    const IsotopeData& isotope_data) const {
    int num_bins_to_add = response_kernel.empty() ? 0 : response_kernel.size() - 1;

    std::vector<double> extended_grid = energy_grid;
    for (int i = 0; i < num_bins_to_add; ++i) {
        double new_prefix = extended_grid.front() * 
                          std::pow(extended_grid.front() / extended_grid[1], 1);
        extended_grid.insert(extended_grid.begin(), new_prefix);
        
        double new_suffix = extended_grid.back() * 
                          std::pow(extended_grid.back() / extended_grid[extended_grid.size() - 2], 1);
        extended_grid.push_back(new_suffix);
    }

    std::vector<double> integrated_values;
    for (size_t i = 0; i < extended_grid.size() - 1; ++i) {
        double emin = extended_grid[i];
        double emax = extended_grid[i + 1];

        if (emin <= 0 || emax <= 0) {
            integrated_values.push_back(0.0);
            continue;
        }

        double integral = 0.5 * (
            linear_interp(isotope_data.energies, isotope_data.values, emin) + 
            linear_interp(isotope_data.energies, isotope_data.values, emax)
        ) * (emax - emin);

        double avg_xs = integral / (emax - emin);
        integrated_values.push_back(avg_xs);
    }

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

    if (integrated_values.size() != expected_size) {
        integrated_values.resize(expected_size, integrated_values.back());
    }

    std::replace_if(integrated_values.begin(), integrated_values.end(),
                   [](double val) { return std::isnan(val); }, 0.0);

    return integrated_values;
}