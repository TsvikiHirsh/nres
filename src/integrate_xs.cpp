#include "integrate_xs.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>



void CrossSectionCalculator::initialize(const std::vector<double>& grid,
                                        double flight_path_length,
                                        double K,
                                        double tau,
                                        double x0,
                                        double tstep) {
                                       
    energy_grid = grid;
    flight_path = flight_path_length;
    default_K = K;
    default_tau = tau;
    default_x0 = x0;
    this->tstep = tstep;
}


void CrossSectionCalculator::add_xs_data(
    const std::vector<double>& energies,
    const std::map<std::string, std::vector<double>>& xs_data) {
    
    shared_energies = energies;
    isotope_names.clear();
    
    // Iterate without structured binding
    for (const auto& pair : xs_data) {
        const std::string& isotope_name = pair.first;
        const std::vector<double>& xs_values = pair.second;
        
        if (xs_values.size() != energies.size()) {
            throw std::invalid_argument(
                "Cross-section data size must match energy size for isotope " + 
                isotope_name);
        }
        
        isotope_xs_data[isotope_name] = {xs_values};
        isotope_names.push_back(isotope_name);
    }
}

std::vector<double> CrossSectionCalculator::calculate_xs(
    const std::vector<double>& user_energy_grid,
    const std::map<std::string, double>& fractions,
    double t0, double L0, double K, double tau, double x0) const {
    
    K = (K != 1.0) ? K : default_K;
    tau = (tau != 1.0) ? tau : default_tau;
    x0 = (x0 != 0.0) ? x0 : default_x0;
    
    std::vector<double> response = calculate_response(t0, L0, K, tau, x0);
    std::vector<double> total_xs(user_energy_grid.size(), 0.0);
    
    for (const auto& isotope_name : isotope_names) {
        auto fraction_it = fractions.find(isotope_name);
        if (fraction_it == fractions.end()) {
            throw std::invalid_argument("Missing fraction for isotope: " + isotope_name);
        }
        
        const auto& isotope = isotope_xs_data.at(isotope_name);
        std::vector<double> integrated = integrate_isotope_xs(isotope, response, user_energy_grid);
        
        for (size_t j = 0; j < total_xs.size(); ++j) {
            total_xs[j] += integrated[j] * fraction_it->second;
        }
    }
    
    return total_xs;
}



std::vector<double> CrossSectionCalculator::integrate_isotope_xs(
    const IsotopeData& isotope_data,
    const std::vector<double>& kernel,
    const std::vector<double>& user_energy_grid = {}) const {
    
    const std::vector<double>& grid_to_use = user_energy_grid.empty() ? energy_grid : user_energy_grid;
    int num_bins_to_add = kernel.empty() ? 0 : kernel.size() - 1;

    std::vector<double> extended_grid = grid_to_use;
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
            linear_interp(shared_energies, isotope_data.values, emin) + 
            linear_interp(shared_energies, isotope_data.values, emax)
        ) * (emax - emin);

        double avg_xs = integral / (emax - emin);
        integrated_values.push_back(avg_xs);
    }

    size_t expected_size = grid_to_use.size();
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

    if (!kernel.empty()) {
        integrated_values = convolve_with_kernel(integrated_values, kernel);
    }

    std::replace_if(integrated_values.begin(), integrated_values.end(),
                   [](double val) { return std::isnan(val); }, 0.0);

    std::cout << "Energy grid size: " << energy_grid.size() 
          << ", Kernel size: " << kernel.size() << "\n";

    return integrated_values;
}


std::vector<double> CrossSectionCalculator::convolve_with_kernel(
    const std::vector<double>& values,
    const std::vector<double>& kernel) const {
    int values_size = values.size();
    int kernel_size = kernel.size();
    int pad_size = kernel_size / 2;

    std::vector<double> padded_values(values_size + 2 * pad_size, 0.0);
    std::copy(values.begin(), values.end(), padded_values.begin() + pad_size);

    std::vector<double> result(values_size, 0.0);
    for (int i = 0; i < values_size; ++i) {
        double conv_sum = 0.0;
        for (int j = 0; j < kernel_size; ++j) {
            conv_sum += padded_values[i + j] * kernel[j];
        }
        result[i] = conv_sum;
    }
    return result;
}

double CrossSectionCalculator::linear_interp(
    const std::vector<double>& xs_energies,
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



std::vector<double> CrossSectionCalculator::calculate_response(
    double t0, double L0, double K, double tau, double x0) const {
    const int nbins = 300;
    const double tstep = 1.5625e-9;  // Specified step size
    
    // Create time grid
    std::vector<double> tgrid;
    tgrid.reserve(2 * nbins + 1);
    for (int i = -nbins; i <= nbins; i++) {
        tgrid.push_back(x0 + i * tstep);
    }
    
    // First, create the Gaussian function
    // sigma = tau for the Gaussian part
    std::vector<double> gaussian;
    gaussian.reserve(tgrid.size());
    
    // Create exponential function
    // lambda = 1/(K*tau) for the exponential part
    std::vector<double> exponential;
    exponential.reserve(tgrid.size());
    
    double lambda = 1.0 / (K * tau);
    double sigma = tau;
    
    // Calculate both functions centered at x0
    double max_gauss = 0.0;
    double max_exp = 0.0;
    
    for (size_t i = 0; i < tgrid.size(); i++) {
        double t = tgrid[i] - x0;
        
        // Gaussian calculation in log space
        double gauss_log = -0.5 * (t * t) / (sigma * sigma);
        double gauss = std::exp(gauss_log);
        gaussian.push_back(gauss);
        max_gauss = std::max(max_gauss, gsuss);
        
        // Exponential calculation (only for t >= 0)
        double exp_val = (t >= 0) ? std::exp(-lambda * t) : 0.0;
        exponential.push_back(exp_val);
        max_exp = std::max(max_exp, exp_val);
    }
    
    // Normalize both functions
    for (double& g : gaussian) {
        g /= (max_gauss * std::sqrt(2.0 * M_PI) * sigma);
    }
    
    double exp_norm = lambda;  // Normalization factor for exponential
    for (double& e : exponential) {
        e *= exp_norm / max_exp;
    }
    
    // Perform the convolution
    std::vector<double> response(tgrid.size(), 0.0);
    int half_size = static_cast<int>(tgrid.size()) / 2;
    
    // Use the convolution theorem: conv(f,g) = IFFT(FFT(f) * FFT(g))
    // For our purposes, direct convolution is sufficient given the size
    for (int i = 0; i < static_cast<int>(tgrid.size()); i++) {
        double conv_val = 0.0;
        for (int j = std::max(0, i - half_size); 
             j < std::min(static_cast<int>(tgrid.size()), i + half_size + 1); j++) {
            int k = i - j;
            if (k >= 0 && k < static_cast<int>(tgrid.size())) {
                conv_val += gaussian[j] * exponential[k];
            }
        }
        response[i] = conv_val * tstep;  // Scale by time step for proper integration
    }
    
    // // Normalize the final response
    // double sum = 0.0;
    // for (double val : response) {
    //     sum += val;
    // }
    
    // if (sum > 0.0) {
    //     for (double& val : response) {
    //         val /= sum;
    //     }
    // }
    
    // // Find the center of mass
    // double weighted_sum = 0.0;
    // int max_idx = 0;
    // double max_val = 0.0;
    
    // for (size_t i = 0; i < response.size(); i++) {
    //     weighted_sum += response[i] * i;
    //     if (response[i] > max_val) {
    //         max_val = response[i];
    //         max_idx = i;
    //     }
    // }
    
    // // Center and trim as before
    // size_t center = response.size() / 2;
    // int shift = static_cast<int>(center) - max_idx;
    // if (shift != 0) {
    //     std::rotate(response.begin(), 
    //                response.begin() + (shift > 0 ? response.size() + shift : -shift), 
    //                response.end());
    // }
    
    // // Trim to significant values
    // const double eps = 1.0e-6;
    // int width = 0;
    
    // for (int i = 1; i <= static_cast<int>(center); i++) {
    //     if (center + i >= response.size() || center - i < 0 ||
    //         (response[center + i] < eps && response[center - i] < eps)) {
    //         width = i - 1;
    //         break;
    //     }
    // }
    
    // std::vector<double> final_response;
    // if (width > 0) {
    //     final_response.assign(
    //         response.begin() + (center - width),
    //         response.begin() + (center + width + 1)
    //     );
    // } else {
    //     final_response = {0.0, 1.0, 0.0};
    // }
    
    return response;
}

std::vector<double> CrossSectionCalculator::get_response(
    double t0, double L0, double K, double tau, double x0) const {
    K = (K != 1.0) ? K : default_K;
    tau = (tau != 1.0) ? tau : default_tau;
    x0 = (x0 != 0.0) ? x0 : default_x0;

    return calculate_response(t0, L0, K, tau, x0);
}
