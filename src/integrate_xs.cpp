#include "integrate_xs.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>



void CrossSectionCalculator::initialize(const std::vector<double>& grid,
                                        double flight_path_length,
                                        double tstep,
                                        double tau0,
                                        double tau1,
                                        double tau2,
                                        double sigma0,
                                        double sigma1,
                                        double sigma2,
                                        double x0) {
    energy_grid = grid;
    flight_path = flight_path_length;
    default_tstep = tstep;  // This line ensures the default value is set
    // default_tau = tau;
    // default_sigma = sigma;
    // default_x0 = x0;
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
    double t0, double L0, double tau0, double tau1, double tau2,
                          double sigma0, double sigma1, double sigma2,
                          double x0) const {
    
    
    std::vector<double> response = calculate_response(t0, L0, tau0, tau1, tau2,
                                                             sigma0, sigma1, sigma2, x0);
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

    // std::cout << "Energy grid size: " << energy_grid.size() 
    //       << ", Kernel size: " << kernel.size() << "\n";

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
    double t0, double L0,double tau0,double tau1,double tau2,
                         double sigma0,double sigma1,double sigma2,
                         double x0) const {
    const int nbins = 300;
    // const double tstep = 1.5625e-9;
    // std::cout<<"tstep "<<default_tstep<<std::endl;
    const double tstep = default_tstep;
    double energy = 1.0;


    
    // Create time grid from -300 to 300
    std::vector<double> tgrid;
    tgrid.reserve(2 * nbins + 1);
    for (int i = -nbins; i <= nbins; i++) {
        tgrid.push_back(i * tstep);
    }
    
    // Create exponential term (only for x > 0)
    std::vector<double> exponential;
    exponential.reserve(tgrid.size());
    double tau = tau0 + tau1 * energy + tau2 * std::log(energy);
    
    for (double t : tgrid) {
        double x = (t - x0) / tau;
        double exp_val = (x > 0) ? std::exp(-x) : 0.0;
        exponential.push_back(exp_val);
    }
    
    // Create Gaussian term
    std::vector<double> gaussian;
    gaussian.reserve(tgrid.size());

    double sigma = sigma0 + sigma1 * energy + sigma2 * std::log(energy);

    double gauss_norm = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
    
    for (double t : tgrid) {
        double x = (t - x0) / sigma;
        gaussian.push_back(gauss_norm * std::exp(-0.5 * x * x));
    }
    
    // Perform convolution with correct window sliding
    std::vector<double> response(tgrid.size(), 0.0);
    int half_size = static_cast<int>(tgrid.size()) / 2;
    
    for (int i = 0; i < static_cast<int>(tgrid.size()); i++) {
        double sum = 0.0;
        for (int j = 0; j < static_cast<int>(tgrid.size()); j++) {
            // Shift j to center the convolution window
            int shifted_j = j - half_size;
            int k = i - shifted_j;
            
            if (k >= 0 && k < static_cast<int>(tgrid.size())) {
                sum += gaussian[j] * exponential[k];
            }
        }
        response[i] = sum * tstep;
    }
    
    // Normalize the convolution result
    double sum = 0.0;
    for (double val : response) {
        sum += val;
    }
    
    if (sum > 0.0) {
        for (double& val : response) {
            val /= sum;
        }
    }
    
    // Cut array symmetrically based on threshold
    const double eps = 1.0e-6;  // Threshold from Python implementation
    int center = response.size() / 2;
    int left_idx = center;
    int right_idx = center;
    
    // Find symmetric bounds
    while (left_idx > 0 && right_idx < static_cast<int>(response.size()) - 1) {
        if (response[left_idx-1] < eps && response[right_idx+1] < eps) {
            break;
        }
        if (response[left_idx-1] >= eps || response[right_idx+1] >= eps) {
            left_idx--;
            right_idx++;
        }
    }
    
    // Ensure odd number of elements
    if ((right_idx - left_idx + 1) % 2 == 0) {
        right_idx++;
    }
    
    // Extract the final response
    std::vector<double> final_response(
        response.begin() + left_idx,
        response.begin() + right_idx + 1
    );
    
    return final_response;

}

std::vector<double> CrossSectionCalculator::get_response(
    double t0, double L0, double tau0,double tau1,double tau2,
                         double sigma0,double sigma1,double sigma2,
                         double x0) const {

    return calculate_response(t0, L0, tau0, tau1, tau2,
                                    sigma0, sigma1, sigma2,
                                    x0);
}
