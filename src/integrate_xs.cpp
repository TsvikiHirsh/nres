#include "integrate_xs.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <map>
#include <complex>
#include <fftw3.h>
#include <omp.h>

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
    default_tstep = tstep;
}

void CrossSectionCalculator::add_xs_data(
    const std::vector<double>& energies,
    const std::map<std::string, std::vector<double>>& xs_data) {
    
    shared_energies = energies;
    isotope_names.clear();
    
    for (const auto& pair : xs_data) {
        isotope_xs_data[pair.first] = {pair.second};
        isotope_names.push_back(pair.first);
    }
}

std::vector<double> CrossSectionCalculator::calculate_xs(
    const std::vector<double>& user_energy_grid,
    const std::map<std::string, double>& fractions,
    double t0, double L0,
    double tau0, double tau1, double tau2,
    double sigma0, double sigma1, double sigma2,
    double x0) const {
    
    std::vector<double> total_xs(user_energy_grid.size(), 0.0);
    
    #pragma omp parallel for
    for (size_t i = 0; i < user_energy_grid.size(); ++i) {
        double current_energy = user_energy_grid[i];
        std::vector<double> response = calculate_response(
            t0, L0, tau0, tau1, tau2, sigma0, sigma1, sigma2, x0, current_energy
        );
        
        for (const auto& isotope_name : isotope_names) {
            const auto& isotope = isotope_xs_data.at(isotope_name);
            std::vector<double> integrated = integrate_isotope_xs(isotope, response, {current_energy});
            total_xs[i] += integrated[0] * fractions.at(isotope_name);
        }
    }
    return total_xs;
}

std::vector<double> CrossSectionCalculator::integrate_isotope_xs(
    const IsotopeData& isotope_data,
    const std::vector<double>& kernel,
    const std::vector<double>& user_energy_grid) const {
    
    const std::vector<double>& grid_to_use = user_energy_grid.empty() ? energy_grid : user_energy_grid;
    size_t num_bins_to_add = kernel.empty() ? 0 : kernel.size() - 1;
    
    std::vector<double> extended_grid(grid_to_use);
    double front = extended_grid.front();
    double back = extended_grid.back();
    
    for (size_t i = 0; i < num_bins_to_add; ++i) {
        extended_grid.insert(extended_grid.begin(), front * 0.9);
        extended_grid.push_back(back * 1.1);
    }
    
    std::vector<double> integrated_values;
    for (size_t i = 0; i < extended_grid.size() - 1; ++i) {
        double emin = extended_grid[i];
        double emax = extended_grid[i + 1];
        
        double integral = 0.5 * (
            linear_interp(shared_energies, isotope_data.values, emin) + 
            linear_interp(shared_energies, isotope_data.values, emax)
        ) * (emax - emin);
        
        integrated_values.push_back(integral / (emax - emin));
    }
    
    if (!kernel.empty()) {
        integrated_values = convolve_with_kernel(integrated_values, kernel);
    }
    return integrated_values;
}

std::vector<double> CrossSectionCalculator::convolve_with_kernel(
    const std::vector<double>& values,
    const std::vector<double>& kernel) const {
    
    int N = values.size();
    fftw_complex *in, *out;
    fftw_plan p;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    for (int i = 0; i < N; i++) {
        in[i][0] = values[i];
        in[i][1] = 0.0;
    }
    
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    
    std::vector<double> result(N);
    for (int i = 0; i < N; i++) {
        result[i] = out[i][0] / N;
    }
    
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return result;
}

size_t binary_search(const std::vector<double>& arr, double value) {
    auto it = std::lower_bound(arr.begin(), arr.end(), value);
    return std::distance(arr.begin(), it);
}

double CrossSectionCalculator::linear_interp(
    const std::vector<double>& xs_energies,
    const std::vector<double>& xs_values,
    double energy) const {
    
    if (energy <= xs_energies.front()) return xs_values.front();
    if (energy >= xs_energies.back()) return xs_values.back();
    
    size_t i = binary_search(xs_energies, energy);
    if (i == 0) return xs_values[0];
    if (i >= xs_energies.size()) return xs_values.back();
    
    double e0 = xs_energies[i - 1], e1 = xs_energies[i];
    double v0 = xs_values[i - 1], v1 = xs_values[i];
    
    return v0 + (v1 - v0) * (energy - e0) / (e1 - e0);
}

std::vector<double> CrossSectionCalculator::calculate_response(
    double t0, double L0, double tau0, double tau1, double tau2,
    double sigma0, double sigma1, double sigma2, double x0, double energy) const {
    
    const int nbins = 300;
    const double tstep = default_tstep;
    
    std::vector<double> tgrid;
    tgrid.reserve(2 * nbins + 1);
    for (int i = -nbins; i <= nbins; i++) {
        tgrid.push_back(i * tstep);
    }
    
    std::vector<double> exponential;
    exponential.reserve(tgrid.size());
    double tau = tau0 + tau1 * energy + tau2 * std::log(energy);
    
    for (double t : tgrid) {
        double x = (t - x0) / tau;
        double exp_val = (x > 0) ? std::exp(-x) : 0.0;
        exponential.push_back(exp_val);
    }
    
    std::vector<double> gaussian;
    gaussian.reserve(tgrid.size());

    double sigma = sigma0 + sigma1 * energy + sigma2 * std::log(energy);
    double gauss_norm = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
    
    for (double t : tgrid) {
        double x = (t - x0) / sigma;
        gaussian.push_back(gauss_norm * std::exp(-0.5 * x * x));
    }
    
    std::vector<double> response(tgrid.size(), 0.0);
    int half_size = static_cast<int>(tgrid.size()) / 2;
    
    for (int i = 0; i < static_cast<int>(tgrid.size()); i++) {
        double sum = 0.0;
        for (int j = 0; j < static_cast<int>(tgrid.size()); j++) {
            int shifted_j = j - half_size;
            int k = i - shifted_j;
            
            if (k >= 0 && k < static_cast<int>(tgrid.size())) {
                sum += gaussian[j] * exponential[k];
            }
        }
        response[i] = sum * tstep;
    }
    
    double sum = 0.0;
    for (double val : response) {
        sum += val;
    }
    
    if (sum > 0.0) {
        for (double& val : response) {
            val /= sum;
        }
    }
    
    const double eps = 1.0e-6;
    int center = response.size() / 2;
    int left_idx = center;
    int right_idx = center;
    
    while (left_idx > 0 && right_idx < static_cast<int>(response.size()) - 1) {
        if (response[left_idx-1] < eps && response[right_idx+1] < eps) {
            break;
        }
        if (response[left_idx-1] >= eps || response[right_idx+1] >= eps) {
            left_idx--;
            right_idx++;
        }
    }
    
    if ((right_idx - left_idx + 1) % 2 == 0) {
        right_idx++;
    }
    
    std::vector<double> final_response(
        response.begin() + left_idx,
        response.begin() + right_idx + 1
    );
    
    return final_response;
}

std::vector<double> CrossSectionCalculator::get_response(
    double t0, double L0, double tau0, double tau1, double tau2,
    double sigma0, double sigma1, double sigma2, double x0) const {

    return calculate_response(t0, L0, tau0, tau1, tau2,
                              sigma0, sigma1, sigma2, x0, 1.0);
}
