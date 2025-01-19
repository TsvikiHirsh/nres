#ifndef INTEGRATE_XS_H
#define INTEGRATE_XS_H

#include <vector>
#include <map>
#include <string>

class CrossSectionCalculator {
public:
    // Move struct to public section
    struct IsotopeData {
        std::vector<double> values;
    };

    CrossSectionCalculator() = default;
    
    void initialize(const std::vector<double>& grid, 
                   double flight_path_length = 10.59,
                   double tstep = 1.5625e-9,                   
                   double tau = 1.0,
                   double sigma = 1.0,
                   double x0 = 0.0);
    
    void add_xs_data(const std::vector<double>& energies,
                     const std::map<std::string, std::vector<double>>& xs_data);
    
    std::vector<double> calculate_xs(const std::vector<double>& user_energy_grid,
                                   const std::map<std::string, double>& fractions,
                                   double t0 = 0.0,
                                   double L0 = 1.0,
                                   double tau = 1.0,
                                   double sigma = 1.0,
                                   double x0 = 0.0) const;
    
    std::vector<std::string> get_isotope_names() const { return isotope_names; }
    std::vector<double> get_energy_grid() const { return energy_grid; }
    double get_flight_path() const { return flight_path; }
    std::vector<double> get_response(double t0, double L0, double tau = 1.0, double sigma = 1.0,double x0 = 0.0) const;

private:    
    std::map<std::string, IsotopeData> isotope_xs_data;
    std::vector<double> energy_grid;
    std::vector<double> shared_energies;
    std::vector<std::string> isotope_names;
    double flight_path;
    double default_sigma;
    double default_tau;
    double default_x0;
    double tstep; 

    double linear_interp(const std::vector<double>& xs_energies, 
                        const std::vector<double>& xs_values, 
                        double energy) const;
    
    std::vector<double> calculate_response(double t0, double L0, 
                                         double tau, double sigma, double x0) const;
    
    std::vector<double> convolve_with_kernel(const std::vector<double>& values,
                                           const std::vector<double>& kernel) const;
    
    std::vector<double> integrate_isotope_xs(const IsotopeData& isotope_data,
                                           const std::vector<double>& kernel,
                                           const std::vector<double>& user_energy_grid) const;
    

};

#endif // INTEGRATE_XS_H
