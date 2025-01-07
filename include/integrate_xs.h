#ifndef INTEGRATE_XS_H
#define INTEGRATE_XS_H

#include <vector>
#include <map>
#include <string>

class CrossSectionCalculator {
public:
    CrossSectionCalculator() = default;
    
    void initialize(const std::vector<double>& grid, 
                   const std::vector<double>& kernel = std::vector<double>{0., 1., 0.});
    
    void add_isotope(const std::string& name,
                    const std::vector<double>& energies,
                    const std::vector<double>& xs_values);
    
    std::vector<double> calculate_xs(const std::vector<double>& fractions) const;
    
    // Getter methods for Python access
    std::vector<std::string> get_isotope_names() const { return isotope_names; }
    std::vector<double> get_energy_grid() const { return energy_grid; }
    std::vector<double> get_response_kernel() const { return response_kernel; }

private:
    struct IsotopeData {
        std::vector<double> energies;
        std::vector<double> values;
    };
    
    std::map<std::string, IsotopeData> isotope_xs_data;
    std::vector<double> energy_grid;
    std::vector<double> response_kernel;
    std::vector<std::string> isotope_names;

    double linear_interp(const std::vector<double>& xs_energies, 
                        const std::vector<double>& xs_values, 
                        double energy) const;
    
    std::vector<double> convolve_with_kernel(const std::vector<double>& values) const;
    
    std::vector<double> integrate_isotope_xs(const IsotopeData& isotope_data) const;
};

// For backward compatibility
std::vector<double> integrate_cross_section(
    const std::vector<double>& xs_energies,
    const std::vector<double>& xs_values,
    const std::vector<double>& energy_grid,
    const std::vector<double>& response = std::vector<double>{0., 1., 0.}
);

#endif // INTEGRATE_XS_H