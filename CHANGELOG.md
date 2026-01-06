# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-06

### Added
- **Grouped Data Support**: Fit imaging detector data and multi-sample measurements with parallel processing
- **Advanced Rebinning**: Combine time-of-flight bins with automatic energy calibration preservation
- **Save/Load Functionality**: Serialize models and fit results to JSON for reproducible analysis
- **Enhanced Visualization**: Transmission maps, multi-parameter plots, and interactive HTML reports
- **Memory Management**: Efficient parallel fitting with automatic memory optimization
- GroupedFitResult class for managing and visualizing fit results from grouped data
- `from_grouped_arrays` method for creating Data objects from 2D grouped transmission data
- Support for 1D and 2D index patterns in grouped data loading
- Transmission map plotting capabilities for imaging data
- Multi-parameter comparison plots for grouped fits
- HTML report generation for grouped fit results

### Changed
- Updated Development Status classifier from "Planning" to "Beta"
- Improved build configuration with MANIFEST.in for proper package distribution
- Fixed C++ extension build issues with proper include path resolution
- Pinned setuptools version to avoid metadata compatibility issues

### Fixed
- Pre-commit hook failures (ruff linting, mypy type checking)
- Import organization and unused import warnings
- Exception handling to follow EM101/EM102 rules
- Build system configuration for PyPI publishing

### Dependencies
- Added type stub dependencies for better IDE support (pandas-stubs, types-requests, types-tqdm)
- Constrained setuptools version to >=68,<75 for compatibility

## [0.3.0] - 2025-10-28

### Added
- Rietveld fitting option added to the fit method
- TOF (Time-of-Flight) calibration options
- Enhanced correction parameters in TransmissionModel
- Cross section tests for improved reliability
- Materials database with detailed materials support

### Changed
- **BREAKING**: Removed total cross section method
- Upgraded Rietveld method to use accumulative parameter refinement
- Improved interpolation function implementation
- Updated tutorial documentation with latest features
- Enhanced material handling - removed material name restrictions

### Fixed
- Fixed interpolation issues that affected cross section calculations
- Corrected mul method implementation
- Various bug fixes and code cleanup

### Dependencies
- Added matplotlib as explicit dependency
- Specified C++17 requirement in setup.py
- Updated pybind11 integration

## [0.2.2] - 2024-12-28

### Changed
- Minor version bump with bug fixes
- Improved cross section interpolation

## Previous Versions

See git history for details on versions prior to 0.2.2.

---

[0.4.0]: https://github.com/TsvikiHirsh/nres/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/TsvikiHirsh/nres/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/TsvikiHirsh/nres/releases/tag/v0.2.2
