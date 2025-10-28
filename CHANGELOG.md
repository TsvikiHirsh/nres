# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.3.0]: https://github.com/TsvikiHirsh/nres/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/TsvikiHirsh/nres/releases/tag/v0.2.2
