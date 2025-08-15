# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-15

### Added
- Initial release of Delta-Audit
- Complete Δ-Attribution metrics suite implementation
- Support for 5 algorithms (Logistic Regression, SVM, Random Forest, Gradient Boosting, KNN)
- Support for 3 datasets (Breast Cancer, Wine, Digits)
- Command-line interface with quickstart, run, figures, and check commands
- Comprehensive plotting utilities
- YAML-based configuration system
- GitHub Actions CI/CD pipeline
- Documentation website
- Original experiment scripts for reproducibility

### Features
- **BAC (Behavioral Alignment Coefficient)**: Correlation between attribution change magnitude and output change magnitude
- **DCE (Differential Conservation Error)**: Difference between sum of attribution changes and actual output change
- **Δ Magnitude L1**: L1 norm of attribution differences
- **Δ TopK10**: Fraction of total magnitude captured by top-10 features
- **Δ Entropy**: Entropy of normalized attribution differences
- **Rank Overlap @10**: Overlap between top-10 features of two attribution sets
- **JSD (Jensen-Shannon Divergence)**: Distributional shift between attribution sets
- **COΔF**: Conservation of relevant features for fixes and regressions
- **Δ Stability**: Robustness to input perturbations

### Technical Details
- Python 3.9+ compatibility
- Comprehensive test suite
- Type hints throughout codebase
- Modular architecture with separate modules for metrics, explainers, runners, plotting, and I/O
- MIT license 