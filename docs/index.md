# Delta-Audit Documentation

Welcome to the Delta-Audit documentation! Delta-Audit is a lightweight Δ-Attribution suite for auditing model updates (A→B) with behavioural linkage and robustness checks.

## What is Δ-Attribution?

Δ-Attribution (Delta-Attribution) is a framework for understanding how model explanations change when models are updated. When you train a new version of a model, not only do the predictions change, but also the explanations of those predictions. Delta-Audit provides metrics to quantify and analyze these changes.

## Key Concepts

- **Model Pair (A→B)**: Two models where B is an updated version of A
- **Attribution Change (Δφ)**: Difference in feature attributions between models A and B
- **Behavioral Alignment**: How well attribution changes correlate with output changes
- **Conservation Error**: How much the sum of attribution changes differs from actual output changes

## Quick Start

```bash
# Install Delta-Audit
pip install delta-audit

# Run a quick demonstration
delta-audit quickstart

# Run the full benchmark
delta-audit run --config configs/full_benchmark.yaml
```

## Core Metrics

Delta-Audit implements several key metrics:

- **BAC (Behavioral Alignment Coefficient)**: Measures correlation between attribution change magnitude and output change magnitude
- **DCE (Differential Conservation Error)**: Measures the difference between sum of attribution changes and actual output change
- **Δ Magnitude L1**: L1 norm of attribution differences
- **Rank Overlap @10**: Overlap between top-10 features of two attribution sets
- **JSD (Jensen-Shannon Divergence)**: Distributional shift between attribution sets

## Supported Algorithms

- Logistic Regression
- Support Vector Classification
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

## Supported Datasets

- Breast Cancer (binary classification)
- Wine (multi-class classification)
- Digits (multi-class classification)

## Documentation Sections

- [Getting Started](getting-started.md) - Installation and first steps
- [Concepts](concepts.md) - Detailed explanation of Δ-Attribution concepts
- [Metrics](metrics.md) - Complete description of all metrics
- [API Reference](api.md) - Python API documentation
- [CLI Reference](cli.md) - Command-line interface guide
- [Benchmarks](benchmarks.md) - Reproducing the paper results
- [FAQ](faq.md) - Frequently asked questions

## Citation

If you use Delta-Audit in your research, please cite:

```bibtex
@article{hemmat2025delta,
  title={Delta-Audit: Explaining What Changes When Models Change},
  author={Hemmat, Arshia},
  journal={arXiv preprint},
  year={2025}
}
```

## Getting Help

- Check the [FAQ](faq.md) for common issues
- Open an issue on [GitHub](https://github.com/arshiahemmat/delta-audit)
- Review the [API documentation](api.md) for detailed function descriptions 