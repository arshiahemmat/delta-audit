# Getting Started with Delta-Audit

This guide will help you get up and running with Delta-Audit quickly.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/arshiahemmat/delta-audit.git
cd delta-audit

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
pip install -r requirements.txt
```

### Install Dependencies

The main dependencies are:
- numpy >= 1.20
- pandas >= 1.2
- scikit-learn >= 1.0
- matplotlib >= 3.4
- scipy >= 1.7
- seaborn >= 0.11

## Quick Start

### 1. Run a Quick Demonstration

```bash
# This runs a single experiment (wine dataset, logistic regression)
delta-audit quickstart
```

This will:
- Train a pair of logistic regression models on the wine dataset
- Compute Δ-Attribution metrics
- Save results to `results/quickstart_results.csv`
- Print key metrics (BAC and DCE)

### 2. Run the Full Benchmark

```bash
# This runs all 45 experiments (5 algorithms × 3 pairs × 3 datasets)
delta-audit run --config configs/full_benchmark.yaml
```

This will:
- Train model pairs across all algorithms and datasets
- Compute comprehensive Δ-Attribution metrics
- Save detailed results to `results/delta_summary.csv`
- Save standard metrics to `results/standard_summary.csv`

### 3. Generate Figures

```bash
# Generate overview figures from results
delta-audit figures --summary results/_summary --out figures/
```

This will create:
- BAC vs DCE scatter plot
- Algorithm comparison bar charts
- Dataset heatmaps
- Performance impact analysis
- Comprehensive overview figure

### 4. Check Results

```bash
# Run sanity checks and print key statistics
delta-audit check
```

This will display:
- Total number of experiments
- Top performing algorithms
- Performance impact summary
- Key metric ranges

## Understanding the Output

### Results Files

- `delta_summary.csv`: Contains all Δ-Attribution metrics
- `standard_summary.csv`: Contains standard performance metrics
- `figures/`: Directory with generated plots

### Key Metrics to Look For

- **BAC (Behavioral Alignment Coefficient)**: Higher values indicate better alignment between attribution changes and output changes
- **DCE (Differential Conservation Error)**: Lower values indicate better conservation of attribution sums
- **Δ Magnitude L1**: Measures the overall magnitude of attribution changes
- **Rank Overlap @10**: Measures how much the top-10 features overlap between models

## Configuration

Delta-Audit uses YAML configuration files to define experiments:

```yaml
datasets:
  - breast_cancer
  - wine
  - digits

algo_pairs:
  logreg:
    - A: {C: 1.0, penalty: l2, solver: lbfgs}
      B: {C: 0.1, penalty: l2, solver: lbfgs}
      pair_name: pair1

experiment_settings:
  test_size: 0.2
  random_state: 42
  baseline_method: mean
```

## Next Steps

- Read the [Concepts](concepts.md) guide to understand Δ-Attribution theory
- Check the [Metrics](metrics.md) documentation for detailed metric descriptions
- Review the [API Reference](api.md) for programmatic usage
- See [Benchmarks](benchmarks.md) for reproducing paper results

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the virtual environment
2. **Memory issues**: The full benchmark requires ~2GB RAM
3. **Slow execution**: The full benchmark takes 10-30 minutes depending on your machine

### Getting Help

- Check the [FAQ](faq.md) for common questions
- Open an issue on GitHub with error details
- Review the logs for specific error messages 