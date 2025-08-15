# Command-Line Interface

Delta-Audit provides a comprehensive command-line interface for running experiments and generating figures.

## Overview

```bash
delta-audit [command] [options]
```

## Commands

### quickstart

Run a quick demonstration with a subset of experiments.

```bash
delta-audit quickstart [--output-dir RESULTS_DIR]
```

**Options:**
- `--output-dir`: Output directory for results (default: "results")

**Example:**
```bash
# Run quickstart with default settings
delta-audit quickstart

# Run quickstart with custom output directory
delta-audit quickstart --output-dir my_results
```

**What it does:**
- Trains a pair of logistic regression models on the wine dataset
- Computes Δ-Attribution metrics
- Saves results to `quickstart_results.csv`
- Prints key metrics (BAC and DCE)

### run

Run the full benchmark across all datasets, algorithms, and pairs.

```bash
delta-audit run --config CONFIG_FILE [--output-dir RESULTS_DIR]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--output-dir`: Output directory for results (default: "results")

**Example:**
```bash
# Run full benchmark
delta-audit run --config configs/full_benchmark.yaml

# Run with custom output directory
delta-audit run --config configs/full_benchmark.yaml --output-dir my_results
```

**What it does:**
- Runs all 45 experiments (5 algorithms × 3 pairs × 3 datasets)
- Computes comprehensive Δ-Attribution metrics
- Saves detailed results to `delta_summary.csv`
- Saves standard metrics to `standard_summary.csv`

### figures

Generate overview figures from results.

```bash
delta-audit figures --summary SUMMARY_DIR --out OUTPUT_DIR
```

**Options:**
- `--summary`: Path to summary results directory (required)
- `--out`: Output directory for figures (required)

**Example:**
```bash
# Generate figures from results
delta-audit figures --summary results/_summary --out figures/

# Generate figures with custom paths
delta-audit figures --summary my_results --out my_figures
```

**What it creates:**
- BAC vs DCE scatter plot (`fig1_bac_vs_dce.png`)
- Algorithm comparison bar charts (`fig2_bars_bac_by_algo.png`, etc.)
- Dataset heatmaps (`fig_dataset_heatmap_BAC.png`, etc.)
- Performance impact analysis (`fig_performance_impact.png`)
- Comprehensive overview figure (`fig0_overview.png`)

### check

Run sanity checks on results.

```bash
delta-audit check [--summary SUMMARY_DIR]
```

**Options:**
- `--summary`: Path to summary results directory (default: "results/_summary")

**Example:**
```bash
# Run checks with default path
delta-audit check

# Run checks with custom path
delta-audit check --summary my_results
```

**What it displays:**
- Total number of experiments
- Datasets, algorithms, and pairs covered
- Top performing algorithms
- Performance impact summary
- Key metric ranges

## Configuration Files

Delta-Audit uses YAML configuration files to define experiments.

### Quickstart Configuration

```yaml
# configs/quickstart.yaml
datasets:
  - wine

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

### Full Benchmark Configuration

```yaml
# configs/full_benchmark.yaml
datasets:
  - breast_cancer
  - wine
  - digits

algo_pairs:
  logreg:
    - A: {C: 1.0, penalty: l2, solver: lbfgs}
      B: {C: 0.1, penalty: l2, solver: lbfgs}
      pair_name: pair1
    - A: {C: 1.0, penalty: l2, solver: liblinear}
      B: {C: 1.0, penalty: l1, solver: liblinear}
      pair_name: pair2
    - A: {C: 1.0, penalty: l2, solver: lbfgs}
      B: {C: 1.0, penalty: l2, solver: saga}
      pair_name: pair3
  # ... more algorithms

experiment_settings:
  test_size: 0.2
  random_state: 42
  baseline_method: mean
```

## Output Structure

### Results Directory

```
results/
├── delta_summary.csv          # All Δ-Attribution metrics
├── standard_summary.csv       # Standard performance metrics
├── quickstart_results.csv     # Quickstart results (if run)
└── _summary/                  # Summary directory for figures
    ├── delta_summary.csv
    └── standard_summary.csv
```

### Figures Directory

```
figures/
├── fig0_overview.png          # Comprehensive overview
├── fig1_bac_vs_dce.png        # BAC vs DCE scatter
├── fig2_bars_bac_by_algo.png  # BAC by algorithm
├── fig3_bars_dce_by_algo.png  # DCE by algorithm
├── fig4_bars_deltamag_by_algo.png  # Δ Magnitude by algorithm
├── fig_dataset_heatmap_BAC.png     # BAC heatmap
├── fig_dataset_heatmap_DCE.png     # DCE heatmap
├── fig_dataset_heatmap_DeltaMag_L1.png  # Δ Magnitude heatmap
└── fig_performance_impact.png      # Performance impact
```

## Error Handling

The CLI includes comprehensive error handling:

- **Missing files**: Clear error messages for missing configuration or results files
- **Invalid arguments**: Helpful error messages for incorrect command usage
- **Processing errors**: Continues processing other experiments if one fails
- **Graceful exit**: Handles keyboard interrupts gracefully

## Examples

### Complete Workflow

```bash
# 1. Run quickstart to test installation
delta-audit quickstart

# 2. Run full benchmark
delta-audit run --config configs/full_benchmark.yaml

# 3. Generate figures
delta-audit figures --summary results/_summary --out figures/

# 4. Check results
delta-audit check
```

### Custom Analysis

```bash
# Create custom configuration
cat > my_config.yaml << EOF
datasets:
  - wine
  - digits

algo_pairs:
  svc:
    - A: {C: 1.0, kernel: rbf, gamma: scale}
      B: {C: 1.0, kernel: linear, gamma: scale}
      pair_name: kernel_change

experiment_settings:
  test_size: 0.3
  random_state: 123
  baseline_method: mean
EOF

# Run custom experiments
delta-audit run --config my_config.yaml --output-dir custom_results

# Generate figures
delta-audit figures --summary custom_results --out custom_figures
```

## Troubleshooting

### Common Issues

1. **"Command not found"**: Make sure Delta-Audit is installed and in your PATH
2. **"No module named 'delta_audit'"**: Install the package with `pip install -e .`
3. **"File not found"**: Check that configuration and results files exist
4. **Memory errors**: The full benchmark requires ~2GB RAM

### Getting Help

```bash
# Show help for all commands
delta-audit --help

# Show help for specific command
delta-audit quickstart --help
delta-audit run --help
delta-audit figures --help
delta-audit check --help
``` 