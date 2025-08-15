# API Reference

This page documents the Python API for Delta-Audit.

## Core Modules

### delta_audit.metrics

Core Δ-Attribution metrics computation.

#### `compute_all_metrics(phi_A, phi_B, delta_phi, delta_f, top_m_features=None, A_correct=None, B_correct=None)`

Compute the complete suite of Δ-Attribution metrics.

**Parameters:**
- `phi_A` (np.ndarray): Attributions for model A (n_samples, n_features)
- `phi_B` (np.ndarray): Attributions for model B (n_samples, n_features)
- `delta_phi` (np.ndarray): Difference phi_B - phi_A (n_samples, n_features)
- `delta_f` (np.ndarray): Difference f_B - f_A (n_samples,)
- `top_m_features` (List[int], optional): Indices of relevant features for COΔF
- `A_correct` (np.ndarray, optional): Boolean mask of correct predictions for A
- `B_correct` (np.ndarray, optional): Boolean mask of correct predictions for B

**Returns:**
- `Dict[str, float]`: Dictionary of metric name to value

#### `compute_bac(delta_phi, delta_f)`

Compute Behavioral Alignment Coefficient.

**Parameters:**
- `delta_phi` (np.ndarray): Attribution differences
- `delta_f` (np.ndarray): Output differences

**Returns:**
- `float`: BAC value

#### `compute_dce(delta_phi, delta_f)`

Compute Differential Conservation Error.

**Parameters:**
- `delta_phi` (np.ndarray): Attribution differences
- `delta_f` (np.ndarray): Output differences

**Returns:**
- `float`: DCE value

### delta_audit.explainers

Attribution computation methods.

#### `compute_occlusion_attributions(model, X, baseline)`

Compute occlusion-based attributions.

**Parameters:**
- `model` (Pipeline): Fitted scikit-learn pipeline
- `X` (np.ndarray): Input features (n_samples, n_features)
- `baseline` (np.ndarray): Baseline values (n_features,)

**Returns:**
- `np.ndarray`: Attributions (n_samples, n_features)

#### `compute_mean_baseline(X)`

Compute mean baseline for attribution methods.

**Parameters:**
- `X` (np.ndarray): Input features

**Returns:**
- `np.ndarray`: Mean baseline (n_features,)

### delta_audit.runners

Training and evaluation pipelines.

#### `run_benchmark(config_path, output_dir="results")`

Run the full benchmark across all datasets, algorithms, and pairs.

**Parameters:**
- `config_path` (str): Path to configuration file
- `output_dir` (str): Directory to save results

**Returns:**
- `None`

#### `run_quickstart(output_dir="results")`

Run a quick demonstration with a subset of experiments.

**Parameters:**
- `output_dir` (str): Directory to save results

**Returns:**
- `None`

#### `train_model_pair(X_train, y_train, algo, config_A, config_B)`

Train a pair of models with different configurations.

**Parameters:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training labels
- `algo` (str): Algorithm name
- `config_A` (Dict): Configuration for model A
- `config_B` (Dict): Configuration for model B

**Returns:**
- `Tuple[Pipeline, Pipeline]`: (model_A, model_B)

### delta_audit.plotting

Figure generation utilities.

#### `make_overview_figure(summary_dir, output_dir)`

Generate comprehensive overview figure from results.

**Parameters:**
- `summary_dir` (str): Directory containing summary CSV files
- `output_dir` (str): Directory to save generated figures

**Returns:**
- `None`

#### `plot_bac_vs_dce(delta_results, save_path=None)`

Plot BAC vs DCE scatter plot.

**Parameters:**
- `delta_results` (pd.DataFrame): DataFrame with BAC and DCE columns
- `save_path` (str, optional): Path to save the figure

**Returns:**
- `None`

### delta_audit.io

Data loading and saving utilities.

#### `load_results(file_path)`

Load results from a CSV file.

**Parameters:**
- `file_path` (Path): Path to the CSV file

**Returns:**
- `Optional[pd.DataFrame]`: DataFrame with results or None

#### `save_results(results, file_path)`

Save results to a CSV file.

**Parameters:**
- `results` (pd.DataFrame): DataFrame to save
- `file_path` (Path): Path where to save the file

**Returns:**
- `None`

#### `load_config(config_path)`

Load configuration from a YAML file.

**Parameters:**
- `config_path` (Path): Path to the configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

## Example Usage

```python
import numpy as np
import pandas as pd
from delta_audit import (
    compute_all_metrics,
    compute_occlusion_attributions,
    compute_mean_baseline,
    train_model_pair,
    run_quickstart
)

# Run quickstart
run_quickstart()

# Or run custom analysis
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load data
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model pair
config_A = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
config_B = {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
model_A, model_B = train_model_pair(X_train, y_train, 'logreg', config_A, config_B)

# Compute attributions
baseline = compute_mean_baseline(X_train)
phi_A = compute_occlusion_attributions(model_A, X_test, baseline)
phi_B = compute_occlusion_attributions(model_B, X_test, baseline)

# Compute metrics
delta_phi = phi_B - phi_A
delta_f = model_B.predict_proba(X_test).max(axis=1) - model_A.predict_proba(X_test).max(axis=1)

metrics = compute_all_metrics(phi_A, phi_B, delta_phi, delta_f)
print(f"BAC: {metrics['bac']:.3f}")
print(f"DCE: {metrics['dce']:.3f}")
```

## Configuration Format

Delta-Audit uses YAML configuration files:

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

## Data Formats

### Input Data
- Features should be numerical arrays
- Labels should be integer arrays
- All data is automatically scaled using StandardScaler

### Output Results
- CSV files with one row per experiment
- Columns include all metrics plus metadata (dataset, algorithm, pair)
- Figures saved as PNG files with high DPI

## Error Handling

Most functions include error handling and will:
- Print informative error messages
- Return None or empty results on failure
- Continue processing other experiments if one fails 