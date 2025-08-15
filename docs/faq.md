# Frequently Asked Questions

This page addresses common questions and issues when using Delta-Audit.

## General Questions

### What is Δ-Attribution?

Δ-Attribution (Delta-Attribution) is a framework for understanding how model explanations change when models are updated. It provides metrics to quantify and analyze these changes, helping researchers and practitioners understand the impact of model updates on interpretability.

### Why should I use Delta-Audit?

Delta-Audit helps you:
- Understand how model explanations change when you update your models
- Quantify the behavioral alignment between attribution changes and output changes
- Identify which algorithms maintain better explanation consistency
- Audit model updates for interpretability changes

### What algorithms does Delta-Audit support?

Delta-Audit supports:
- Logistic Regression
- Support Vector Classification (SVM)
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

### What datasets are included?

Delta-Audit includes:
- Breast Cancer (binary classification)
- Wine (multi-class classification)
- Digits (multi-class classification)

## Installation and Setup

### How do I install Delta-Audit?

```bash
# Clone the repository
git clone https://github.com/arshiahemmat/delta-audit.git
cd delta-audit

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
pip install -r requirements.txt
```

### What are the system requirements?

- Python 3.9 or higher
- 4GB+ RAM (for full benchmark)
- 1GB+ free disk space
- Multi-core processor recommended

### I get "Command not found: delta-audit"

This usually means the package isn't installed properly. Try:
```bash
pip install -e .
```
Then verify installation:
```bash
python -c "import delta_audit; print('Installed successfully')"
```

## Usage Questions

### How long does the full benchmark take?

- **Quickstart**: 2-5 minutes
- **Full benchmark**: 10-30 minutes (depending on your machine)
- **Figure generation**: 1-2 minutes

### How much memory does Delta-Audit use?

- **Quickstart**: ~500MB RAM
- **Full benchmark**: ~2GB RAM
- **Figure generation**: ~1GB RAM

### Can I run only specific algorithms or datasets?

Yes! Create a custom configuration file:

```yaml
datasets:
  - wine  # Only wine dataset

algo_pairs:
  logreg:  # Only logistic regression
    - A: {C: 1.0, penalty: l2, solver: lbfgs}
      B: {C: 0.1, penalty: l2, solver: lbfgs}
      pair_name: pair1
```

Then run:
```bash
delta-audit run --config my_config.yaml
```

### How do I interpret the metrics?

- **BAC (Behavioral Alignment Coefficient)**: Higher values (closer to 1) indicate better alignment between attribution changes and output changes
- **DCE (Differential Conservation Error)**: Lower values indicate better conservation of attribution sums
- **Δ Magnitude L1**: Higher values indicate larger changes in feature importance
- **Rank Overlap @10**: Higher values indicate more similar top features between models

## Technical Questions

### What attribution method does Delta-Audit use?

Delta-Audit uses occlusion-based attribution methods. It replaces each feature with a baseline value (typically the mean) and measures the change in model output.

### How are baselines computed?

By default, Delta-Audit uses the mean of each feature across the training set as the baseline. This ensures that occlusion values are centered around the data distribution.

### What's the difference between BAC and DCE?

- **BAC** measures correlation between attribution change magnitude and output change magnitude
- **DCE** measures how much the sum of attribution changes differs from the actual output change

### Why do I get DCE = 0?

DCE = 0 typically means that the sum of attribution changes exactly equals the output changes. This is rare and usually indicates:
1. Perfect conservation (ideal case)
2. Very small attribution changes
3. Numerical precision issues

### Why do I get Rank Overlap = 1?

Rank Overlap = 1 means the top-10 features are identical between models A and B. This can happen when:
1. Models are very similar
2. Attribution changes are minimal
3. Only a few features have non-zero attributions

### How do I handle different random seeds?

Delta-Audit uses a fixed random seed (42) for reproducibility. To use different seeds:
1. Modify the `random_state` in your configuration file
2. Or modify the code in `src/delta_audit/runners.py`

## Troubleshooting

### "No module named 'delta_audit'"

This means the package isn't installed. Try:
```bash
pip install -e .
```

### "File not found" errors

Check that:
1. Configuration files exist in the specified paths
2. Results directories exist
3. You have write permissions

### Memory errors during full benchmark

Try:
1. Reduce the number of algorithms in your configuration
2. Use fewer datasets
3. Increase your system's available RAM
4. Close other applications

### Slow execution

To speed up execution:
1. Use fewer algorithms or datasets
2. Use a machine with more CPU cores
3. Consider using GPU acceleration (if available)

### Import errors for dependencies

Install missing dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib scipy seaborn pyyaml
```

### Figure generation fails

Common issues:
1. Missing results files - run the benchmark first
2. Insufficient disk space
3. Permission issues - check write permissions

## Extending Delta-Audit

### How do I add a new algorithm?

1. Add the algorithm configuration to your YAML config
2. Implement the algorithm in `src/delta_audit/runners.py`
3. Test with quickstart first

### How do I add a new dataset?

1. Add dataset loading function to `src/delta_audit/runners.py`
2. Add the dataset to your configuration file
3. Ensure proper preprocessing (scaling, etc.)

### How do I add a new metric?

1. Implement the metric in `src/delta_audit/metrics.py`
2. Add it to the `compute_all_metrics` function
3. Update documentation

### Can I use my own models?

Yes! You can use Delta-Audit with your own models by:
1. Implementing the attribution computation for your model
2. Using the metrics functions directly
3. Following the same interface as the built-in algorithms

## Performance Questions

### How do I optimize performance?

1. **Use fewer experiments**: Customize your configuration
2. **Parallel processing**: The code uses joblib for parallel processing
3. **Memory management**: Close other applications
4. **Hardware**: Use a machine with more CPU cores and RAM

### Can I use GPU acceleration?

Currently, Delta-Audit is CPU-based. GPU acceleration would require:
1. Implementing GPU versions of attribution methods
2. Using libraries like PyTorch or TensorFlow
3. Modifying the metrics computation for GPU

### How do I profile performance?

Use Python's built-in profiling:
```bash
python -m cProfile -o profile.stats -m delta_audit.cli run --config configs/full_benchmark.yaml
```

Then analyze with:
```python
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
```

## Research Questions

### How do I cite Delta-Audit?

```bibtex
@article{hemmat2025delta,
  title={Delta-Audit: Explaining What Changes When Models Change},
  author={Hemmat, Arshia},
  journal={arXiv preprint},
  year={2025}
}
```

### How do I reproduce the paper results?

Follow the exact steps in the [Benchmarks](benchmarks.md) documentation:
1. Install Delta-Audit
2. Run the full benchmark
3. Generate figures
4. Compare with published results

### Can I contribute to Delta-Audit?

Yes! Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Where can I get help?

- Check this FAQ first
- Review the [documentation](index.md)
- Open an issue on [GitHub](https://github.com/arshiahemmat/delta-audit)
- Check the [paper](paper/ICCKE_delta.pdf) for theoretical details 