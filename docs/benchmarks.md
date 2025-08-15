# Benchmarks

This page describes the benchmark experiments and how to reproduce the results from the paper.

## Overview

The Delta-Audit benchmark consists of 45 experiments covering:
- **5 algorithms**: Logistic Regression, SVM, Random Forest, Gradient Boosting, KNN
- **3 datasets**: Breast Cancer, Wine, Digits
- **3 pairs per algorithm**: Different hyperparameter configurations

## Experimental Setup

### Datasets

| Dataset | Type | Samples | Features | Classes |
|---------|------|---------|----------|---------|
| Breast Cancer | Binary | 569 | 30 | 2 |
| Wine | Multi-class | 178 | 13 | 3 |
| Digits | Multi-class | 1797 | 64 | 10 |

### Algorithms and Pairs

#### Logistic Regression (logreg)

| Pair | Model A | Model B | Description |
|------|---------|---------|-------------|
| pair1 | C=1.0, penalty=l2, solver=lbfgs | C=0.1, penalty=l2, solver=lbfgs | Regularization strength |
| pair2 | C=1.0, penalty=l2, solver=liblinear | C=1.0, penalty=l1, solver=liblinear | Penalty type |
| pair3 | C=1.0, penalty=l2, solver=lbfgs | C=1.0, penalty=l2, solver=saga | Solver algorithm |

#### Support Vector Classification (svc)

| Pair | Model A | Model B | Description |
|------|---------|---------|-------------|
| pair1 | C=1.0, kernel=rbf, gamma=scale | C=1.0, kernel=linear, gamma=scale | Kernel type |
| pair2 | C=1.0, kernel=rbf, gamma=scale | C=1.0, kernel=rbf, gamma=auto | Gamma parameter |
| pair3 | C=1.0, kernel=poly, degree=3, gamma=scale | C=1.0, kernel=rbf, gamma=scale | Kernel complexity |

#### Random Forest (rf)

| Pair | Model A | Model B | Description |
|------|---------|---------|-------------|
| pair1 | n_estimators=100, max_depth=None, max_features=None | n_estimators=300, max_depth=None, max_features=None | Ensemble size |
| pair2 | n_estimators=200, max_depth=None, max_features=None | n_estimators=200, max_depth=5, max_features=None | Tree depth |
| pair3 | n_estimators=200, max_depth=None, max_features=sqrt | n_estimators=200, max_depth=None, max_features=log2 | Feature selection |

#### Gradient Boosting (gb)

| Pair | Model A | Model B | Description |
|------|---------|---------|-------------|
| pair1 | n_estimators=150, learning_rate=0.1, max_depth=3 | n_estimators=150, learning_rate=0.05, max_depth=3 | Learning rate |
| pair2 | n_estimators=100, learning_rate=0.1, max_depth=3 | n_estimators=200, learning_rate=0.1, max_depth=3 | Ensemble size |
| pair3 | n_estimators=150, learning_rate=0.1, max_depth=3 | n_estimators=150, learning_rate=0.1, max_depth=5 | Tree depth |

#### K-Nearest Neighbors (knn)

| Pair | Model A | Model B | Description |
|------|---------|---------|-------------|
| pair1 | n_neighbors=5, weights=uniform, algorithm=auto | n_neighbors=10, weights=uniform, algorithm=auto | Number of neighbors |
| pair2 | n_neighbors=5, weights=uniform, algorithm=auto | n_neighbors=5, weights=distance, algorithm=auto | Weighting scheme |
| pair3 | n_neighbors=5, weights=uniform, algorithm=auto | n_neighbors=5, weights=uniform, algorithm=ball_tree | Algorithm type |

## Reproducing Results

### Prerequisites

```bash
# Install Delta-Audit
pip install -e .
pip install -r requirements.txt
```

### Run Full Benchmark

```bash
# Run all 45 experiments
delta-audit run --config configs/full_benchmark.yaml
```

This will:
1. Train model pairs across all algorithms and datasets
2. Compute Δ-Attribution metrics
3. Save results to `results/delta_summary.csv`
4. Save standard metrics to `results/standard_summary.csv`

### Generate Figures

```bash
# Generate all figures from results
delta-audit figures --summary results/_summary --out results/figures/
```

### Check Results

```bash
# Run sanity checks
delta-audit check
```

## Expected Results

### Key Statistics

| Metric | Expected Range | Description |
|--------|----------------|-------------|
| BAC | 0.1 - 0.8 | Behavioral Alignment Coefficient |
| DCE | 0.01 - 0.5 | Differential Conservation Error |
| Δ Magnitude L1 | 0.1 - 2.0 | Attribution change magnitude |
| Rank Overlap @10 | 0.2 - 0.8 | Feature overlap |
| JSD | 0.01 - 0.3 | Distributional shift |

### Performance Impact

- **Improvements**: ~40-60% of experiments show accuracy improvement
- **Degradations**: ~20-40% of experiments show accuracy degradation
- **No change**: ~10-20% of experiments show no accuracy change

### Algorithm Rankings

**Best BAC (typically):**
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. SVM
5. KNN

**Lowest DCE (typically):**
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. SVM
5. KNN

## Computational Requirements

### Time Requirements

| Component | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| Quickstart | 2-5 minutes | 1-2 minutes |
| Full benchmark | 10-30 minutes | 5-15 minutes |
| Figure generation | 1-2 minutes | 1-2 minutes |

### Memory Requirements

- **Quickstart**: ~500MB RAM
- **Full benchmark**: ~2GB RAM
- **Figure generation**: ~1GB RAM

### Hardware Recommendations

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB+ for full benchmark
- **Storage**: 1GB+ free space for results and figures

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use fewer algorithms
2. **Slow execution**: Use fewer datasets or algorithm pairs
3. **Import errors**: Ensure all dependencies are installed
4. **File not found**: Check that configuration files exist

### Performance Optimization

```bash
# Run with specific algorithms only
cat > custom_config.yaml << EOF
datasets:
  - wine
  - digits

algo_pairs:
  logreg:
    - A: {C: 1.0, penalty: l2, solver: lbfgs}
      B: {C: 0.1, penalty: l2, solver: lbfgs}
      pair_name: pair1
EOF

delta-audit run --config custom_config.yaml
```

### Validation

To validate your results:

1. **Check file sizes**: Results files should be ~50-100KB
2. **Verify metrics**: BAC should be in [-1, 1], DCE should be positive
3. **Compare with paper**: Key statistics should match published results
4. **Run quickstart**: Should complete in <5 minutes

## Extending Benchmarks

### Adding New Algorithms

1. Add algorithm configuration to `configs/full_benchmark.yaml`
2. Implement algorithm in `src/delta_audit/runners.py`
3. Test with quickstart first

### Adding New Datasets

1. Add dataset loading function to `src/delta_audit/runners.py`
2. Add dataset to configuration file
3. Ensure proper preprocessing (scaling, etc.)

### Custom Metrics

1. Implement metric in `src/delta_audit/metrics.py`
2. Add to `compute_all_metrics` function
3. Update documentation and tests 