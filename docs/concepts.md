# Δ-Attribution Concepts

This page explains the core concepts behind Δ-Attribution and how Delta-Audit implements them.

## What is Δ-Attribution?

Δ-Attribution (Delta-Attribution) is a framework for understanding how model explanations change when models are updated. When you train a new version of a model, not only do the predictions change, but also the explanations of those predictions.

## Core Concepts

### Model Pairs (A→B)

A model pair consists of two models:
- **Model A**: The baseline/original model
- **Model B**: The updated/new model

The goal is to understand how explanations change from A to B.

### Attribution Methods

Delta-Audit uses occlusion-based attribution methods:

1. **Occlusion**: Replace a feature with a baseline value and measure the change in model output
2. **Clamping**: Same as occlusion, but emphasizes the feature replacement aspect
3. **Common Class Anchor**: Use class-specific baseline values

### Baseline Computation

- **Mean Baseline**: Use the mean value of each feature across the training set
- **Median Baseline**: Use the median value of each feature
- **Zero Baseline**: Use zero as the baseline value

## Key Metrics

### Behavioral Alignment Coefficient (BAC)

BAC measures how well attribution changes correlate with output changes:

```
BAC = Corr(||Δφ||₁, |Δf|)
```

Where:
- Δφ = φ_B - φ_A (attribution differences)
- Δf = f_B - f_A (output differences)
- ||·||₁ = L1 norm

**Interpretation**: Higher BAC values indicate that when the model output changes significantly, the attributions also change significantly.

### Differential Conservation Error (DCE)

DCE measures how much the sum of attribution changes differs from the actual output change:

```
DCE = E[|ΣΔφ - Δf|]
```

**Interpretation**: Lower DCE values indicate better conservation - the sum of attribution changes closely matches the output change.

### Δ Magnitude L1

Measures the overall magnitude of attribution changes:

```
Δ Magnitude L1 = E[||Δφ||₁]
```

**Interpretation**: Higher values indicate larger changes in feature importance between models.

### Rank Overlap @K

Measures how much the top-K features overlap between models:

```
Rank Overlap @K = |TopK(φ_A) ∩ TopK(φ_B)| / |TopK(φ_A) ∪ TopK(φ_B)|
```

**Interpretation**: Higher values indicate that the most important features remain similar between models.

### Jensen-Shannon Divergence (JSD)

Measures the distributional shift between attribution sets:

```
JSD = 0.5 * KL(φ_A || M) + 0.5 * KL(φ_B || M)
```

Where M = 0.5 * (φ_A + φ_B)

**Interpretation**: Higher values indicate larger distributional changes in feature importance.

## Experimental Design

### Algorithm Pairs

Each algorithm is tested with 3 different hyperparameter pairs:

1. **Pair 1**: Different regularization/strength parameters
2. **Pair 2**: Different structural parameters (e.g., kernel, penalty)
3. **Pair 3**: Different optimization parameters (e.g., solver, algorithm)

### Datasets

- **Breast Cancer**: Binary classification with 30 features
- **Wine**: Multi-class classification with 13 features  
- **Digits**: Multi-class classification with 64 features

### Evaluation Protocol

1. Stratified train/test split (80/20)
2. StandardScaler applied to all features
3. Random state fixed to 42 for reproducibility
4. Mean baseline computed from training data
5. Metrics computed on test set

## Interpretation Guidelines

### High BAC, Low DCE
- Good behavioral alignment
- Attribution changes correlate well with output changes
- Model updates are well-explained

### Low BAC, High DCE
- Poor behavioral alignment
- Attribution changes don't correlate with output changes
- Model updates are poorly explained

### High Δ Magnitude
- Large changes in feature importance
- Model behavior has changed significantly

### Low Rank Overlap
- Different features are important in the new model
- Model has learned different patterns

### High JSD
- Large distributional shift in attributions
- Model explanations have changed substantially 