# Δ-Attribution Metrics

This page provides detailed descriptions of all Δ-Attribution metrics implemented in Delta-Audit.

## Core Metrics

### Behavioral Alignment Coefficient (BAC)

**Definition**: Correlation between attribution change magnitude and output change magnitude.

**Formula**: `BAC = Corr(||Δφ||₁, |Δf|)`

**Range**: [-1, 1]

**Interpretation**: 
- Values close to 1: Strong positive correlation
- Values close to 0: No correlation
- Values close to -1: Strong negative correlation

**Use Case**: Measures how well attribution changes align with output changes.

### Differential Conservation Error (DCE)

**Definition**: Mean absolute difference between sum of attribution changes and actual output change.

**Formula**: `DCE = E[|ΣΔφ - Δf|]`

**Range**: [0, ∞)

**Interpretation**: 
- Lower values indicate better conservation
- Higher values indicate poor conservation

**Use Case**: Measures how much the sum of attribution changes differs from the actual output change.

### Δ Magnitude L1

**Definition**: Mean L1 norm of attribution differences.

**Formula**: `Δ Magnitude L1 = E[||Δφ||₁]`

**Range**: [0, ∞)

**Interpretation**: 
- Higher values indicate larger changes in feature importance
- Lower values indicate smaller changes

**Use Case**: Measures the overall magnitude of attribution changes.

### Δ TopK10

**Definition**: Mean fraction of total magnitude captured by top-10 features.

**Formula**: `Δ TopK10 = E[Σᵢ∈TopK |Δφᵢ| / Σⱼ |Δφⱼ|]`

**Range**: [0, 1]

**Interpretation**: 
- Higher values indicate concentration in top features
- Lower values indicate more distributed changes

**Use Case**: Measures how concentrated attribution changes are in the most important features.

### Δ Entropy

**Definition**: Mean entropy of normalized attribution differences.

**Formula**: `Δ Entropy = E[H(|Δφ| / Σ|Δφ|)]`

**Range**: [0, log(n_features)]

**Interpretation**: 
- Higher values indicate more uniform distribution
- Lower values indicate more concentrated distribution

**Use Case**: Measures the distributional complexity of attribution changes.

## Rank-Based Metrics

### Rank Overlap @10

**Definition**: Mean overlap between top-10 features of two attribution sets.

**Formula**: `Rank Overlap @10 = E[|TopK(φ_A) ∩ TopK(φ_B)| / |TopK(φ_A) ∪ TopK(φ_B)|]`

**Range**: [0, 1]

**Interpretation**: 
- Higher values indicate similar top features
- Lower values indicate different top features

**Use Case**: Measures how much the most important features overlap between models.

### Rank Overlap Median

**Definition**: Median overlap between top-10 features.

**Formula**: `Median(Rank Overlap @10)`

**Range**: [0, 1]

**Interpretation**: Robust measure of feature overlap.

**Use Case**: Less sensitive to outliers than mean rank overlap.

## Distributional Metrics

### Jensen-Shannon Divergence (JSD)

**Definition**: Distributional shift between attribution sets.

**Formula**: `JSD = 0.5 * KL(φ_A || M) + 0.5 * KL(φ_B || M)`

Where `M = 0.5 * (φ_A + φ_B)`

**Range**: [0, log(2)]

**Interpretation**: 
- Higher values indicate larger distributional changes
- Lower values indicate similar distributions

**Use Case**: Measures how much the attribution distributions differ.

## Conservation Metrics

### COΔF (Conservation of Relevant Features)

**Definition**: Fraction of attribution changes in relevant features for fixes and regressions.

**Formula**: 
- `COΔF_fix = E[Σᵢ∈TopM |Δφᵢ| / Σⱼ |Δφⱼ|]` for fixes
- `COΔF_reg = E[Σᵢ∈TopM |Δφᵢ| / Σⱼ |Δφⱼ|]` for regressions

**Range**: [0, 1]

**Interpretation**: 
- Higher values indicate changes concentrated in relevant features
- Lower values indicate changes in irrelevant features

**Use Case**: Measures how well attribution changes focus on relevant features for performance changes.

## Stability Metrics

### Δ Stability

**Definition**: Robustness of attribution changes to input perturbations.

**Formula**: `Δ Stability = E[||Δφ(x+ε) - Δφ(x)||₁ / ||ε||₂]`

**Range**: [0, ∞)

**Interpretation**: 
- Lower values indicate more stable attributions
- Higher values indicate less stable attributions

**Use Case**: Measures how robust attribution changes are to input noise.

## Standard Metrics

### Accuracy A/B

**Definition**: Classification accuracy of models A and B.

**Range**: [0, 1]

**Use Case**: Standard performance comparison.

### Macro F1 A/B

**Definition**: Macro-averaged F1 score of models A and B.

**Range**: [0, 1]

**Use Case**: Performance comparison for multi-class problems.

### Macro Precision A/B

**Definition**: Macro-averaged precision of models A and B.

**Range**: [0, 1]

**Use Case**: Performance comparison for multi-class problems.

## Metric Relationships

### Complementary Metrics
- **BAC and DCE**: BAC measures correlation, DCE measures conservation
- **Δ Magnitude and Rank Overlap**: Magnitude measures size, overlap measures similarity
- **JSD and Rank Overlap**: JSD measures distributional change, overlap measures feature similarity

### Interpretation Guidelines
1. **High BAC, Low DCE**: Good behavioral alignment and conservation
2. **Low BAC, High DCE**: Poor behavioral alignment and conservation
3. **High Δ Magnitude, Low Rank Overlap**: Large changes with different important features
4. **High JSD, Low Rank Overlap**: Large distributional changes with different important features 