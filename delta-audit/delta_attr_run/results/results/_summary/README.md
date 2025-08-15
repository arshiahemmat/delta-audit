# Δ‑Attribution Experiment Summary

This summary reports aggregated metrics across all datasets, algorithms and configuration pairs.

## Interpretation guidelines

- **ΔTopK high, ΔEntropy low**: updates concentrated on a small set of features.
- **Positive BAC**: larger changes in attributions accompany larger changes in model behaviour.
- **Smaller DCE**: Δ attributions better explain the output change (conservation).
- **COΔF_fix > COΔF_reg**: Δ mass moved onto task‑relevant signals for corrections (good).
- **Baseline sensitivity**: percentage change of Δ metrics when using median baseline instead of mean; lower values indicate robustness.

Refer to the per‑pair metric files in the respective directories for detailed results.
