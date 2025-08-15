"""
Δ-Attribution metrics computation module.

This module implements the complete suite of Δ-Attribution metrics for auditing
model updates, including behavioral alignment, conservation error, and stability measures.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import entropy as scipy_entropy
from sklearn.pipeline import Pipeline


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p: Probability vector (n_features,) summing to 1.
        q: Probability vector (n_features,) summing to 1.

    Returns:
        JS divergence (float)
    """
    m = 0.5 * (p + q)
    # KL divergence with protection from zeros
    def kl(a, b):
        return np.sum(a * np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_delta_magnitude_l1(delta_phi: np.ndarray) -> float:
    """Compute L1 norm of attribution differences.
    
    Args:
        delta_phi: Difference in attributions phi_B - phi_A (n_samples, n_features)
        
    Returns:
        Mean L1 norm across samples
    """
    l1_norms = np.sum(np.abs(delta_phi), axis=1)
    return float(np.mean(l1_norms))


def compute_delta_topk_frac(delta_phi: np.ndarray, k: int = 10) -> float:
    """Compute fraction of total magnitude captured by top-k features.
    
    Args:
        delta_phi: Difference in attributions phi_B - phi_A (n_samples, n_features)
        k: Number of top features to consider
        
    Returns:
        Mean top-k fraction across samples
    """
    n_features = delta_phi.shape[1]
    k = min(k, n_features)
    topk_fractions = []
    
    for i in range(len(delta_phi)):
        abs_delta = np.abs(delta_phi[i])
        denom = np.sum(abs_delta)
        if denom == 0:
            topk_fractions.append(0.0)
        else:
            top_indices = np.argsort(abs_delta)[-k:]
            top_fraction = np.sum(abs_delta[top_indices]) / (denom + 1e-12)
            topk_fractions.append(float(top_fraction))
    
    return float(np.mean(topk_fractions))


def compute_delta_entropy(delta_phi: np.ndarray) -> float:
    """Compute entropy of normalized attribution differences.
    
    Args:
        delta_phi: Difference in attributions phi_B - phi_A (n_samples, n_features)
        
    Returns:
        Mean entropy across samples
    """
    entropies = []
    
    for i in range(len(delta_phi)):
        abs_delta = np.abs(delta_phi[i])
        denom = np.sum(abs_delta)
        if denom == 0:
            entropies.append(0.0)
        else:
            p = abs_delta / denom
            entropies.append(float(scipy_entropy(p + 1e-12, base=np.e)))
    
    return float(np.mean(entropies))


def compute_rank_overlap_at_k(phi_A: np.ndarray, phi_B: np.ndarray, k: int = 10) -> Tuple[float, float]:
    """Compute rank overlap between top-k features of two attribution sets.
    
    Args:
        phi_A: Attributions for model A (n_samples, n_features)
        phi_B: Attributions for model B (n_samples, n_features)
        k: Number of top features to consider
        
    Returns:
        Tuple of (mean_overlap, median_overlap)
    """
    n_features = phi_A.shape[1]
    k = min(k, n_features)
    overlaps = []
    
    for i in range(len(phi_A)):
        abs_A = np.abs(phi_A[i])
        abs_B = np.abs(phi_B[i])
        idx_A = set(np.argsort(abs_A)[-k:])
        idx_B = set(np.argsort(abs_B)[-k:])
        intersection = len(idx_A.intersection(idx_B))
        union = len(idx_A.union(idx_B))
        if union == 0:
            overlaps.append(1.0)
        else:
            overlaps.append(intersection / union)
    
    overlaps = np.array(overlaps)
    return float(np.mean(overlaps)), float(np.median(overlaps))


def compute_js_divergence(phi_A: np.ndarray, phi_B: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between attribution distributions.
    
    Args:
        phi_A: Attributions for model A (n_samples, n_features)
        phi_B: Attributions for model B (n_samples, n_features)
        
    Returns:
        Mean JS divergence across samples
    """
    js_vals = []
    
    for i in range(len(phi_A)):
        abs_A = np.abs(phi_A[i])
        abs_B = np.abs(phi_B[i])
        sum_A = np.sum(abs_A)
        sum_B = np.sum(abs_B)
        if sum_A == 0 or sum_B == 0:
            js_vals.append(0.0)
        else:
            pA = abs_A / sum_A
            pB = abs_B / sum_B
            js_vals.append(float(js_divergence(pA, pB)))
    
    return float(np.mean(js_vals))


def compute_dce(delta_phi: np.ndarray, delta_f: np.ndarray) -> float:
    """Compute Differential Conservation Error (DCE).
    
    DCE measures the difference between sum of attribution changes and
    the actual change in model output.
    
    Args:
        delta_phi: Difference in attributions phi_B - phi_A (n_samples, n_features)
        delta_f: Difference in model outputs f_B - f_A (n_samples,)
        
    Returns:
        Mean absolute difference between sum(delta_phi) and delta_f
    """
    sum_delta_phi = np.sum(delta_phi, axis=1)
    return float(np.mean(np.abs(sum_delta_phi - delta_f)))


def compute_bac(delta_phi: np.ndarray, delta_f: np.ndarray) -> float:
    """Compute Behavioral Alignment Coefficient (BAC).
    
    BAC measures the correlation between the magnitude of attribution changes
    and the magnitude of output changes.
    
    Args:
        delta_phi: Difference in attributions phi_B - phi_A (n_samples, n_features)
        delta_f: Difference in model outputs f_B - f_A (n_samples,)
        
    Returns:
        Correlation coefficient between ||delta_phi||_1 and |delta_f|
    """
    n_samples = len(delta_phi)
    if n_samples <= 1:
        return 0.0
    
    l1_norms = np.sum(np.abs(delta_phi), axis=1)
    corr = np.corrcoef(l1_norms, np.abs(delta_f))[0, 1]
    return float(corr if not np.isnan(corr) else 0.0)


def compute_codf(delta_phi: np.ndarray, top_m_features: List[int], 
                A_correct: np.ndarray, B_correct: np.ndarray) -> Tuple[float, float]:
    """Compute COΔF (Conservation of Relevant Features) for fixes and regressions.
    
    Args:
        delta_phi: Difference in attributions phi_B - phi_A (n_samples, n_features)
        top_m_features: Indices of proxy relevant features
        A_correct: Boolean mask of whether model A predicted correctly
        B_correct: Boolean mask of whether model B predicted correctly
        
    Returns:
        Tuple of (codf_fix, codf_reg) for fixes and regressions
    """
    # Identify fixes (A wrong, B correct) and regressions (A correct, B wrong)
    fixes = np.where((~A_correct) & (B_correct))[0]
    regressions = np.where((A_correct) & (~B_correct))[0]
    
    def compute_codf_for_indices(indices):
        if len(indices) == 0:
            return np.nan
        fractions = []
        for idx in indices:
            abs_delta = np.abs(delta_phi[idx])
            denom = np.sum(abs_delta)
            if denom == 0:
                fractions.append(0.0)
            else:
                relevant_sum = np.sum(abs_delta[top_m_features])
                fractions.append(relevant_sum / (denom + 1e-12))
        return float(np.mean(fractions))
    
    codf_fix = compute_codf_for_indices(fixes)
    codf_reg = compute_codf_for_indices(regressions)
    
    return codf_fix, codf_reg


def compute_stability(model_A: Pipeline, model_B: Pipeline, X: np.ndarray, 
                     baseline: np.ndarray, sigma: float, max_samples: int = 50) -> float:
    """Compute Δ-stability by adding Gaussian noise and measuring Δφ difference.
    
    Args:
        model_A: Fitted model A
        model_B: Fitted model B
        X: Standardized input samples
        baseline: Baseline values for occlusion
        sigma: Standard deviation of Gaussian noise
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Mean of ||Δφ(x+ε) − Δφ(x)||_1 / ||ε||_2 across selected samples
    """
    from .explainers import compute_occlusion_attributions
    
    n_samples = len(X)
    indices = np.random.default_rng(42).choice(
        n_samples, size=min(max_samples, n_samples), replace=False
    )
    selected_X = X[indices]
    
    # Compute delta phi for original
    phi_A_orig = compute_occlusion_attributions(model_A, selected_X, baseline)
    phi_B_orig = compute_occlusion_attributions(model_B, selected_X, baseline)
    delta_phi_orig = phi_B_orig - phi_A_orig
    
    # Add noise
    noise = np.random.default_rng(42).normal(scale=sigma, size=selected_X.shape)
    X_noisy = selected_X + noise
    phi_A_noisy = compute_occlusion_attributions(model_A, X_noisy, baseline)
    phi_B_noisy = compute_occlusion_attributions(model_B, X_noisy, baseline)
    delta_phi_noisy = phi_B_noisy - phi_A_noisy
    
    # Compute stability measure
    diffs = np.sum(np.abs(delta_phi_noisy - delta_phi_orig), axis=1)
    norms = np.linalg.norm(noise, axis=1)
    # Avoid division by zero
    ratios = diffs / (norms + 1e-12)
    return float(np.mean(ratios))


def compute_grouped_occlusion_ratio(delta_phi: np.ndarray, feature_groups: List[List[int]]) -> float:
    """Compute ratio of attribution changes within feature groups.
    
    Args:
        delta_phi: Difference in attributions phi_B - phi_A (n_samples, n_features)
        feature_groups: List of feature group indices
        
    Returns:
        Mean ratio of within-group attribution changes
    """
    ratios = []
    
    for i in range(len(delta_phi)):
        abs_delta = np.abs(delta_phi[i])
        total_magnitude = np.sum(abs_delta)
        
        if total_magnitude == 0:
            ratios.append(0.0)
        else:
            within_group_sum = 0
            for group in feature_groups:
                within_group_sum += np.sum(abs_delta[group])
            ratios.append(within_group_sum / total_magnitude)
    
    return float(np.mean(ratios))


def compute_all_metrics(
    phi_A: np.ndarray,
    phi_B: np.ndarray,
    delta_phi: np.ndarray,
    delta_f: np.ndarray,
    top_m_features: Optional[List[int]] = None,
    A_correct: Optional[np.ndarray] = None,
    B_correct: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute the complete suite of Δ-Attribution metrics.
    
    Args:
        phi_A: Occlusion attributions for model A (n_samples, n_features)
        phi_B: Occlusion attributions for model B (n_samples, n_features)
        delta_phi: Difference phi_B - phi_A (n_samples, n_features)
        delta_f: Per-sample difference in margins f_B - f_A (n_samples,)
        top_m_features: Indices of proxy relevant features for COΔF
        A_correct: Boolean mask of whether model A predicted correctly
        B_correct: Boolean mask of whether model B predicted correctly
        
    Returns:
        Dictionary of metric name to value
    """
    metrics: Dict[str, float] = {}
    
    # Core Δ metrics
    metrics['delta_mag_l1'] = compute_delta_magnitude_l1(delta_phi)
    metrics['delta_topk_frac'] = compute_delta_topk_frac(delta_phi)
    metrics['delta_entropy'] = compute_delta_entropy(delta_phi)
    
    # Rank overlap
    mean_overlap, median_overlap = compute_rank_overlap_at_k(phi_A, phi_B)
    metrics['rank_overlap_mean'] = mean_overlap
    metrics['rank_overlap_median'] = median_overlap
    
    # Distributional shift
    metrics['delta_js'] = compute_js_divergence(phi_A, phi_B)
    
    # Conservation metrics
    metrics['dce'] = compute_dce(delta_phi, delta_f)
    metrics['bac'] = compute_bac(delta_phi, delta_f)
    
    # COΔF metrics
    if top_m_features is not None and A_correct is not None and B_correct is not None:
        codf_fix, codf_reg = compute_codf(delta_phi, top_m_features, A_correct, B_correct)
        metrics['codf_fix'] = codf_fix
        metrics['codf_reg'] = codf_reg
    else:
        metrics['codf_fix'] = np.nan
        metrics['codf_reg'] = np.nan
    
    return metrics 