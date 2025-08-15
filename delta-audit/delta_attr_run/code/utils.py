"""
Utility functions for the Delta‑Attribution experiments.

This module provides helpers for loading datasets, constructing models with
different hyperparameter settings, computing model scores, generating
occlusion‑based attributions, and computing the suite of Δ‑Attribution metrics.

The occlusion method used here is model‑agnostic and works by clamping
individual features to a baseline (mean or median) and measuring the change
in the model’s class‑specific margin. See the main script for usage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from scipy.stats import entropy as scipy_entropy


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a dataset by name and return train/test splits and feature names.

    Args:
        name: Name of the dataset. One of {'breast_cancer', 'wine', 'digits'}.
        test_size: Fraction of data to use for the test split.
        random_state: Random seed for the stratified split.

    Returns:
        X_train, X_test: feature matrices (after flattening digits)
        y_train, y_test: target vectors
        feature_names: list of feature names
    """
    if name == 'breast_cancer':
        data = load_breast_cancer()
    elif name == 'wine':
        data = load_wine()
    elif name == 'digits':
        data = load_digits()
        # Flatten images into vectors
        data = {
            'data': data.data.reshape((len(data.data), -1)),
            'target': data.target,
            'feature_names': [f'pixel_{i}' for i in range(data.data.reshape((len(data.data), -1)).shape[1])]
        }
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = data['data']
    y = data['target']
    feature_names = data.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names


def build_model(algo: str, config: Dict[str, any]) -> Pipeline:
    """Construct a model pipeline based on algorithm name and config.

    All models are wrapped in a StandardScaler except RandomForest,
    for which scaling generally does not matter but is still applied for
    consistency. This ensures that occlusion baseline values are centred.

    Args:
        algo: Name of the algorithm. One of {'logreg', 'svc', 'rf', 'gb', 'knn'}.
        config: Hyperparameters specific to the algorithm.

    Returns:
        A scikit‑learn Pipeline ready to fit.
    """
    scaler = StandardScaler()
    if algo == 'logreg':
        # Extract parameters with defaults
        C = config.get('C', 1.0)
        penalty = config.get('penalty', 'l2')
        solver = config.get('solver', 'lbfgs')
        max_iter = config.get('max_iter', 1000)
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=42)
    elif algo == 'svc':
        C = config.get('C', 1.0)
        kernel = config.get('kernel', 'rbf')
        gamma = config.get('gamma', 'scale')
        degree = config.get('degree', 3)
        # always probability=True for margin computation via predict_proba if decision_function not available
        model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True, random_state=42)
    elif algo == 'rf':
        n_estimators = config.get('n_estimators', 100)
        max_depth = config.get('max_depth', None)
        max_features = config.get('max_features', None)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42
        )
    elif algo == 'gb':
        n_estimators = config.get('n_estimators', 100)
        learning_rate = config.get('learning_rate', 0.1)
        max_depth = config.get('max_depth', 3)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    elif algo == 'knn':
        n_neighbors = config.get('n_neighbors', 5)
        weights = config.get('weights', 'uniform')
        algorithm_knn = config.get('algorithm', 'auto')
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm_knn)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Always include scaler to maintain consistent baseline mean ~0
    return Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])


def compute_decision_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Compute the class‑specific margin for each sample.

    If the underlying estimator has a decision_function, that is used.
    Otherwise, predict_proba is used and log odds are returned.

    Args:
        model: Fitted pipeline with a model.
        X: Samples (n_samples, n_features) in scaled space.

    Returns:
        scores: Array of shape (n_samples,) giving the margin for each sample
    """
    clf = model.named_steps['model']
    # Determine predicted class per sample
    preds = model.predict(X)
    # Ensure we call decision_function/predict_proba on scaled features directly
    try:
        decision = model.decision_function(X)
        if decision.ndim == 1:
            # Binary classification returns shape (n_samples,)
            # but we want per class; treat positive class as class 1
            # margin sign corresponds to class predictions
            margin = decision
        else:
            # Multi‑class: shape (n_samples, n_classes)
            # For each sample, pick the margin of the predicted class
            margin = decision[np.arange(len(X)), preds]
        return margin
    except Exception:
        # Fallback to predict_proba and log odds
        probs = model.predict_proba(X)
        # For each sample, predicted class index
        # compute logit: log(p/(1-p)). In multi‑class, use 1-p as sum of others.
        p_pred = probs[np.arange(len(X)), preds]
        one_minus = 1.0 - p_pred
        # Avoid log(0)
        logits = np.log(np.maximum(p_pred, 1e-15) / np.maximum(one_minus, 1e-15))
        return logits


def compute_attributions(
    model: Pipeline,
    X: np.ndarray,
    baseline: np.ndarray
) -> np.ndarray:
    """Compute occlusion attributions for a batch of samples.

    This method clamps each feature one at a time to a baseline value and
    measures the change in the class‑specific margin. It returns an
    attribution matrix of shape (n_samples, n_features).

    Args:
        model: Fitted pipeline with a model and scaler.
        X: Standardized input samples (n_samples, n_features).
        baseline: Baseline values for each feature in standardized space.

    Returns:
        attributions: Array (n_samples, n_features) of occlusion attributions.
    """
    n_samples, n_features = X.shape
    attributions = np.zeros((n_samples, n_features), dtype=float)
    # Compute original scores
    f_original = compute_decision_scores(model, X)
    # For each feature, clamp and recompute
    for j in range(n_features):
        X_clamped = X.copy()
        X_clamped[:, j] = baseline[j]
        f_clamped = compute_decision_scores(model, X_clamped)
        attributions[:, j] = f_original - f_clamped
    return attributions


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen‑Shannon divergence between two probability distributions.

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


def compute_delta_metrics(
    phi_A: np.ndarray,
    phi_B: np.ndarray,
    delta_phi: np.ndarray,
    delta_f: np.ndarray,
    phi_A_full: np.ndarray,
    phi_B_full: np.ndarray,
    top_m_features: Optional[List[int]] = None,
    A_correct: Optional[np.ndarray] = None,
    B_correct: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute the suite of Δ‑Attribution metrics for a set of samples.

    Args:
        phi_A: Occlusion attributions for model A (n_samples, n_features).
        phi_B: Occlusion attributions for model B (n_samples, n_features).
        delta_phi: Difference phi_B - phi_A (n_samples, n_features).
        delta_f: Per‑sample difference in margins f_B - f_A (n_samples,).
        phi_A_full: |phi_A| for full dataset used to compute RankOverlap.
        phi_B_full: |phi_B| for full dataset used to compute RankOverlap.
        top_m_features: Indices of proxy relevant features for COΔF.
        A_correct: Boolean mask of whether model A predicted correctly for each sample.
        B_correct: Boolean mask of whether model B predicted correctly for each sample.

    Returns:
        Dictionary of metric name to value.
    """
    metrics: Dict[str, float] = {}
    n_samples, n_features = delta_phi.shape
    # Δ magnitude L1
    l1_norms = np.sum(np.abs(delta_phi), axis=1)
    metrics['delta_mag_l1'] = float(np.mean(l1_norms))
    # Δ TopK
    k = min(10, n_features)
    topk_fractions = []
    # Δ entropy
    entropies = []
    js_vals = []
    for i in range(n_samples):
        abs_delta = np.abs(delta_phi[i])
        denom = np.sum(abs_delta)
        if denom == 0:
            topk_fractions.append(0.0)
            entropies.append(0.0)
        else:
            # Top K fraction
            top_indices = np.argsort(abs_delta)[-k:]
            top_fraction = np.sum(abs_delta[top_indices]) / (denom + 1e-12)
            topk_fractions.append(float(top_fraction))
            # Normalized distribution for entropy
            p = abs_delta / denom
            entropies.append(float(scipy_entropy(p + 1e-12, base=np.e)))
        # Distributional shift (JS divergence) between phi_A and phi_B
        # Use same normalization trick
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
    metrics['delta_topk_frac'] = float(np.mean(topk_fractions))
    metrics['delta_entropy'] = float(np.mean(entropies))
    metrics['delta_js'] = float(np.mean(js_vals))
    # Rank overlap between top10 of phi_A and phi_B
    overlaps = []
    for i in range(n_samples):
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
    metrics['rank_overlap_mean'] = float(np.mean(overlaps))
    metrics['rank_overlap_median'] = float(np.median(overlaps))
    # DCE: difference between sum Δφ and Δf
    sum_delta_phi = np.sum(delta_phi, axis=1)
    metrics['dce'] = float(np.mean(np.abs(sum_delta_phi - delta_f)))
    # BAC: correlation between ||Δφ|| and |Δf|
    if n_samples > 1:
        corr = np.corrcoef(l1_norms, np.abs(delta_f))[0, 1]
        # In case corr is nan (e.g., constant arrays)
        metrics['bac'] = float(corr if not np.isnan(corr) else 0.0)
    else:
        metrics['bac'] = 0.0
    # COΔF: requires top_m_features and correctness labels
    if top_m_features is not None and A_correct is not None and B_correct is not None:
        # Identify fixes (A wrong, B correct) and regressions (A correct, B wrong)
        fixes = np.where((~A_correct) & (B_correct))[0]
        regressions = np.where((A_correct) & (~B_correct))[0]
        def compute_codf(indices):
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
        metrics['codf_fix'] = compute_codf(fixes)
        metrics['codf_reg'] = compute_codf(regressions)
    else:
        metrics['codf_fix'] = np.nan
        metrics['codf_reg'] = np.nan
    return metrics


def compute_stability(
    model_A: Pipeline,
    model_B: Pipeline,
    X: np.ndarray,
    baseline: np.ndarray,
    sigma: float,
    max_samples: int = 50
) -> float:
    """Compute Δ‑stability by adding Gaussian noise and measuring Δφ difference.

    For computational efficiency, we randomly select a subset of samples.

    Args:
        model_A: Fitted model A.
        model_B: Fitted model B.
        X: Standardized input samples.
        baseline: Baseline values for occlusion.
        sigma: Standard deviation of Gaussian noise.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Mean of ||Δφ(x+ε) − Δφ(x)||_1 / ||ε||_2 across selected samples.
    """
    n_samples = len(X)
    indices = np.random.default_rng(42).choice(n_samples, size=min(max_samples, n_samples), replace=False)
    selected_X = X[indices]
    # Compute delta phi for original
    phi_A_orig = compute_attributions(model_A, selected_X, baseline)
    phi_B_orig = compute_attributions(model_B, selected_X, baseline)
    delta_phi_orig = phi_B_orig - phi_A_orig
    # Add noise
    noise = np.random.default_rng(42).normal(scale=sigma, size=selected_X.shape)
    X_noisy = selected_X + noise
    phi_A_noisy = compute_attributions(model_A, X_noisy, baseline)
    phi_B_noisy = compute_attributions(model_B, X_noisy, baseline)
    delta_phi_noisy = phi_B_noisy - phi_A_noisy
    # Compute stability measure
    diffs = np.sum(np.abs(delta_phi_noisy - delta_phi_orig), axis=1)
    norms = np.linalg.norm(noise, axis=1)
    # Avoid division by zero
    ratios = diffs / (norms + 1e-12)
    return float(np.mean(ratios))
