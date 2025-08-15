"""
Attribution computation methods for Δ-Audit.

This module provides various attribution methods including occlusion,
clamping, and grouped occlusion techniques.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from sklearn.pipeline import Pipeline


def compute_decision_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Compute decision scores for a model on input data.
    
    Args:
        model: Fitted scikit-learn pipeline
        X: Input features (n_samples, n_features)
        
    Returns:
        Decision scores (n_samples,)
    """
    try:
        # Try decision_function first (for SVM, LogisticRegression)
        scores = model.decision_function(X)
        if scores.ndim > 1:
            # Multi-class case, take the maximum score
            scores = np.max(scores, axis=1)
    except (AttributeError, ValueError):
        # Fall back to predict_proba and take the maximum probability
        probs = model.predict_proba(X)
        scores = np.max(probs, axis=1)
    
    return scores


def compute_occlusion_attributions(model: Pipeline, X: np.ndarray, 
                                  baseline: np.ndarray) -> np.ndarray:
    """Compute occlusion-based attributions.
    
    Args:
        model: Fitted scikit-learn pipeline
        X: Input features (n_samples, n_features)
        baseline: Baseline values for occlusion (n_features,)
        
    Returns:
        Attributions (n_samples, n_features)
    """
    n_samples, n_features = X.shape
    attributions = np.zeros((n_samples, n_features))
    
    # Compute original scores
    f_original = compute_decision_scores(model, X)
    
    # Compute attributions by occluding each feature
    for j in range(n_features):
        X_occluded = X.copy()
        X_occluded[:, j] = baseline[j]
        f_occluded = compute_decision_scores(model, X_occluded)
        attributions[:, j] = f_original - f_occluded
    
    return attributions


def compute_clamping_attributions(model: Pipeline, X: np.ndarray,
                                 baseline: np.ndarray) -> np.ndarray:
    """Compute clamping-based attributions (same as occlusion).
    
    Args:
        model: Fitted scikit-learn pipeline
        X: Input features (n_samples, n_features)
        baseline: Baseline values for clamping (n_features,)
        
    Returns:
        Attributions (n_samples, n_features)
    """
    return compute_occlusion_attributions(model, X, baseline)


def compute_common_class_anchor(model: Pipeline, X: np.ndarray,
                               y: np.ndarray) -> np.ndarray:
    """Compute attributions using common class anchor baseline.
    
    Args:
        model: Fitted scikit-learn pipeline
        X: Input features (n_samples, n_features)
        y: True labels (n_samples,)
        
    Returns:
        Attributions (n_samples, n_features)
    """
    # For each class, compute the mean feature vector
    unique_classes = np.unique(y)
    class_means = {}
    
    for cls in unique_classes:
        class_mask = (y == cls)
        class_means[cls] = np.mean(X[class_mask], axis=0)
    
    n_samples, n_features = X.shape
    attributions = np.zeros((n_samples, n_features))
    
    # Compute original scores
    f_original = compute_decision_scores(model, X)
    
    # Compute attributions using class-specific baselines
    for i in range(n_samples):
        baseline = class_means[y[i]]
        for j in range(n_features):
            X_clamped = X[i:i+1].copy()
            X_clamped[0, j] = baseline[j]
            f_clamped = compute_decision_scores(model, X_clamped)
            attributions[i, j] = f_original[i] - f_clamped[0]
    
    return attributions


def compute_grouped_occlusion(model: Pipeline, X: np.ndarray,
                             baseline: np.ndarray, feature_groups: List[List[int]]) -> np.ndarray:
    """Compute grouped occlusion attributions.
    
    Args:
        model: Fitted scikit-learn pipeline
        X: Input features (n_samples, n_features)
        baseline: Baseline values for occlusion (n_features,)
        feature_groups: List of feature group indices
        
    Returns:
        Group-level attributions (n_samples, n_groups)
    """
    n_samples = len(X)
    n_groups = len(feature_groups)
    group_attributions = np.zeros((n_samples, n_groups))
    
    # Compute original scores
    f_original = compute_decision_scores(model, X)
    
    # Compute attributions for each group
    for g, group_indices in enumerate(feature_groups):
        X_group_occluded = X.copy()
        for j in group_indices:
            X_group_occluded[:, j] = baseline[j]
        f_group_occluded = compute_decision_scores(model, X_group_occluded)
        group_attributions[:, g] = f_original - f_group_occluded
    
    return group_attributions


def compute_mean_baseline(X: np.ndarray) -> np.ndarray:
    """Compute mean baseline for attribution methods.
    
    Args:
        X: Input features (n_samples, n_features)
        
    Returns:
        Mean baseline (n_features,)
    """
    return np.mean(X, axis=0)


def compute_median_baseline(X: np.ndarray) -> np.ndarray:
    """Compute median baseline for attribution methods.
    
    Args:
        X: Input features (n_samples, n_features)
        
    Returns:
        Median baseline (n_features,)
    """
    return np.median(X, axis=0)


def compute_zero_baseline(n_features: int) -> np.ndarray:
    """Compute zero baseline for attribution methods.
    
    Args:
        n_features: Number of features
        
    Returns:
        Zero baseline (n_features,)
    """
    return np.zeros(n_features)


def compute_permutation_importance(model: Pipeline, X: np.ndarray, 
                                  y: np.ndarray, n_repeats: int = 10) -> np.ndarray:
    """Compute permutation importance for feature ranking.
    
    Args:
        model: Fitted scikit-learn pipeline
        X: Input features (n_samples, n_features)
        y: True labels (n_samples,)
        n_repeats: Number of permutation repeats
        
    Returns:
        Permutation importance scores (n_features,)
    """
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    return result.importances_mean


def get_top_features_by_importance(model: Pipeline, X: np.ndarray, 
                                  y: np.ndarray, top_k: int = 10) -> List[int]:
    """Get top-k features by permutation importance.
    
    Args:
        model: Fitted scikit-learn pipeline
        X: Input features (n_samples, n_features)
        y: True labels (n_samples,)
        top_k: Number of top features to return
        
    Returns:
        List of top feature indices
    """
    importance_scores = compute_permutation_importance(model, X, y)
    top_indices = np.argsort(importance_scores)[-top_k:]
    return top_indices.tolist() 