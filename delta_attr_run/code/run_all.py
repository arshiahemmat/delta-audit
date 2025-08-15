"""
Main entry point for running Δ‑Attribution experiments.

This script orchestrates training of baseline (A) and updated (B) models
across multiple algorithms and hyperparameter pairs on three datasets. It
computes standard evaluation metrics as well as a comprehensive suite of
Δ‑Attribution metrics and exports results and plots to the specified
directory structure.

Usage:
    python run_all.py

All paths are relative to the project root (delta_attr_run). Ensure that
the directory structure exists or is created by this script.
"""

from __future__ import annotations

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib

# Use Agg backend for non‑interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Dict, List, Tuple

from utils import (
    load_dataset,
    build_model,
    compute_decision_scores,
    compute_attributions,
    compute_delta_metrics,
    compute_stability,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix


# Define configuration for experiments
DATASETS = ['breast_cancer', 'wine', 'digits']

ALGO_PAIRS: Dict[str, List[Tuple[Dict[str, any], Dict[str, any], str]]] = {
    # Each tuple: (A_config, B_config, pair_name)
    'logreg': [
        ({'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}, {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}, 'pair1'),
        ({'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}, {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}, 'pair2'),
        ({'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}, {'C': 1.0, 'penalty': 'l2', 'solver': 'saga'}, 'pair3'),
    ],
    'svc': [
        ({'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}, {'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}, 'pair1'),
        ({'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}, {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'}, 'pair2'),
        ({'C': 1.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'}, {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}, 'pair3'),
    ],
    'rf': [
        ({'n_estimators': 100, 'max_depth': None, 'max_features': None}, {'n_estimators': 300, 'max_depth': None, 'max_features': None}, 'pair1'),
        ({'n_estimators': 200, 'max_depth': None, 'max_features': None}, {'n_estimators': 200, 'max_depth': 5, 'max_features': None}, 'pair2'),
        ({'n_estimators': 200, 'max_depth': None, 'max_features': 'sqrt'}, {'n_estimators': 200, 'max_depth': None, 'max_features': 'log2'}, 'pair3'),
    ],
    'gb': [
        ({'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 3}, {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 3}, 'pair1'),
        ({'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}, {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3}, 'pair2'),
        ({'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 3}, {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5}, 'pair3'),
    ],
    'knn': [
        ({'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}, {'n_neighbors': 10, 'weights': 'uniform', 'algorithm': 'auto'}, 'pair1'),
        ({'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}, {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto'}, 'pair2'),
        ({'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}, {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'ball_tree'}, 'pair3'),
    ],
}


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def save_confusion_matrix(cm: np.ndarray, labels: List[int], filepath: str) -> None:
    """Save confusion matrix as a plot."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    # Show all ticks and label them
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def run_experiment() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    results_dir = os.path.join(project_root, 'results')
    summary_dir = os.path.join(results_dir, '_summary')
    ensure_dir(summary_dir)
    # Prepare summary dataframes
    standard_rows = []
    delta_rows = []
    # Loop over datasets
    for dataset_name in DATASETS:
        print(f"Processing dataset: {dataset_name}")
        X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset_name)
        # Compute baseline mean and median in original feature space
        baseline_mean = np.mean(X_train, axis=0)
        baseline_median = np.median(X_train, axis=0)
        for algo, pairs in ALGO_PAIRS.items():
            for (config_A, config_B, pair_name) in pairs:
                print(f"  Running {algo} {pair_name} on {dataset_name}")
                # Build models
                model_A = build_model(algo, config_A)
                model_B = build_model(algo, config_B)
                # Fit models
                model_A.fit(X_train, y_train)
                model_B.fit(X_train, y_train)
                # Evaluate on test data
                y_pred_A = model_A.predict(X_test)
                y_pred_B = model_B.predict(X_test)
                # Standard metrics
                acc_A = accuracy_score(y_test, y_pred_A)
                acc_B = accuracy_score(y_test, y_pred_B)
                precision_A = precision_score(y_test, y_pred_A, average='macro', zero_division=0)
                precision_B = precision_score(y_test, y_pred_B, average='macro', zero_division=0)
                f1_A = f1_score(y_test, y_pred_A, average='macro', zero_division=0)
                f1_B = f1_score(y_test, y_pred_B, average='macro', zero_division=0)
                # Confusion matrices
                cm_A = confusion_matrix(y_test, y_pred_A)
                cm_B = confusion_matrix(y_test, y_pred_B)
                # Determine output directories
                out_dir = os.path.join(results_dir, dataset_name, algo, pair_name)
                ensure_dir(out_dir)
                # Save confusion matrix plots
                label_list = list(np.unique(y_test))
                save_confusion_matrix(cm_A, label_list, os.path.join(out_dir, 'confusion_A.png'))
                save_confusion_matrix(cm_B, label_list, os.path.join(out_dir, 'confusion_B.png'))
                # Compute attributions on subset for Δ metrics
                # Use a smaller subset for efficiency
                subset_size = min(128, len(X_test))
                indices = np.arange(subset_size)
                # Using the first subset_size samples for reproducibility
                X_sub = X_test[indices]
                y_sub = y_test[indices]
                # Attributions for mean baseline
                phi_A = compute_attributions(model_A, X_sub, baseline_mean)
                phi_B = compute_attributions(model_B, X_sub, baseline_mean)
                delta_phi = phi_B - phi_A
                # f(x) values and delta f
                f_A = compute_decision_scores(model_A, X_sub)
                f_B = compute_decision_scores(model_B, X_sub)
                delta_f = f_B - f_A
                # Compute correctness flags
                A_correct = y_pred_A[indices] == y_sub
                B_correct = y_pred_B[indices] == y_sub
                # Permutation importance on model B for proxy relevant features (top 10)
                try:
                    perm_import = permutation_importance(model_B, X_test, y_test, n_repeats=3, random_state=42)
                    # Use absolute importances
                    importances = np.abs(perm_import.importances_mean)
                    top_m = 10
                    top_m_indices = list(np.argsort(importances)[-top_m:])
                except Exception:
                    top_m_indices = None
                # Compute Δ metrics for mean baseline
                delta_metrics_mean = compute_delta_metrics(
                    phi_A, phi_B, delta_phi, delta_f, phi_A, phi_B,
                    top_m_features=top_m_indices,
                    A_correct=A_correct,
                    B_correct=B_correct
                )
                # Compute stability metrics
                stability_001 = compute_stability(model_A, model_B, X_sub, baseline_mean, sigma=0.01, max_samples=50)
                stability_005 = compute_stability(model_A, model_B, X_sub, baseline_mean, sigma=0.05, max_samples=50)
                # Baseline sensitivity: compute Δ metrics with median baseline
                phi_A_med = compute_attributions(model_A, X_sub, baseline_median)
                phi_B_med = compute_attributions(model_B, X_sub, baseline_median)
                delta_phi_med = phi_B_med - phi_A_med
                # Compute delta mag and topk for median baseline
                delta_mag_med = float(np.mean(np.sum(np.abs(delta_phi_med), axis=1)))
                # TopK fraction for median baseline
                l_med = []
                k = min(10, X_sub.shape[1])
                for i in range(len(delta_phi_med)):
                    abs_delta = np.abs(delta_phi_med[i])
                    denom = np.sum(abs_delta)
                    if denom == 0:
                        l_med.append(0.0)
                    else:
                        top_inds = np.argsort(abs_delta)[-k:]
                        l_med.append(np.sum(abs_delta[top_inds]) / (denom + 1e-12))
                delta_topk_med = float(np.mean(l_med))
                # Baseline sensitivity percentage change
                base_delta_mag = delta_metrics_mean['delta_mag_l1']
                base_delta_topk = delta_metrics_mean['delta_topk_frac']
                sens_mag = float((delta_mag_med - base_delta_mag) / (base_delta_mag + 1e-12) * 100.0)
                sens_topk = float((delta_topk_med - base_delta_topk) / (base_delta_topk + 1e-12) * 100.0)
                # Save plots
                # Global delta top10 bar plot
                mean_abs_delta = np.mean(np.abs(delta_phi), axis=0)
                # Top 10 features
                top_idx = np.argsort(mean_abs_delta)[-k:][::-1]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(range(len(top_idx)), mean_abs_delta[top_idx])
                ax.set_xticks(range(len(top_idx)))
                ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha='right')
                ax.set_title('Top 10 Δ‑Attribution Magnitudes')
                ax.set_ylabel('Mean |Δφ|')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, 'delta_topk10_bar.png'))
                plt.close(fig)
                # DCE histogram
                dce_values = np.abs(np.sum(delta_phi, axis=1) - delta_f)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(dce_values, bins=20)
                ax.set_title('Histogram of |ΣΔφ − Δf|')
                ax.set_xlabel('|ΣΔφ − Δf|')
                ax.set_ylabel('Count')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, 'dce_hist.png'))
                plt.close(fig)
                # BAC scatter
                l1_norms = np.sum(np.abs(delta_phi), axis=1)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(l1_norms, np.abs(delta_f))
                ax.set_xlabel('||Δφ||₁')
                ax.set_ylabel('|Δf|')
                ax.set_title('BAC Scatter (||Δφ|| vs |Δf|)')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, 'bac_scatter.png'))
                plt.close(fig)
                # Stability line plot
                fig, ax = plt.subplots(figsize=(6, 4))
                sigmas = [0.01, 0.05]
                stabilities = [stability_001, stability_005]
                ax.plot(sigmas, stabilities, marker='o')
                ax.set_xlabel('σ')
                ax.set_ylabel('Δ‑Stability')
                ax.set_title('Δ‑Stability vs Noise Level')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, 'stability_line.png'))
                plt.close(fig)
                # Write standard metrics to CSV
                standard_row = {
                    'dataset': dataset_name,
                    'algorithm': algo,
                    'pair': pair_name,
                    'accuracy_A': acc_A,
                    'accuracy_B': acc_B,
                    'macro_precision_A': precision_A,
                    'macro_precision_B': precision_B,
                    'macro_f1_A': f1_A,
                    'macro_f1_B': f1_B,
                }
                standard_rows.append(standard_row)
                # Write delta metrics to CSV
                delta_row = {
                    'dataset': dataset_name,
                    'algorithm': algo,
                    'pair': pair_name,
                    'delta_mag_l1': delta_metrics_mean['delta_mag_l1'],
                    'delta_topk_frac': delta_metrics_mean['delta_topk_frac'],
                    'delta_entropy': delta_metrics_mean['delta_entropy'],
                    'rank_overlap_mean': delta_metrics_mean['rank_overlap_mean'],
                    'rank_overlap_median': delta_metrics_mean['rank_overlap_median'],
                    'delta_js': delta_metrics_mean['delta_js'],
                    'dce': delta_metrics_mean['dce'],
                    'bac': delta_metrics_mean['bac'],
                    'codf_fix': delta_metrics_mean['codf_fix'],
                    'codf_reg': delta_metrics_mean['codf_reg'],
                    'delta_stability_sigma001': stability_001,
                    'delta_stability_sigma005': stability_005,
                    'baseline_sens_delta_mag_pct': sens_mag,
                    'baseline_sens_delta_topk_pct': sens_topk,
                }
                delta_rows.append(delta_row)
                # Save individual metrics files for reproducibility
                # Standard metrics
                pd.DataFrame([standard_row]).to_csv(
                    os.path.join(out_dir, 'metrics_standard.csv'), index=False
                )
                # Delta metrics
                pd.DataFrame([delta_row]).to_csv(
                    os.path.join(out_dir, 'metrics_delta.csv'), index=False
                )
    # Save summary CSVs
    pd.DataFrame(standard_rows).to_csv(
        os.path.join(summary_dir, 'standard_summary.csv'), index=False
    )
    pd.DataFrame(delta_rows).to_csv(
        os.path.join(summary_dir, 'delta_summary.csv'), index=False
    )
    # Write README summarising key interpretations (template)
    readme_path = os.path.join(summary_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Δ‑Attribution Experiment Summary\n\n")
        f.write("This summary reports aggregated metrics across all datasets, algorithms and configuration pairs.\n\n")
        f.write("## Interpretation guidelines\n\n")
        f.write("- **ΔTopK high, ΔEntropy low**: updates concentrated on a small set of features.\n")
        f.write("- **Positive BAC**: larger changes in attributions accompany larger changes in model behaviour.\n")
        f.write("- **Smaller DCE**: Δ attributions better explain the output change (conservation).\n")
        f.write("- **COΔF_fix > COΔF_reg**: Δ mass moved onto task‑relevant signals for corrections (good).\n")
        f.write("- **Baseline sensitivity**: percentage change of Δ metrics when using median baseline instead of mean; lower values indicate robustness.\n\n")
        f.write("Refer to the per‑pair metric files in the respective directories for detailed results.\n")


if __name__ == '__main__':
    run_experiment()