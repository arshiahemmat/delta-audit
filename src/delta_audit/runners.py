"""
Training and evaluation pipelines for Δ-Audit.

This module provides functions for training model pairs, computing attributions,
and evaluating Δ-Attribution metrics across different algorithms and datasets.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

from .metrics import compute_all_metrics
from .explainers import (
    compute_occlusion_attributions,
    compute_mean_baseline,
    get_top_features_by_importance
)


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

    All models are wrapped in a StandardScaler for consistency.

    Args:
        algo: Name of the algorithm. One of {'logreg', 'svc', 'rf', 'gb', 'knn'}.
        config: Hyperparameters specific to the algorithm.

    Returns:
        A scikit-learn Pipeline ready to fit.
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


def train_model_pair(X_train: np.ndarray, y_train: np.ndarray, 
                    algo: str, config_A: Dict, config_B: Dict) -> Tuple[Pipeline, Pipeline]:
    """Train a pair of models (A and B) with different configurations.
    
    Args:
        X_train: Training features
        y_train: Training labels
        algo: Algorithm name
        config_A: Configuration for model A
        config_B: Configuration for model B
        
    Returns:
        Tuple of (model_A, model_B) fitted pipelines
    """
    model_A = build_model(algo, config_A)
    model_B = build_model(algo, config_B)
    
    model_A.fit(X_train, y_train)
    model_B.fit(X_train, y_train)
    
    return model_A, model_B


def evaluate_model_pair(model_A: Pipeline, model_B: Pipeline, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       baseline: np.ndarray) -> Dict[str, any]:
    """Evaluate a pair of models and compute Δ-Attribution metrics.
    
    Args:
        model_A: Fitted model A
        model_B: Fitted model B
        X_test: Test features
        y_test: Test labels
        baseline: Baseline values for attribution computation
        
    Returns:
        Dictionary containing standard metrics and Δ-Attribution metrics
    """
    # Standard evaluation metrics
    y_pred_A = model_A.predict(X_test)
    y_pred_B = model_B.predict(X_test)
    
    # Compute decision scores
    from .explainers import compute_decision_scores
    f_A = compute_decision_scores(model_A, X_test)
    f_B = compute_decision_scores(model_B, X_test)
    delta_f = f_B - f_A
    
    # Compute attributions
    phi_A = compute_occlusion_attributions(model_A, X_test, baseline)
    phi_B = compute_occlusion_attributions(model_B, X_test, baseline)
    delta_phi = phi_B - phi_A
    
    # Get top features for COΔF computation
    top_m_features = get_top_features_by_importance(model_A, X_test, y_test, top_k=10)
    
    # Compute correctness masks
    A_correct = (y_pred_A == y_test)
    B_correct = (y_pred_B == y_test)
    
    # Compute Δ-Attribution metrics
    delta_metrics = compute_all_metrics(
        phi_A, phi_B, delta_phi, delta_f,
        top_m_features=top_m_features,
        A_correct=A_correct,
        B_correct=B_correct
    )
    
    # Standard metrics
    standard_metrics = {
        'accuracy_A': accuracy_score(y_test, y_pred_A),
        'accuracy_B': accuracy_score(y_test, y_pred_B),
        'macro_precision_A': precision_score(y_test, y_pred_A, average='macro', zero_division=0),
        'macro_precision_B': precision_score(y_test, y_pred_B, average='macro', zero_division=0),
        'macro_f1_A': f1_score(y_test, y_pred_A, average='macro', zero_division=0),
        'macro_f1_B': f1_score(y_test, y_pred_B, average='macro', zero_division=0),
    }
    
    # Combine metrics
    results = {**standard_metrics, **delta_metrics}
    
    return results


def run_quickstart(output_dir: str = "results") -> None:
    """Run a quick demonstration with a subset of experiments.
    
    Args:
        output_dir: Directory to save results
    """
    print("Running Delta-Audit quickstart...")
    
    # Run a small subset: one dataset, one algorithm, one pair
    dataset = 'wine'
    algo = 'logreg'
    config_A, config_B, pair_name = ALGO_PAIRS[algo][0]  # Use first pair
    
    # Load dataset
    X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset)
    
    # Compute baseline
    baseline = compute_mean_baseline(X_train)
    
    # Train models
    model_A, model_B = train_model_pair(X_train, y_train, algo, config_A, config_B)
    
    # Evaluate
    results = evaluate_model_pair(model_A, model_B, X_test, y_test, baseline)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    results_df['dataset'] = dataset
    results_df['algorithm'] = algo
    results_df['pair'] = pair_name
    
    results_df.to_csv(f"{output_dir}/quickstart_results.csv", index=False)
    print(f"Quickstart completed! Results saved to {output_dir}/quickstart_results.csv")
    print(f"BAC: {results['bac']:.3f}, DCE: {results['dce']:.3f}")


def run_benchmark(config_path: str, output_dir: str = "results") -> None:
    """Run the full benchmark across all datasets, algorithms, and pairs.
    
    Args:
        config_path: Path to configuration file (optional, uses defaults if not provided)
        output_dir: Directory to save results
    """
    print("Running Delta-Audit full benchmark...")
    
    # Load config if provided
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        datasets = config.get('datasets', DATASETS)
        algo_pairs = config.get('algo_pairs', ALGO_PAIRS)
    else:
        datasets = DATASETS
        algo_pairs = ALGO_PAIRS
    
    # Prepare results storage
    all_results = []
    
    # Run experiments
    total_experiments = sum(len(pairs) for pairs in algo_pairs.values()) * len(datasets)
    experiment_count = 0
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # Load dataset
        X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset)
        baseline = compute_mean_baseline(X_train)
        
        for algo, pairs in algo_pairs.items():
            print(f"  Algorithm: {algo}")
            
            for config_A, config_B, pair_name in pairs:
                experiment_count += 1
                print(f"    Pair {experiment_count}/{total_experiments}: {pair_name}")
                
                try:
                    # Train models
                    model_A, model_B = train_model_pair(X_train, y_train, algo, config_A, config_B)
                    
                    # Evaluate
                    results = evaluate_model_pair(model_A, model_B, X_test, y_test, baseline)
                    
                    # Add metadata
                    results['dataset'] = dataset
                    results['algorithm'] = algo
                    results['pair'] = pair_name
                    
                    all_results.append(results)
                    
                except Exception as e:
                    print(f"    Error in {dataset}/{algo}/{pair_name}: {e}")
                    continue
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/delta_summary.csv", index=False)
    
    # Save standard metrics separately
    standard_cols = ['dataset', 'algorithm', 'pair', 'accuracy_A', 'accuracy_B', 
                    'macro_precision_A', 'macro_precision_B', 'macro_f1_A', 'macro_f1_B']
    standard_df = results_df[standard_cols].copy()
    standard_df.to_csv(f"{output_dir}/standard_summary.csv", index=False)
    
    print(f"Benchmark completed! Results saved to {output_dir}/")
    print(f"Total experiments: {len(all_results)}")
    print(f"Average BAC: {results_df['bac'].mean():.3f}")
    print(f"Average DCE: {results_df['dce'].mean():.3f}")


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True) 