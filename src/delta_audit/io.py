"""
Input/Output utilities for Delta-Audit.

This module provides functions for loading and saving results, creating manifests,
and handling configuration files.
"""

from __future__ import annotations

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def load_results(file_path: Path) -> Optional[pd.DataFrame]:
    """Load results from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with results or None if file doesn't exist
    """
    try:
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_results(results: pd.DataFrame, file_path: Path) -> None:
    """Save results to a CSV file.
    
    Args:
        results: DataFrame to save
        file_path: Path where to save the file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(file_path, index=False)
    print(f"Results saved to: {file_path}")


def create_manifest(results_dir: Path, output_file: Path) -> None:
    """Create a manifest file listing all results and their metadata.
    
    Args:
        results_dir: Directory containing results
        output_file: Path to save the manifest
    """
    manifest = {
        "created_at": datetime.now().isoformat(),
        "results_directory": str(results_dir),
        "files": []
    }
    
    # Find all CSV files
    for csv_file in results_dir.rglob("*.csv"):
        relative_path = csv_file.relative_to(results_dir)
        file_info = {
            "path": str(relative_path),
            "size_bytes": csv_file.stat().st_size,
            "modified": datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat()
        }
        
        # Try to get basic info about the CSV
        try:
            df = pd.read_csv(csv_file)
            file_info["rows"] = len(df)
            file_info["columns"] = len(df.columns)
            file_info["column_names"] = list(df.columns)
        except Exception:
            file_info["rows"] = "unknown"
            file_info["columns"] = "unknown"
            file_info["column_names"] = []
        
        manifest["files"].append(file_info)
    
    # Save manifest
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to: {output_file}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path where to save the configuration
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {config_path}")


def get_default_config() -> Dict[str, Any]:
    """Get the default configuration for Delta-Audit experiments.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "datasets": ["breast_cancer", "wine", "digits"],
        "algo_pairs": {
            "logreg": [
                {
                    "A": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
                    "B": {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
                    "pair_name": "pair1"
                },
                {
                    "A": {"C": 1.0, "penalty": "l2", "solver": "liblinear"},
                    "B": {"C": 1.0, "penalty": "l1", "solver": "liblinear"},
                    "pair_name": "pair2"
                },
                {
                    "A": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
                    "B": {"C": 1.0, "penalty": "l2", "solver": "saga"},
                    "pair_name": "pair3"
                }
            ],
            "svc": [
                {
                    "A": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
                    "B": {"C": 1.0, "kernel": "linear", "gamma": "scale"},
                    "pair_name": "pair1"
                },
                {
                    "A": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
                    "B": {"C": 1.0, "kernel": "rbf", "gamma": "auto"},
                    "pair_name": "pair2"
                },
                {
                    "A": {"C": 1.0, "kernel": "poly", "degree": 3, "gamma": "scale"},
                    "B": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
                    "pair_name": "pair3"
                }
            ],
            "rf": [
                {
                    "A": {"n_estimators": 100, "max_depth": None, "max_features": None},
                    "B": {"n_estimators": 300, "max_depth": None, "max_features": None},
                    "pair_name": "pair1"
                },
                {
                    "A": {"n_estimators": 200, "max_depth": None, "max_features": None},
                    "B": {"n_estimators": 200, "max_depth": 5, "max_features": None},
                    "pair_name": "pair2"
                },
                {
                    "A": {"n_estimators": 200, "max_depth": None, "max_features": "sqrt"},
                    "B": {"n_estimators": 200, "max_depth": None, "max_features": "log2"},
                    "pair_name": "pair3"
                }
            ],
            "gb": [
                {
                    "A": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
                    "B": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
                    "pair_name": "pair1"
                },
                {
                    "A": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
                    "B": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
                    "pair_name": "pair2"
                },
                {
                    "A": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
                    "B": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5},
                    "pair_name": "pair3"
                }
            ],
            "knn": [
                {
                    "A": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
                    "B": {"n_neighbors": 10, "weights": "uniform", "algorithm": "auto"},
                    "pair_name": "pair1"
                },
                {
                    "A": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
                    "B": {"n_neighbors": 5, "weights": "distance", "algorithm": "auto"},
                    "pair_name": "pair2"
                },
                {
                    "A": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
                    "B": {"n_neighbors": 5, "weights": "uniform", "algorithm": "ball_tree"},
                    "pair_name": "pair3"
                }
            ]
        },
        "experiment_settings": {
            "test_size": 0.2,
            "random_state": 42,
            "baseline_method": "mean"
        },
        "output_settings": {
            "save_individual_results": True,
            "save_summary": True,
            "create_manifest": True
        }
    }


def validate_results(results: pd.DataFrame, expected_columns: list) -> bool:
    """Validate that results DataFrame has expected structure.
    
    Args:
        results: DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        True if validation passes, False otherwise
    """
    if results is None or results.empty:
        print("Results are empty or None")
        return False
    
    missing_columns = set(expected_columns) - set(results.columns)
    if missing_columns:
        print(f"Missing expected columns: {missing_columns}")
        return False
    
    # Check for required metadata columns
    required_metadata = ['dataset', 'algorithm', 'pair']
    missing_metadata = set(required_metadata) - set(results.columns)
    if missing_metadata:
        print(f"Missing required metadata columns: {missing_metadata}")
        return False
    
    return True


def export_to_latex(results: pd.DataFrame, output_file: Path, 
                   caption: str = "Delta-Audit Results") -> None:
    """Export results to LaTeX table format.
    
    Args:
        results: DataFrame to export
        output_file: Path to save the LaTeX table
        caption: Table caption
    """
    # Select key columns for LaTeX export
    key_columns = ['dataset', 'algorithm', 'pair', 'bac', 'dce', 'delta_mag_l1', 
                   'accuracy_A', 'accuracy_B']
    
    # Filter to available columns
    available_columns = [col for col in key_columns if col in results.columns]
    export_df = results[available_columns].copy()
    
    # Round numeric columns
    numeric_columns = export_df.select_dtypes(include=[np.number]).columns
    export_df[numeric_columns] = export_df[numeric_columns].round(3)
    
    # Generate LaTeX
    latex_table = export_df.to_latex(
        index=False,
        caption=caption,
        label="tab:delta-audit-results",
        float_format="%.3f"
    )
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to: {output_file}") 