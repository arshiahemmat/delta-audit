"""
Plotting utilities for Delta-Audit.

This module provides functions for generating overview figures and various
visualizations of Δ-Attribution results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from .io import load_results


def setup_plotting_style():
    """Set up consistent plotting style for all figures."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set font sizes
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14


def plot_bac_vs_dce(delta_results: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot BAC vs DCE scatter plot.
    
    Args:
        delta_results: DataFrame with BAC and DCE columns
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot with algorithm colors
    algorithms = delta_results['algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    for i, algo in enumerate(algorithms):
        mask = delta_results['algorithm'] == algo
        ax.scatter(delta_results.loc[mask, 'dce'], 
                  delta_results.loc[mask, 'bac'],
                  c=[colors[i]], label=algo, alpha=0.7, s=60)
    
    ax.set_xlabel('Differential Conservation Error (DCE)')
    ax.set_ylabel('Behavioral Alignment Coefficient (BAC)')
    ax.set_title('BAC vs DCE by Algorithm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_algorithm_comparison(delta_results: pd.DataFrame, metric: str, 
                            save_path: Optional[str] = None) -> None:
    """Plot algorithm comparison for a specific metric.
    
    Args:
        delta_results: DataFrame with results
        metric: Metric name to plot
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by algorithm and compute mean
    algo_means = delta_results.groupby('algorithm')[metric].mean().sort_values()
    
    bars = ax.bar(range(len(algo_means)), algo_means.values, 
                  color=sns.color_palette("husl", len(algo_means)))
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} by Algorithm')
    ax.set_xticks(range(len(algo_means)))
    ax.set_xticklabels(algo_means.index, rotation=45)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, algo_means.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_dataset_heatmap(delta_results: pd.DataFrame, metric: str,
                        save_path: Optional[str] = None) -> None:
    """Plot heatmap of metric values across datasets and algorithms.
    
    Args:
        delta_results: DataFrame with results
        metric: Metric name to plot
        save_path: Optional path to save the figure
    """
    # Pivot data for heatmap
    pivot_data = delta_results.pivot_table(
        values=metric, 
        index='dataset', 
        columns='algorithm', 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    ax.set_title(f'{metric.replace("_", " ").title()} Heatmap')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_impact(standard_results: pd.DataFrame,
                          save_path: Optional[str] = None) -> None:
    """Plot performance impact analysis.
    
    Args:
        standard_results: DataFrame with standard metrics
        save_path: Optional path to save the figure
    """
    # Compute performance changes
    standard_results['acc_delta'] = standard_results['accuracy_B'] - standard_results['accuracy_A']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of accuracy changes
    ax1.hist(standard_results['acc_delta'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='No change')
    ax1.set_xlabel('Accuracy Change (B - A)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Accuracy Changes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot by algorithm
    standard_results.boxplot(column='acc_delta', by='algorithm', ax=ax2)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Accuracy Change (B - A)')
    ax2.set_title('Accuracy Changes by Algorithm')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def make_overview_figure(summary_dir: str, output_dir: str) -> None:
    """Generate comprehensive overview figure from results.
    
    Args:
        summary_dir: Directory containing summary CSV files
        output_dir: Directory to save generated figures
    """
    setup_plotting_style()
    
    # Load results
    delta_results = load_results(Path(summary_dir) / "delta_summary.csv")
    standard_results = load_results(Path(summary_dir) / "standard_summary.csv")
    
    if delta_results is None or standard_results is None:
        raise ValueError("Could not load results files")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate individual plots
    print("Generating BAC vs DCE plot...")
    plot_bac_vs_dce(delta_results, str(output_path / "fig1_bac_vs_dce.png"))
    
    print("Generating algorithm comparison plots...")
    plot_algorithm_comparison(delta_results, 'bac', str(output_path / "fig2_bars_bac_by_algo.png"))
    plot_algorithm_comparison(delta_results, 'dce', str(output_path / "fig3_bars_dce_by_algo.png"))
    plot_algorithm_comparison(delta_results, 'delta_mag_l1', str(output_path / "fig4_bars_deltamag_by_algo.png"))
    
    print("Generating dataset heatmaps...")
    plot_dataset_heatmap(delta_results, 'bac', str(output_path / "fig_dataset_heatmap_BAC.png"))
    plot_dataset_heatmap(delta_results, 'dce', str(output_path / "fig_dataset_heatmap_DCE.png"))
    plot_dataset_heatmap(delta_results, 'delta_mag_l1', str(output_path / "fig_dataset_heatmap_DeltaMag_L1.png"))
    
    print("Generating performance impact analysis...")
    plot_performance_impact(standard_results, str(output_path / "fig_performance_impact.png"))
    
    # Generate comprehensive overview figure
    print("Generating comprehensive overview figure...")
    create_comprehensive_overview(delta_results, standard_results, str(output_path / "fig0_overview.png"))
    
    print(f"All figures saved to: {output_path}")


def create_comprehensive_overview(delta_results: pd.DataFrame, 
                                standard_results: pd.DataFrame,
                                save_path: str) -> None:
    """Create a comprehensive 6-panel overview figure.
    
    Args:
        delta_results: DataFrame with Δ-Attribution metrics
        standard_results: DataFrame with standard metrics
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: BAC vs DCE scatter
    ax1 = fig.add_subplot(gs[0, 0])
    algorithms = delta_results['algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    for i, algo in enumerate(algorithms):
        mask = delta_results['algorithm'] == algo
        ax1.scatter(delta_results.loc[mask, 'dce'], 
                   delta_results.loc[mask, 'bac'],
                   c=[colors[i]], label=algo, alpha=0.7, s=50)
    
    ax1.set_xlabel('DCE')
    ax1.set_ylabel('BAC')
    ax1.set_title('BAC vs DCE')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: BAC by algorithm
    ax2 = fig.add_subplot(gs[0, 1])
    bac_means = delta_results.groupby('algorithm')['bac'].mean().sort_values()
    bars = ax2.bar(range(len(bac_means)), bac_means.values, 
                   color=sns.color_palette("husl", len(bac_means)))
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('BAC')
    ax2.set_title('BAC by Algorithm')
    ax2.set_xticks(range(len(bac_means)))
    ax2.set_xticklabels(bac_means.index, rotation=45)
    
    # Panel 3: DCE by algorithm
    ax3 = fig.add_subplot(gs[0, 2])
    dce_means = delta_results.groupby('algorithm')['dce'].mean().sort_values()
    bars = ax3.bar(range(len(dce_means)), dce_means.values, 
                   color=sns.color_palette("husl", len(dce_means)))
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('DCE')
    ax3.set_title('DCE by Algorithm')
    ax3.set_xticks(range(len(dce_means)))
    ax3.set_xticklabels(dce_means.index, rotation=45)
    
    # Panel 4: Delta Magnitude by dataset
    ax4 = fig.add_subplot(gs[1, 0])
    delta_means = delta_results.groupby('dataset')['delta_mag_l1'].mean().sort_values()
    bars = ax4.bar(range(len(delta_means)), delta_means.values, 
                   color=sns.color_palette("husl", len(delta_means)))
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Δ Magnitude L1')
    ax4.set_title('Δ Magnitude by Dataset')
    ax4.set_xticks(range(len(delta_means)))
    ax4.set_xticklabels(delta_means.index, rotation=45)
    
    # Panel 5: Performance impact
    ax5 = fig.add_subplot(gs[1, 1])
    standard_results['acc_delta'] = standard_results['accuracy_B'] - standard_results['accuracy_A']
    ax5.hist(standard_results['acc_delta'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Accuracy Change')
    ax5.set_ylabel('Count')
    ax5.set_title('Performance Impact')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Key statistics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Compute key statistics
    total_experiments = len(delta_results)
    avg_bac = delta_results['bac'].mean()
    avg_dce = delta_results['dce'].mean()
    avg_delta = delta_results['delta_mag_l1'].mean()
    
    positive_changes = (standard_results['acc_delta'] > 0).sum()
    negative_changes = (standard_results['acc_delta'] < 0).sum()
    
    stats_text = f"""Key Statistics:
    
Total Experiments: {total_experiments}
Average BAC: {avg_bac:.3f}
Average DCE: {avg_dce:.3f}
Average Δ Magnitude: {avg_delta:.1f}

Performance Changes:
Improvements: {positive_changes}
Degradations: {negative_changes}

Top Algorithm (BAC): {delta_results.loc[delta_results['bac'].idxmax(), 'algorithm']}
Top Dataset (Δ): {delta_results.loc[delta_results['delta_mag_l1'].idxmax(), 'dataset']}"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 