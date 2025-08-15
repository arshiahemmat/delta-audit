"""
Delta-Audit: A lightweight Δ-Attribution suite for auditing model updates.

This package provides tools for computing Δ-Attribution metrics between model pairs (A→B),
including behavioral alignment, conservation error, and stability measures.
"""

__version__ = "0.1.0"
__author__ = "Arshia Hemmat"

from .metrics import (
    compute_delta_magnitude_l1,
    compute_delta_topk_frac,
    compute_delta_entropy,
    compute_rank_overlap_at_k,
    compute_js_divergence,
    compute_dce,
    compute_bac,
    compute_codf,
    compute_stability,
    compute_grouped_occlusion_ratio,
)

from .explainers import (
    compute_occlusion_attributions,
    compute_clamping_attributions,
    compute_common_class_anchor,
    compute_grouped_occlusion,
)

from .runners import (
    run_benchmark,
    run_quickstart,
    train_model_pair,
    evaluate_model_pair,
)

from .plotting import (
    make_overview_figure,
    plot_bac_vs_dce,
    plot_algorithm_comparison,
    plot_dataset_heatmap,
    plot_performance_impact,
)

from .io import (
    load_results,
    save_results,
    create_manifest,
    load_config,
)

__all__ = [
    # Metrics
    "compute_delta_magnitude_l1",
    "compute_delta_topk_frac", 
    "compute_delta_entropy",
    "compute_rank_overlap_at_k",
    "compute_js_divergence",
    "compute_dce",
    "compute_bac",
    "compute_codf",
    "compute_stability",
    "compute_grouped_occlusion_ratio",
    # Explainers
    "compute_occlusion_attributions",
    "compute_clamping_attributions",
    "compute_common_class_anchor",
    "compute_grouped_occlusion",
    # Runners
    "run_benchmark",
    "run_quickstart",
    "train_model_pair",
    "evaluate_model_pair",
    # Plotting
    "make_overview_figure",
    "plot_bac_vs_dce",
    "plot_algorithm_comparison",
    "plot_dataset_heatmap",
    "plot_performance_impact",
    # IO
    "load_results",
    "save_results",
    "create_manifest",
    "load_config",
] 