"""
Command-line interface for Delta-Audit.

This module provides the main CLI entry point for running experiments,
generating figures, and performing sanity checks.
"""

import argparse
import sys
from pathlib import Path

from .runners import run_benchmark, run_quickstart
from .plotting import make_overview_figure
from .io import load_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Delta-Audit: A lightweight Δ-Attribution suite for auditing model updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  delta-audit quickstart                    # Run a quick demonstration
  delta-audit run --config configs/full_benchmark.yaml  # Run full benchmark
  delta-audit figures --summary results/_summary --out figures/  # Generate figures
  delta-audit check                        # Run sanity checks
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # Quickstart command
    quickstart_parser = subparsers.add_parser(
        "quickstart", 
        help="Run a quick demonstration (5 minutes)"
    )
    quickstart_parser.add_argument(
        "--output-dir", 
        default="results", 
        help="Output directory for results (default: results)"
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run", 
        help="Run the full benchmark"
    )
    run_parser.add_argument(
        "--config", 
        required=True, 
        help="Path to configuration file"
    )
    run_parser.add_argument(
        "--output-dir", 
        default="results", 
        help="Output directory for results (default: results)"
    )
    
    # Figures command
    figures_parser = subparsers.add_parser(
        "figures", 
        help="Generate overview figures from results"
    )
    figures_parser.add_argument(
        "--summary", 
        required=True, 
        help="Path to summary results directory"
    )
    figures_parser.add_argument(
        "--out", 
        required=True, 
        help="Output directory for figures"
    )
    
    # Check command
    check_parser = subparsers.add_parser(
        "check", 
        help="Run sanity checks on results"
    )
    check_parser.add_argument(
        "--summary", 
        default="results/_summary", 
        help="Path to summary results directory (default: results/_summary)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "quickstart":
            print("🚀 Starting Delta-Audit quickstart...")
            run_quickstart(args.output_dir)
            
        elif args.command == "run":
            print("🔬 Starting Delta-Audit full benchmark...")
            run_benchmark(args.config, args.output_dir)
            
        elif args.command == "figures":
            print("📊 Generating Delta-Audit figures...")
            summary_path = Path(args.summary)
            out_path = Path(args.out)
            
            if not summary_path.exists():
                print(f"❌ Summary directory not found: {summary_path}")
                sys.exit(1)
            
            out_path.mkdir(parents=True, exist_ok=True)
            make_overview_figure(str(summary_path), str(out_path))
            print(f"✅ Figures saved to: {out_path}")
            
        elif args.command == "check":
            print("🔍 Running Delta-Audit sanity checks...")
            summary_path = Path(args.summary)
            
            if not summary_path.exists():
                print(f"❌ Summary directory not found: {summary_path}")
                sys.exit(1)
            
            # Load results
            delta_results = load_results(summary_path / "delta_summary.csv")
            standard_results = load_results(summary_path / "standard_summary.csv")
            
            if delta_results is None or standard_results is None:
                print("❌ Could not load results files")
                sys.exit(1)
            
            # Print headline statistics
            print("\n📈 Headline Statistics:")
            print(f"   Total experiments: {len(delta_results)}")
            print(f"   Datasets: {', '.join(delta_results['dataset'].unique())}")
            print(f"   Algorithms: {', '.join(delta_results['algorithm'].unique())}")
            print(f"   Pairs: {', '.join(delta_results['pair'].unique())}")
            
            print("\n🏆 Top Performers:")
            # Best BAC
            best_bac_idx = delta_results['bac'].idxmax()
            best_bac = delta_results.loc[best_bac_idx]
            print(f"   Best BAC: {best_bac['bac']:.3f} ({best_bac['dataset']}, {best_bac['algorithm']}, {best_bac['pair']})")
            
            # Lowest DCE
            best_dce_idx = delta_results['dce'].idxmin()
            best_dce = delta_results.loc[best_dce_idx]
            print(f"   Lowest DCE: {best_dce['dce']:.3f} ({best_dce['dataset']}, {best_dce['algorithm']}, {best_dce['pair']})")
            
            # Highest Delta Magnitude
            best_delta_idx = delta_results['delta_mag_l1'].idxmax()
            best_delta = delta_results.loc[best_delta_idx]
            print(f"   Highest Δ: {best_delta['delta_mag_l1']:.1f} ({best_delta['dataset']}, {best_delta['algorithm']}, {best_delta['pair']})")
            
            print("\n📊 Performance Impact:")
            standard_results['acc_delta'] = standard_results['accuracy_B'] - standard_results['accuracy_A']
            positive_changes = (standard_results['acc_delta'] > 0).sum()
            negative_changes = (standard_results['acc_delta'] < 0).sum()
            no_changes = (standard_results['acc_delta'] == 0).sum()
            
            print(f"   Improvements: {positive_changes} ({positive_changes/len(standard_results)*100:.1f}%)")
            print(f"   Degradations: {negative_changes} ({negative_changes/len(standard_results)*100:.1f}%)")
            print(f"   No change: {no_changes} ({no_changes/len(standard_results)*100:.1f}%)")
            
            print("\n✅ Sanity checks completed!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 