#!/usr/bin/env python3
"""
OpenMP Benchmark Results Analyzer
=================================

This script analyzes the results from the OpenMP benchmark suite and generates
visualizations and detailed performance analysis reports.

Author: Automated Benchmark Analyzer
Date: November 2024
"""

import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
import sys
from datetime import datetime


class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates reports and visualizations."""
    
    def __init__(self, results_file: str, output_dir: str = None):
        """Initialize the analyzer with results file."""
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir) if output_dir else self.results_file.parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        self.df = self.load_results()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self) -> pd.DataFrame:
        """Load benchmark results from CSV or JSON file."""
        print(f"Loading results from {self.results_file}")
        
        if self.results_file.suffix == '.json':
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.results_file.suffix == '.csv':
            df = pd.read_csv(self.results_file)
        else:
            raise ValueError("Results file must be CSV or JSON format")
        
        # Filter successful runs only
        successful_df = df[df['exit_code'] == 0].copy()
        
        print(f"Loaded {len(df)} total results, {len(successful_df)} successful runs")
        return successful_df
    
    def calculate_statistics(self):
        """Calculate performance statistics for each application and thread configuration."""
        print("Calculating performance statistics...")
        
        # Group by application and thread count
        stats = self.df.groupby(['application', 'threads']).agg({
            'wall_time': ['mean', 'std', 'min', 'max'],
            'cpu_time': ['mean', 'std'],
            'max_memory_kb': ['mean', 'max']
        }).round(4)
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        
        # Calculate speedup (relative to single thread)
        speedup_data = []
        
        for app in self.df['application'].unique():
            app_data = self.df[self.df['application'] == app]
            
            # Get baseline (1 thread) performance
            baseline = app_data[app_data['threads'] == 1]['wall_time'].mean()
            
            for threads in sorted(app_data['threads'].unique()):
                thread_time = app_data[app_data['threads'] == threads]['wall_time'].mean()
                speedup = baseline / thread_time if thread_time > 0 else 0
                efficiency = speedup / threads if threads > 0 else 0
                
                speedup_data.append({
                    'application': app,
                    'threads': threads,
                    'avg_time': thread_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                })
        
        self.speedup_df = pd.DataFrame(speedup_data)
        self.stats_df = stats
        
        return stats, self.speedup_df
    
    def plot_performance_by_threads(self):
        """Create performance plots by thread count for each application."""
        print("Generating performance by threads plots...")
        
        apps = self.df['application'].unique()
        n_apps = len(apps)
        
        # Create subplots
        fig, axes = plt.subplots(2, (n_apps + 1) // 2, figsize=(16, 10))
        if n_apps == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i, app in enumerate(apps):
            app_data = self.df[self.df['application'] == app]
            
            # Calculate means and std for each thread count
            thread_stats = app_data.groupby('threads')['wall_time'].agg(['mean', 'std']).reset_index()
            
            ax = axes[i]
            ax.errorbar(thread_stats['threads'], thread_stats['mean'], 
                       yerr=thread_stats['std'], marker='o', capsize=5, capthick=2)
            ax.set_title(f'{app} - Execution Time by Thread Count')
            ax.set_xlabel('Number of Threads')
            ax.set_ylabel('Wall Time (seconds)')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_by_threads.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_speedup_analysis(self):
        """Create speedup and efficiency analysis plots."""
        print("Generating speedup analysis plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Speedup plot
        for app in self.speedup_df['application'].unique():
            app_data = self.speedup_df[self.speedup_df['application'] == app]
            ax1.plot(app_data['threads'], app_data['speedup'], 
                    marker='o', label=app, linewidth=2)
        
        # Ideal speedup line
        max_threads = self.speedup_df['threads'].max()
        ideal_threads = range(1, max_threads + 1)
        ax1.plot(ideal_threads, ideal_threads, 'k--', alpha=0.5, label='Ideal Speedup')
        
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Speedup Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Efficiency plot
        for app in self.speedup_df['application'].unique():
            app_data = self.speedup_df[self.speedup_df['application'] == app]
            ax2.plot(app_data['threads'], app_data['efficiency'], 
                    marker='s', label=app, linewidth=2)
        
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Parallel Efficiency')
        ax2.set_title('Parallel Efficiency Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'speedup_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_usage(self):
        """Create memory usage analysis plots."""
        print("Generating memory usage plots...")
        
        if 'max_memory_kb' not in self.df.columns or self.df['max_memory_kb'].sum() == 0:
            print("No memory usage data available, skipping memory plots")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Memory usage by application and thread count
        memory_data = self.df.groupby(['application', 'threads'])['max_memory_kb'].mean().reset_index()
        
        for app in memory_data['application'].unique():
            app_data = memory_data[memory_data['application'] == app]
            ax.plot(app_data['threads'], app_data['max_memory_kb'] / 1024,  # Convert to MB
                   marker='o', label=app, linewidth=2)
        
        ax.set_xlabel('Number of Threads')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage by Thread Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_heatmap(self):
        """Create performance heatmap."""
        print("Generating performance heatmap...")
        
        # Create pivot table for heatmap
        heatmap_data = self.df.groupby(['application', 'threads'])['wall_time'].mean().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Wall Time (seconds)'})
        plt.title('Performance Heatmap - Wall Time by Application and Thread Count')
        plt.xlabel('Number of Threads')
        plt.ylabel('Application')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("Generating analysis report...")
        
        report_file = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("OpenMP Benchmark Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results File: {self.results_file}\n\n")
            
            # Basic statistics
            f.write("Dataset Overview:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total successful runs: {len(self.df)}\n")
            f.write(f"Applications tested: {', '.join(self.df['application'].unique())}\n")
            f.write(f"Thread configurations: {', '.join(map(str, sorted(self.df['threads'].unique())))}\n")
            f.write(f"Iterations per config: {len(self.df) // (len(self.df['application'].unique()) * len(self.df['threads'].unique()))}\n\n")
            
            # Performance summary
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            
            for app in self.df['application'].unique():
                app_data = self.df[self.df['application'] == app]
                f.write(f"\n{app}:\n")
                
                best_time = app_data['wall_time'].min()
                worst_time = app_data['wall_time'].max()
                avg_time = app_data['wall_time'].mean()
                
                f.write(f"  Best time: {best_time:.3f}s\n")
                f.write(f"  Worst time: {worst_time:.3f}s\n")
                f.write(f"  Average time: {avg_time:.3f}s\n")
                
                # Best thread configuration
                best_config = app_data.loc[app_data['wall_time'].idxmin()]
                f.write(f"  Best configuration: {best_config['threads']} threads\n")
            
            # Speedup analysis
            f.write("\nSpeedup Analysis:\n")
            f.write("-" * 17 + "\n")
            
            for app in self.speedup_df['application'].unique():
                app_speedup = self.speedup_df[self.speedup_df['application'] == app]
                max_speedup = app_speedup['speedup'].max()
                max_speedup_threads = app_speedup.loc[app_speedup['speedup'].idxmax(), 'threads']
                
                f.write(f"\n{app}:\n")
                f.write(f"  Maximum speedup: {max_speedup:.2f}x at {max_speedup_threads} threads\n")
                
                # Efficiency at maximum threads
                max_threads = app_speedup['threads'].max()
                efficiency_at_max = app_speedup[app_speedup['threads'] == max_threads]['efficiency'].iloc[0]
                f.write(f"  Efficiency at {max_threads} threads: {efficiency_at_max:.1%}\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            
            # Find optimal thread counts
            for app in self.speedup_df['application'].unique():
                app_speedup = self.speedup_df[self.speedup_df['application'] == app]
                
                # Find thread count with best efficiency > 0.5
                good_efficiency = app_speedup[app_speedup['efficiency'] >= 0.5]
                if not good_efficiency.empty:
                    optimal_threads = good_efficiency['threads'].max()
                    f.write(f"{app}: Use {optimal_threads} threads for optimal performance\n")
                else:
                    best_threads = app_speedup.loc[app_speedup['speedup'].idxmax(), 'threads']
                    f.write(f"{app}: Best performance at {best_threads} threads (low efficiency)\n")
        
        print(f"Analysis report saved to: {report_file}")
        return report_file
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive benchmark analysis...")
        
        # Calculate statistics
        stats, speedup = self.calculate_statistics()
        
        # Generate all plots
        self.plot_performance_by_threads()
        self.plot_speedup_analysis()
        self.plot_memory_usage()
        self.create_heatmap()
        
        # Generate report
        report_file = self.generate_report()
        
        # Save processed data
        stats_file = self.output_dir / "performance_statistics.csv"
        self.stats_df.to_csv(stats_file)
        
        speedup_file = self.output_dir / "speedup_analysis.csv"
        self.speedup_df.to_csv(speedup_file, index=False)
        
        print("\nAnalysis complete! Generated files:")
        print(f"  - Performance plots: {self.output_dir}")
        print(f"  - Analysis report: {report_file}")
        print(f"  - Statistics: {stats_file}")
        print(f"  - Speedup data: {speedup_file}")
        
        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze OpenMP benchmark results and generate visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 analyze_results.py results.csv
  python3 analyze_results.py results.json --output ./analysis
        """
    )
    
    parser.add_argument(
        "results_file",
        help="Path to benchmark results file (CSV or JSON)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for analysis results (default: ./analysis)"
    )
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file '{args.results_file}' not found")
        return 1
    
    try:
        # Check for required packages
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Error: Required package missing: {e}")
        print("Please install required packages with:")
        print("  pip3 install pandas matplotlib seaborn")
        return 1
    
    # Create analyzer and run analysis
    analyzer = BenchmarkAnalyzer(args.results_file, args.output)
    success = analyzer.run_full_analysis()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())