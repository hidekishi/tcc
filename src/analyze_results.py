#!/usr/bin/env python3
"""
Benchmark Results Analyzer
Analyzes and visualizes OmpSCR benchmark results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import sys
from pathlib import Path

def load_results(file_path):
    """Load benchmark results from CSV or JSON file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix == '.json':
        with open(file_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("File must be CSV or JSON format")

def analyze_performance(df, output_dir="analysis_output"):
    """Generate performance analysis plots and reports"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Filter successful runs only
    df_success = df[df['success'] == True].copy()
    
    if df_success.empty:
        print("❌ No successful benchmark runs found!")
        return
    
    # Calculate average times per benchmark/thread combination
    avg_results = df_success.groupby(['benchmark', 'threads'])['wall_time'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance vs Thread Count
    plt.figure(figsize=(15, 10))
    
    benchmarks = df_success['benchmark'].unique()
    n_benchmarks = len(benchmarks)
    
    if n_benchmarks <= 6:
        rows, cols = 2, 3
    elif n_benchmarks <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 4
    
    for i, benchmark in enumerate(benchmarks):
        plt.subplot(rows, cols, i + 1)
        
        bench_data = avg_results[avg_results['benchmark'] == benchmark]
        
        plt.errorbar(bench_data['threads'], bench_data['mean'], 
                    yerr=bench_data['std'], marker='o', capsize=5)
        plt.xlabel('Number of Threads')
        plt.ylabel('Time (seconds)')
        plt.title(f'{benchmark}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_threads.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Speedup Analysis
    plt.figure(figsize=(15, 10))
    
    for i, benchmark in enumerate(benchmarks):
        plt.subplot(rows, cols, i + 1)
        
        bench_data = avg_results[avg_results['benchmark'] == benchmark].copy()
        
        if not bench_data.empty:
            # Calculate speedup relative to single thread
            single_thread_time = bench_data[bench_data['threads'] == 1]['mean'].iloc[0] if len(bench_data[bench_data['threads'] == 1]) > 0 else None
            
            if single_thread_time:
                bench_data['speedup'] = single_thread_time / bench_data['mean']
                
                plt.plot(bench_data['threads'], bench_data['speedup'], 'o-', label='Actual')
                plt.plot(bench_data['threads'], bench_data['threads'], '--', alpha=0.5, label='Ideal')
                plt.xlabel('Number of Threads')
                plt.ylabel('Speedup')
                plt.title(f'{benchmark} - Speedup')
                plt.grid(True, alpha=0.3)
                plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Parallel Efficiency
    plt.figure(figsize=(15, 10))
    
    for i, benchmark in enumerate(benchmarks):
        plt.subplot(rows, cols, i + 1)
        
        bench_data = avg_results[avg_results['benchmark'] == benchmark].copy()
        
        if not bench_data.empty:
            single_thread_time = bench_data[bench_data['threads'] == 1]['mean'].iloc[0] if len(bench_data[bench_data['threads'] == 1]) > 0 else None
            
            if single_thread_time:
                bench_data['speedup'] = single_thread_time / bench_data['mean']
                bench_data['efficiency'] = bench_data['speedup'] / bench_data['threads']
                
                plt.plot(bench_data['threads'], bench_data['efficiency'], 'o-')
                plt.xlabel('Number of Threads')
                plt.ylabel('Parallel Efficiency')
                plt.title(f'{benchmark} - Efficiency')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parallel_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Heatmap
    if len(benchmarks) > 1:
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        heatmap_data = avg_results.pivot(index='benchmark', columns='threads', values='mean')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd_r', 
                   cbar_kws={'label': 'Time (seconds)'})
        plt.title('Performance Heatmap (Average Time)')
        plt.ylabel('Benchmark')
        plt.xlabel('Number of Threads')
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Generate detailed report
    with open(output_dir / 'detailed_analysis.txt', 'w') as f:
        f.write("OmpSCR Benchmark Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write("-" * 20 + "\n")
        total_runs = len(df)
        successful_runs = len(df_success)
        f.write(f"Total runs: {total_runs}\n")
        f.write(f"Successful runs: {successful_runs} ({successful_runs/total_runs*100:.1f}%)\n")
        f.write(f"Unique benchmarks: {len(benchmarks)}\n")
        f.write(f"Thread counts tested: {sorted(df_success['threads'].unique())}\n\n")
        
        f.write("PERFORMANCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        for benchmark in benchmarks:
            f.write(f"\n{benchmark}:\n")
            bench_data = avg_results[avg_results['benchmark'] == benchmark]
            
            # Find optimal thread count
            optimal_threads = bench_data.loc[bench_data['mean'].idxmin(), 'threads']
            best_time = bench_data.loc[bench_data['mean'].idxmin(), 'mean']
            
            f.write(f"  Optimal thread count: {optimal_threads}\n")
            f.write(f"  Best average time: {best_time:.3f}s\n")
            
            # Calculate speedup if single-thread data available
            single_thread_data = bench_data[bench_data['threads'] == 1]
            if not single_thread_data.empty:
                single_time = single_thread_data['mean'].iloc[0]
                max_speedup = single_time / best_time
                f.write(f"  Maximum speedup: {max_speedup:.2f}x\n")
                
                # Efficiency at optimal thread count
                if optimal_threads > 1:
                    efficiency = max_speedup / optimal_threads
                    f.write(f"  Efficiency at optimal threads: {efficiency:.1%}\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 20 + "\n")
        
        # Identify benchmarks that scale well
        scalable_benchmarks = []
        for benchmark in benchmarks:
            bench_data = avg_results[avg_results['benchmark'] == benchmark]
            if len(bench_data) > 1:
                single_thread_data = bench_data[bench_data['threads'] == 1]
                if not single_thread_data.empty:
                    single_time = single_thread_data['mean'].iloc[0]
                    best_time = bench_data['mean'].min()
                    speedup = single_time / best_time
                    if speedup > 2.0:  # Good speedup
                        scalable_benchmarks.append((benchmark, speedup))
        
        if scalable_benchmarks:
            f.write("Benchmarks with good parallel scaling (>2x speedup):\n")
            for benchmark, speedup in sorted(scalable_benchmarks, key=lambda x: x[1], reverse=True):
                f.write(f"  - {benchmark}: {speedup:.2f}x speedup\n")
        
    print(f"📊 Analysis completed! Results saved to: {output_dir}")
    print(f"   📈 Performance plots: {output_dir}/*.png")
    print(f"   📋 Detailed report: {output_dir}/detailed_analysis.txt")

def main():
    parser = argparse.ArgumentParser(description='Analyze OmpSCR benchmark results')
    parser.add_argument('results_file', help='Path to CSV or JSON results file')
    parser.add_argument('--output', default='analysis_output', 
                       help='Output directory for analysis (default: analysis_output)')
    
    args = parser.parse_args()
    
    try:
        # Check if matplotlib is available
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        print(f"📊 Loading results from: {args.results_file}")
        df = load_results(args.results_file)
        
        print(f"📈 Analyzing {len(df)} benchmark runs...")
        analyze_performance(df, args.output)
        
    except ImportError:
        print("⚠️  Warning: matplotlib and/or seaborn not available.")
        print("   Install with: pip3 install matplotlib seaborn")
        print("   For now, showing basic analysis...")
        
        # Basic analysis without plots
        df = load_results(args.results_file)
        df_success = df[df['success'] == True]
        
        print(f"\n📊 Basic Analysis:")
        print(f"   Total runs: {len(df)}")
        print(f"   Successful: {len(df_success)}")
        print(f"   Success rate: {len(df_success)/len(df)*100:.1f}%")
        
        if not df_success.empty:
            print(f"\n⚡ Performance Summary:")
            summary = df_success.groupby(['benchmark', 'threads'])['wall_time'].agg(['mean', 'std']).reset_index()
            for benchmark in df_success['benchmark'].unique():
                bench_data = summary[summary['benchmark'] == benchmark]
                print(f"\n{benchmark}:")
                for _, row in bench_data.iterrows():
                    print(f"  {row['threads']:2d} threads: {row['mean']:.3f}s ± {row['std']:.3f}s")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
