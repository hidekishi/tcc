#!/usr/bin/env python3
"""
OmpSCR Benchmark Runner
Automated benchmark execution with varying thread counts and parameter analysis
"""

import os
import sys
import subprocess
import time
import json
import csv
from datetime import datetime
from pathlib import Path
import argparse
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class EmailNotifier:
    """Handle email notifications for benchmark results"""
    
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.enabled = False
        self.sender_email = None
        self.sender_password = None
        self.recipient_emails = []
    
    def configure(self, sender_email, sender_password, recipient_emails):
        """Configure email settings"""
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails if isinstance(recipient_emails, list) else [recipient_emails]
        self.enabled = True
        
    def send_notification(self, subject, body, attachments=None):
        """Send email notification with results"""
        if not self.enabled:
            print("📧 Email notifications disabled")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if Path(file_path).exists():
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {Path(file_path).name}'
                        )
                        msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.recipient_emails, text)
            server.quit()
            
            print(f"📧 Email sent successfully to: {', '.join(self.recipient_emails)}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False

class BenchmarkAnalyzer:
    """Integrated benchmark analysis and visualization"""
    
    @staticmethod
    def analyze_performance(results, output_dir="analysis_output"):
        """Generate performance analysis plots and reports from results data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame-like structure for analysis
        df_data = []
        for result in results:
            if result.get('success', False):
                df_data.append({
                    'benchmark': result['benchmark'],
                    'threads': result['threads'],
                    'wall_time': result['wall_time'],
                    'problem_size': result.get('problem_size', 'unknown'),
                    'iteration': result.get('iteration', 1)
                })
        
        if not df_data:
            print("❌ No successful benchmark runs found for analysis!")
            return None
            
        # Try to import plotting libraries
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = pd.DataFrame(df_data)
            return BenchmarkAnalyzer._generate_plots_and_report(df, output_dir)
            
        except ImportError:
            print("⚠️  matplotlib/seaborn not available. Generating basic text analysis...")
            return BenchmarkAnalyzer._generate_basic_analysis(df_data, output_dir)
    
    @staticmethod
    def _generate_plots_and_report(df, output_dir):
        """Generate full analysis with plots"""
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Calculate average times per benchmark/thread combination
        avg_results = df.groupby(['benchmark', 'threads'])['wall_time'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for non-interactive use
        import matplotlib
        matplotlib.use('Agg')
        
        plots_generated = []
        
        try:
            # 1. Performance vs Thread Count
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('OmpSCR Benchmark Analysis', fontsize=16, fontweight='bold')
            
            # Performance vs threads plot
            benchmarks = df['benchmark'].unique()
            for i, benchmark in enumerate(benchmarks):
                bench_data = avg_results[avg_results['benchmark'] == benchmark]
                color = sns.color_palette("husl", len(benchmarks))[i]
                axes[0,0].plot(bench_data['threads'], bench_data['mean'], 'o-', 
                             label=benchmark, color=color, linewidth=2, markersize=6)
                axes[0,0].fill_between(bench_data['threads'], 
                                     bench_data['mean'] - bench_data['std'],
                                     bench_data['mean'] + bench_data['std'],
                                     alpha=0.2, color=color)
            
            axes[0,0].set_xlabel('Number of Threads')
            axes[0,0].set_ylabel('Execution Time (seconds)')
            axes[0,0].set_title('Performance vs Thread Count')
            axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_xscale('log', base=2)
            
            # Speedup analysis
            for i, benchmark in enumerate(benchmarks):
                bench_data = avg_results[avg_results['benchmark'] == benchmark]
                if len(bench_data) > 1:
                    single_thread_data = bench_data[bench_data['threads'] == 1]
                    if not single_thread_data.empty:
                        baseline = single_thread_data['mean'].iloc[0]
                        speedup = baseline / bench_data['mean']
                        color = sns.color_palette("husl", len(benchmarks))[i]
                        axes[0,1].plot(bench_data['threads'], speedup, 'o-', 
                                     label=benchmark, color=color, linewidth=2, markersize=6)
            
            # Add ideal speedup line
            max_threads = df['threads'].max()
            ideal_threads = [t for t in range(1, max_threads + 1) if t <= max_threads]
            axes[0,1].plot(ideal_threads, ideal_threads, '--', color='black', 
                         label='Ideal Speedup', alpha=0.7)
            
            axes[0,1].set_xlabel('Number of Threads')
            axes[0,1].set_ylabel('Speedup Factor')
            axes[0,1].set_title('Speedup Analysis')
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_xscale('log', base=2)
            
            # Parallel efficiency
            for i, benchmark in enumerate(benchmarks):
                bench_data = avg_results[avg_results['benchmark'] == benchmark]
                if len(bench_data) > 1:
                    single_thread_data = bench_data[bench_data['threads'] == 1]
                    if not single_thread_data.empty:
                        baseline = single_thread_data['mean'].iloc[0]
                        efficiency = (baseline / bench_data['mean']) / bench_data['threads'] * 100
                        color = sns.color_palette("husl", len(benchmarks))[i]
                        axes[1,0].plot(bench_data['threads'], efficiency, 'o-', 
                                     label=benchmark, color=color, linewidth=2, markersize=6)
            
            axes[1,0].set_xlabel('Number of Threads')
            axes[1,0].set_ylabel('Parallel Efficiency (%)')
            axes[1,0].set_title('Parallel Efficiency')
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_xscale('log', base=2)
            axes[1,0].set_ylim(0, 110)
            
            # Performance heatmap
            heatmap_data = avg_results.pivot(index='benchmark', columns='threads', values='mean')
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[1,1], cbar_kws={'label': 'Execution Time (s)'})
            axes[1,1].set_title('Performance Heatmap')
            axes[1,1].set_xlabel('Number of Threads')
            axes[1,1].set_ylabel('Benchmark')
            
            plt.tight_layout()
            plot_file = output_dir / 'comprehensive_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append(plot_file)
            
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
        
        # Generate detailed text report
        report_file = output_dir / 'detailed_analysis.txt'
        BenchmarkAnalyzer._generate_text_report(df, avg_results, report_file)
        
        return {
            'plots': plots_generated,
            'report': report_file,
            'summary': BenchmarkAnalyzer._get_analysis_summary(df, avg_results)
        }
    
    @staticmethod
    def _generate_basic_analysis(df_data, output_dir):
        """Generate basic analysis without plots"""
        # Simple grouping and analysis
        benchmark_stats = {}
        for result in df_data:
            key = (result['benchmark'], result['threads'])
            if key not in benchmark_stats:
                benchmark_stats[key] = []
            benchmark_stats[key].append(result['wall_time'])
        
        # Calculate averages
        avg_results = []
        for (benchmark, threads), times in benchmark_stats.items():
            avg_results.append({
                'benchmark': benchmark,
                'threads': threads,
                'mean': sum(times) / len(times),
                'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times)
            })
        
        # Generate text report
        report_file = output_dir / 'basic_analysis.txt'
        with open(report_file, 'w') as f:
            f.write("OmpSCR Benchmark Analysis - Basic Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total successful runs: {len(df_data)}\n\n")
            
            f.write("Performance Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Benchmark':<20} {'Threads':<8} {'Avg Time':<10} {'Std Dev':<10}\n")
            f.write("-" * 60 + "\n")
            
            for result in sorted(avg_results, key=lambda x: (x['benchmark'], x['threads'])):
                f.write(f"{result['benchmark']:<20} {result['threads']:<8} "
                       f"{result['mean']:<10.3f} {result['std']:<10.3f}\n")
        
        return {
            'plots': [],
            'report': report_file,
            'summary': f"Basic analysis completed. {len(df_data)} successful runs analyzed."
        }
    
    @staticmethod
    def _generate_text_report(df, avg_results, report_file):
        """Generate detailed text analysis report"""
        with open(report_file, 'w') as f:
            f.write("OmpSCR Benchmark Analysis - Detailed Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total successful runs: {len(df)}\n")
            f.write(f"Benchmarks analyzed: {len(df['benchmark'].unique())}\n")
            f.write(f"Thread counts tested: {sorted(df['threads'].unique())}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Benchmark':<20} {'Threads':<8} {'Avg Time':<10} {'Std Dev':<10} {'Min Time':<10} {'Max Time':<10}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in avg_results.iterrows():
                f.write(f"{row['benchmark']:<20} {row['threads']:<8} "
                       f"{row['mean']:<10.3f} {row['std']:<10.3f} "
                       f"{row['min']:<10.3f} {row['max']:<10.3f}\n")
            
            # Speedup analysis
            f.write("\nSPEEDUP ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            benchmarks = df['benchmark'].unique()
            for benchmark in sorted(benchmarks):
                f.write(f"\n{benchmark}:\n")
                bench_data = avg_results[avg_results['benchmark'] == benchmark].sort_values('threads')
                
                single_thread_data = bench_data[bench_data['threads'] == 1]
                if not single_thread_data.empty:
                    baseline = single_thread_data['mean'].iloc[0]
                    
                    for _, row in bench_data.iterrows():
                        speedup = baseline / row['mean']
                        efficiency = speedup / row['threads'] * 100
                        f.write(f"  {row['threads']:2d} threads: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency\n")
                else:
                    f.write("  No single-thread baseline available\n")
    
    @staticmethod
    def _get_analysis_summary(df, avg_results):
        """Get a brief summary of the analysis"""
        benchmarks = len(df['benchmark'].unique())
        thread_counts = len(df['threads'].unique())
        total_runs = len(df)
        
        # Find best performing configurations
        best_configs = []
        for benchmark in df['benchmark'].unique():
            bench_data = avg_results[avg_results['benchmark'] == benchmark]
            if not bench_data.empty:
                best_row = bench_data.loc[bench_data['mean'].idxmin()]
                best_configs.append(f"{benchmark}: {best_row['threads']} threads ({best_row['mean']:.3f}s)")
        
        summary = f"""
Analysis Summary:
- {benchmarks} benchmarks analyzed
- {thread_counts} different thread counts tested  
- {total_runs} successful benchmark runs
- Best configurations: {', '.join(best_configs[:3])}{'...' if len(best_configs) > 3 else ''}
"""
        return summary

class BenchmarkRunner:
    def __init__(self, output_dir="benchmark_results", check_integrity=False, integrity_threshold=0.1, show_cpu_usage=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize email notifier and analyzer
        self.email_notifier = EmailNotifier()
        self.analyzer = BenchmarkAnalyzer()
        
        # Analysis settings
        self.auto_analyze = False
        self.analysis_output_dir = None
        self.check_integrity_detailed = check_integrity
        self.integrity_threshold = integrity_threshold
        self.show_cpu_usage = show_cpu_usage
        
        # Problem sizes: 5 níveis otimizados para análise de escalabilidade
        # Tamanhos otimizados para i9-14900K (24 cores, 32 threads) com 128GB RAM
        # Foco em workloads que estressam paralelização e aproveitam memória disponível
        self.problem_sizes = {
            # grid_size: Jacobi/iterative solvers | iterations: loop iterations
            # array_size: Pi/Mandelbrot points | fft_size: FFT problem size
            # md_particles/md_steps: Molecular Dynamics | qsort_size: QuickSort size in KB
            # fft_size_kb: FFT size in KB | lu_size: LU matrix size
            'small': {
                'grid_size': 2048, 'iterations': 500, 'array_size': 2000000, 'fft_size': 16384,
                'md_particles': 8192, 'md_steps': 50, 'qsort_size': 2000, 'fft_size_kb': 16, 'lu_size': 512
            },        # ~32 MB
            'medium': {
                'grid_size': 4096, 'iterations': 1000, 'array_size': 8000000, 'fft_size': 65536,
                'md_particles': 16384, 'md_steps': 100, 'qsort_size': 8000, 'fft_size_kb': 64, 'lu_size': 1024
            },     # ~128 MB
            'large': {
                'grid_size': 8192, 'iterations': 2000, 'array_size': 32000000, 'fft_size': 262144,
                'md_particles': 32768, 'md_steps': 200, 'qsort_size': 32000, 'fft_size_kb': 256, 'lu_size': 2048
            },    # ~512 MB
            'huge': {
                'grid_size': 16384, 'iterations': 5000, 'array_size': 128000000, 'fft_size': 1048576,
                'md_particles': 65536, 'md_steps': 500, 'qsort_size': 128000, 'fft_size_kb': 1024, 'lu_size': 4096
            },  # ~2 GB
            'extreme': {
                'grid_size': 32768, 'iterations': 10000, 'array_size': 512000000, 'fft_size': 4194304,
                'md_particles': 131072, 'md_steps': 1000, 'qsort_size': 512000, 'fft_size_kb': 4096, 'lu_size': 8192
            } # ~8 GB
        }
        
        # Results storage
        self.results = []
        
        # Progress tracking
        self.total_runs = 0
        self.completed_runs = 0
        self.start_time = None
        self.current_benchmark = ""
        self.current_config = ""
        
        # Progress file for monitoring
        self.progress_file = None
        
        # Available benchmarks and their test parameters
        self.benchmarks = {
            # Core computational benchmarks
            'c_pi': {
                'binary': 'bin/c_pi.par.gnu',
                'args_template': {
                    'small': ['{array_size}'],
                    'medium': ['{array_size}'],
                    'large': ['{array_size}'],
                    'huge': ['{array_size}'],
                    'extreme': ['{array_size}'],
                },
                'description': 'Pi calculation using numerical integration'
            },
            'c_pi_fine': {
                'binary': 'bin/c_pi_fine.par.gnu',
                'args_template': {
                    'small': ['{array_size}'],
                    'medium': ['{array_size}'],
                    'large': ['{array_size}'],
                    'huge': ['{array_size}'],
                    'extreme': ['{array_size}'],
                },
                'description': 'Pi calculation - Fine-grained version (dynamic scheduling)'
            },
            'c_pi_coarse': {
                'binary': 'bin/c_pi_coarse.par.gnu',
                'args_template': {
                    'small': ['{array_size}'],
                    'medium': ['{array_size}'],
                    'large': ['{array_size}'],
                    'huge': ['{array_size}'],
                    'extreme': ['{array_size}'],
                },
                'description': 'Pi calculation - Coarse-grained version (large static chunks)'
            },
            'c_mandel': {
                'binary': 'bin/c_mandel.par.gnu', 
                'args_template': {
                    'small': ['{array_size}'],
                    'medium': ['{array_size}'],
                    'large': ['{array_size}'],
                    'huge': ['{array_size}'],
                    'extreme': ['{array_size}'],
                },
                'description': 'Mandelbrot set generator'
            },
            'c_mandel_fine': {
                'binary': 'bin/c_mandel_fine.par.gnu', 
                'args_template': {
                    'small': ['{array_size}'],
                    'medium': ['{array_size}'],
                    'large': ['{array_size}'],
                    'huge': ['{array_size}'],
                    'extreme': ['{array_size}'],
                },
                'description': 'Mandelbrot set - Fine-grained (dynamic load balancing)'
            },
            'c_mandel_coarse': {
                'binary': 'bin/c_mandel_coarse.par.gnu', 
                'args_template': {
                    'small': ['{array_size}'],
                    'medium': ['{array_size}'],
                    'large': ['{array_size}'],
                    'huge': ['{array_size}'],
                    'extreme': ['{array_size}'],
                },
                'description': 'Mandelbrot set - Coarse-grained (static large chunks)'
            },
            'c_qsort': {
                'binary': 'bin/c_qsort.par.gnu',
                'args_template': {
                    'small': ['{qsort_size}'],
                    'medium': ['{qsort_size}'],
                    'large': ['{qsort_size}'],
                    'huge': ['{qsort_size}'],
                    'extreme': ['{qsort_size}'],
                },
                'description': 'Parallel quicksort'
            },
            'c_qsort_fine': {
                'binary': 'bin/c_qsort_fine.par.gnu',
                'args_template': {
                    'small': ['{qsort_size}'],
                    'medium': ['{qsort_size}'],
                    'large': ['{qsort_size}'],
                    'huge': ['{qsort_size}'],
                    'extreme': ['{qsort_size}'],
                },
                'description': 'Parallel quicksort - Fine-grained (task cutoff 1k elements)'
            },
            'c_qsort_coarse': {
                'binary': 'bin/c_qsort_coarse.par.gnu',
                'args_template': {
                    'small': ['{qsort_size}'],
                    'medium': ['{qsort_size}'],
                    'large': ['{qsort_size}'],
                    'huge': ['{qsort_size}'],
                    'extreme': ['{qsort_size}'],
                },
                'description': 'Parallel quicksort - Coarse-grained (task cutoff 100k elements)'
            },
            'c_fft': {
                'binary': 'bin/c_fft.par.gnu',
                'args_template': {
                    'small': ['{fft_size_kb}'],
                    'medium': ['{fft_size_kb}'],
                    'large': ['{fft_size_kb}'],
                    'huge': ['{fft_size_kb}'],
                    'extreme': ['{fft_size_kb}'],
                },
                'description': 'Fast Fourier Transform'
            },
            'c_fft6': {
                'binary': 'bin/c_fft6.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': '6-point FFT implementation'
            },
            'c_fft_fine': {
                'binary': 'bin/c_fft_fine.par.gnu',
                'args_template': {
                    'small': ['{fft_size_kb}'],
                    'medium': ['{fft_size_kb}'],
                    'large': ['{fft_size_kb}'],
                    'huge': ['{fft_size_kb}'],
                    'extreme': ['{fft_size_kb}'],
                },
                'description': 'Fast Fourier Transform - Fine-grained (cutoff 64, dynamic scheduling)'
            },
            'c_fft_coarse': {
                'binary': 'bin/c_fft_coarse.par.gnu',
                'args_template': {
                    'small': ['{fft_size_kb}'],
                    'medium': ['{fft_size_kb}'],
                    'large': ['{fft_size_kb}'],
                    'huge': ['{fft_size_kb}'],
                    'extreme': ['{fft_size_kb}'],
                },
                'description': 'Fast Fourier Transform - Coarse-grained (cutoff 4096, static scheduling)'
            },
            'c_md': {
                'binary': 'bin/c_md.par.gnu',
                'args_template': {
                    'small': ['{md_particles}', '{md_steps}'],
                    'medium': ['{md_particles}', '{md_steps}'],
                    'large': ['{md_particles}', '{md_steps}'],
                    'huge': ['{md_particles}', '{md_steps}'],
                    'extreme': ['{md_particles}', '{md_steps}'],
                },
                'description': 'Molecular Dynamics simulation'
            },
            'c_md_fine': {
                'binary': 'bin/c_md_fine.par.gnu',
                'args_template': {
                    'small': ['{md_particles}', '{md_steps}'],
                    'medium': ['{md_particles}', '{md_steps}'],
                    'large': ['{md_particles}', '{md_steps}'],
                    'huge': ['{md_particles}', '{md_steps}'],
                    'extreme': ['{md_particles}', '{md_steps}'],
                },
                'description': 'Molecular Dynamics - Fine-grained (dynamic scheduling, chunk 8)'
            },
            'c_md_coarse': {
                'binary': 'bin/c_md_coarse.par.gnu',
                'args_template': {
                    'small': ['{md_particles}', '{md_steps}'],
                    'medium': ['{md_particles}', '{md_steps}'],
                    'large': ['{md_particles}', '{md_steps}'],
                    'huge': ['{md_particles}', '{md_steps}'],
                    'extreme': ['{md_particles}', '{md_steps}'],
                },
                'description': 'Molecular Dynamics - Coarse-grained (static scheduling, large chunks)'
            },
            'c_lu': {
                'binary': 'bin/c_lu.par.gnu',
                'args_template': {
                    'small': ['{lu_size}'],
                    'medium': ['{lu_size}'],
                    'large': ['{lu_size}'],
                    'huge': ['{lu_size}'],
                    'extreme': ['{lu_size}'],
                },
                'description': 'LU decomposition'
            },
            'c_lu_fine': {
                'binary': 'bin/c_lu_fine.par.gnu',
                'args_template': {
                    'small': ['{lu_size}'],
                    'medium': ['{lu_size}'],
                    'large': ['{lu_size}'],
                    'huge': ['{lu_size}'],
                    'extreme': ['{lu_size}'],
                },
                'description': 'LU decomposition - Fine-grained (dynamic scheduling, chunk 2)'
            },
            'c_lu_coarse': {
                'binary': 'bin/c_lu_coarse.par.gnu',
                'args_template': {
                    'small': ['{lu_size}'],
                    'medium': ['{lu_size}'],
                    'large': ['{lu_size}'],
                    'huge': ['{lu_size}'],
                    'extreme': ['{lu_size}'],
                },
                'description': 'LU decomposition - Coarse-grained (static scheduling, large chunks)'
            },
            
            # Jacobi solver variants (with configurable problem sizes)
            'c_jacobi01': {
                'binary': 'bin/c_jacobi01.par.gnu',
                'args_template': {
                    'tiny': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'huge': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'extreme': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'massive': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'colossal': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'gigantic': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver v1'
            },
            'c_jacobi02': {
                'binary': 'bin/c_jacobi02.par.gnu',
                'args_template': {
                    'tiny': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'huge': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'extreme': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'massive': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'colossal': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'gigantic': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver v2'
            },
            'c_jacobi03': {
                'binary': 'bin/c_jacobi03.par.gnu',
                'args_template': {
                    'tiny': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'huge': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'extreme': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'massive': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'colossal': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'gigantic': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver v3'
            },
            'c_jacobi_fine': {
                'binary': 'bin/c_jacobi_fine.par.gnu',
                'args_template': {
                    'tiny': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'huge': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'extreme': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'massive': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'colossal': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'gigantic': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver - Fine-grained (dynamic scheduling, chunk 4)'
            },
            'c_jacobi_coarse': {
                'binary': 'bin/c_jacobi_coarse.par.gnu',
                'args_template': {
                    'tiny': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'huge': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'extreme': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'massive': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'colossal': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'gigantic': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver - Coarse-grained (static scheduling, large chunks)'
            },
            
            # Graph search variants
            'c_testPath': {
                'binary': 'bin/c_testPath.par.gnu',
                'args_template': {                    'small': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'medium': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'large': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'huge': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'extreme': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                },
                'description': 'Graph path search using workers-farm paradigm'
            },
            'c_testPath_fine': {
                'binary': 'bin/c_testPath_fine.par.gnu',
                'args_template': {                    'small': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'medium': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'large': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'huge': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'extreme': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                },
                'description': 'Graph path search - Fine-grained (single node per pool access)'
            },
            'c_testPath_coarse': {
                'binary': 'bin/c_testPath_coarse.par.gnu',
                'args_template': {                    'small': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'medium': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'large': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'huge': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                    'extreme': ['1', '29', 'applications/c_GraphSearch/exampleGraph_01.gph'],
                },
                'description': 'Graph path search - Coarse-grained (batch of 10 nodes per pool access)'
            },
            
            # Loop dependency examples (correct implementations)
            'c_loopA_sol1': {
                'binary': 'bin/c_loopA.solution1.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': 'Loop A dependency - Solution 1'
            },
            'c_loopA_sol2': {
                'binary': 'bin/c_loopA.solution2.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': 'Loop A dependency - Solution 2'
            },
            'c_loopA_sol3': {
                'binary': 'bin/c_loopA.solution3.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': 'Loop A dependency - Solution 3'
            },
            'c_loopB_pipeline': {
                'binary': 'bin/c_loopB.pipelineSolution.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': 'Loop B dependency - Pipeline Solution'
            },
            
            # Bad implementations (for race detection studies)
            'c_loopA_bad': {
                'binary': 'bin/c_loopA.badSolution.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': 'Loop A dependency - Bad Solution (has races)'
            },
            'c_loopB_bad1': {
                'binary': 'bin/c_loopB.badSolution1.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': 'Loop B dependency - Bad Solution 1 (has races)'
            },
            'c_loopB_bad2': {
                'binary': 'bin/c_loopB.badSolution2.par.gnu',
                'args_template': {
                    
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test'],
                    'huge': ['-test'],
                    'extreme': ['-test'],
                    
                    
                    
                },
                'description': 'Loop B dependency - Bad Solution 2 (has races)'
            }
        }
        
        # Default thread counts to test (updated to include 32 threads)
        self.default_threads = [1, 2, 4, 8, 16, 24, 32]
    
    def enable_auto_analysis(self, analysis_output_dir="analysis_output"):
        """Enable automatic analysis after benchmark completion"""
        self.auto_analyze = True
        self.analysis_output_dir = analysis_output_dir
        
    def configure_email_notifications(self, sender_email, sender_password, recipient_emails, 
                                    smtp_server="smtp.gmail.com", smtp_port=587):
        """Configure email notifications for benchmark completion"""
        self.email_notifier.smtp_server = smtp_server
        self.email_notifier.smtp_port = smtp_port
        self.email_notifier.configure(sender_email, sender_password, recipient_emails)
        
    def check_binary_exists(self, binary_path):
        """Check if the benchmark binary exists and is executable"""
        return os.path.isfile(binary_path) and os.access(binary_path, os.X_OK)
    
    def get_benchmark_args(self, config, problem_size):
        """Get benchmark arguments based on problem size"""
        args_template = config['args_template'][problem_size]
        size_config = self.problem_sizes[problem_size]
        
        # Replace placeholders with actual values
        args = []
        for arg in args_template:
            if '{grid_size}' in arg:
                args.append(arg.format(grid_size=size_config['grid_size']))
            elif '{iterations}' in arg:
                args.append(arg.format(iterations=size_config['iterations']))
            elif '{array_size}' in arg:
                args.append(arg.format(array_size=size_config['array_size']))
            elif '{fft_size}' in arg:
                args.append(arg.format(fft_size=size_config['fft_size']))
            elif '{md_particles}' in arg:
                args.append(arg.format(md_particles=size_config['md_particles']))
            elif '{md_steps}' in arg:
                args.append(arg.format(md_steps=size_config['md_steps']))
            elif '{qsort_size}' in arg:
                args.append(arg.format(qsort_size=size_config['qsort_size']))
            elif '{fft_size_kb}' in arg:
                args.append(arg.format(fft_size_kb=size_config['fft_size_kb']))
            elif '{lu_size}' in arg:
                args.append(arg.format(lu_size=size_config['lu_size']))
            else:
                args.append(arg)
        
        return args
    
    def show_cpu_topology(self):
        """Display CPU topology information"""
        try:
            print("\n🖥️  CPU TOPOLOGY INFORMATION")
            print("=" * 50)
            
            import multiprocessing
            total_cores = multiprocessing.cpu_count()
            print(f"📊 Total logical processors: {total_cores}")
            
            # Try to get physical core count
            try:
                import psutil
                physical_cores = psutil.cpu_count(logical=False)
                logical_cores = psutil.cpu_count(logical=True)
                print(f"🔧 Physical cores: {physical_cores}")
                print(f"🧵 Logical cores (with HT): {logical_cores}")
                
                if logical_cores > physical_cores:
                    print(f"⚡ Hyperthreading: Enabled ({logical_cores//physical_cores}x)")
                else:
                    print(f"⚡ Hyperthreading: Disabled")
                    
            except ImportError:
                print("📝 Note: Install 'psutil' for detailed CPU topology")
            
            # Show OpenMP environment
            print("\n🔧 OpenMP Configuration:")
            print(f"   OMP_PROC_BIND: close (use adjacent cores)")
            print(f"   OMP_PLACES: cores (one thread per core)")
            print(f"   OMP_DISPLAY_AFFINITY: enabled")
            
            # Check for NUMA
            try:
                numa_info = subprocess.run(['numactl', '--hardware'], 
                                         capture_output=True, text=True, timeout=5)
                if numa_info.returncode == 0:
                    numa_nodes = len([line for line in numa_info.stdout.split('\n') 
                                    if 'node' in line and 'cpus:' in line])
                    print(f"🏗️  NUMA nodes: {numa_nodes}")
                else:
                    print(f"🏗️  NUMA: Information not available")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(f"🏗️  NUMA: numactl not available")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"⚠️  Could not retrieve CPU topology: {e}")
    
    def get_cpu_affinity_info(self, thread_count):
        """Get CPU core mapping information for the specified thread count"""
        try:
            # Get CPU information
            import multiprocessing
            total_cores = multiprocessing.cpu_count()
            
            # For close binding, threads typically use cores 0, 1, 2, ..., thread_count-1
            if thread_count <= total_cores:
                cores_used = list(range(thread_count))
                return f"Cores {cores_used} of {total_cores} available"
            else:
                return f"Warning: {thread_count} threads > {total_cores} cores (oversubscription)"
                
        except Exception:
            return None
    
    def extract_cpu_usage_info(self, output):
        """Extract CPU core usage information from benchmark output"""
        cpu_cores = []
        
        # Look for OpenMP affinity information
        affinity_patterns = [
            r'Thread\s+\d+:\s+Core\s+(\d+)',
            r'thread\s+\d+\s+bound\s+to\s+OS\s+proc\s+(\d+)',
            r'Thread\s+\d+.*?core\s+(\d+)',
            r'OMP:\s+Info.*?thread\s+\d+.*?(\d+)'
        ]
        
        for pattern in affinity_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                cpu_cores.extend([int(core) for core in matches])
        
        if cpu_cores:
            unique_cores = sorted(set(cpu_cores))
            if len(unique_cores) == len(cpu_cores):
                return f"{unique_cores}"
            else:
                # Some cores used by multiple threads (hyperthreading or oversubscription)
                return f"{unique_cores} (some shared)"
        
        return None
    
    def update_progress(self, force_save=False):
        """Update progress information and optionally save results"""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        if self.completed_runs > 0:
            avg_time_per_run = elapsed / self.completed_runs
            estimated_total = avg_time_per_run * self.total_runs
            eta = estimated_total - elapsed
        else:
            eta = 0
            
        progress_pct = (self.completed_runs / self.total_runs * 100) if self.total_runs > 0 else 0
        
        # Update progress file
        if self.progress_file:
            progress_info = {
                'timestamp': datetime.now().isoformat(),
                'completed_runs': self.completed_runs,
                'total_runs': self.total_runs,
                'progress_pct': progress_pct,
                'elapsed_time_s': elapsed,
                'eta_s': eta,
                'current_benchmark': self.current_benchmark,
                'current_config': self.current_config,
                'successful_runs': sum(1 for r in self.results if r['success']),
                'failed_runs': sum(1 for r in self.results if not r['success'])
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_info, f, indent=2)
        
        # Print progress update every 10 runs
        if self.completed_runs % 10 == 0 or force_save:
            print(f"\n📊 Progress: {self.completed_runs}/{self.total_runs} ({progress_pct:.1f}%)")
            print(f"⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            print(f"🔄 Currently: {self.current_benchmark} {self.current_config}")
        
        # Save intermediate results every 50 runs or when forced
        if force_save or (self.completed_runs > 0 and self.completed_runs % 50 == 0):
            self.save_intermediate_results()
    
    def save_intermediate_results(self):
        """Save current results as intermediate backup"""
        if not self.results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save intermediate JSON
        json_file = self.output_dir / f"intermediate_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"💾 Intermediate results saved: {json_file.name}")
    
    def run_single_benchmark(self, name, config, thread_count, problem_size='medium', iteration=1):
        """Run a single benchmark with specified thread count and problem size"""
        binary = config['binary']
        args = self.get_benchmark_args(config, problem_size)
        
        if not self.check_binary_exists(binary):
            print(f"  ⚠️  Binary not found: {binary}")
            return None
            
        # Set environment with CPU affinity information
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(thread_count)
        
        # Configure OpenMP to display thread-to-core mapping
        env['OMP_DISPLAY_AFFINITY'] = 'TRUE'
        env['OMP_AFFINITY_FORMAT'] = 'Thread %0.3n: Core %A'
        
        # Set close affinity to use adjacent cores
        env['OMP_PROC_BIND'] = 'close'
        env['OMP_PLACES'] = 'cores'
        
        print(f"  Running {name} ({problem_size}) with {thread_count} threads (iteration {iteration})...")
        
        # Get CPU affinity information before execution
        if self.show_cpu_usage:
            cpu_info = self.get_cpu_affinity_info(thread_count)
            if cpu_info:
                print(f"    💻 CPU Mapping: {cpu_info}")
        else:
            cpu_info = None
        
        # Update current status
        self.current_benchmark = name
        self.current_config = f"{problem_size}, {thread_count}T, iter{iteration}"
        
        start_time = time.time()
        
        try:
            # Run the benchmark
            result = subprocess.run(
                [f"./{binary}"] + args,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout for extreme problems
                env=env
            )
            
            end_time = time.time()
            wall_time = end_time - start_time
            
            # Parse output for additional metrics and CPU usage
            timing_info = self.extract_timing_info(result.stdout)
            
            # Extract CPU affinity information from output
            cpu_usage = self.extract_cpu_usage_info(result.stderr + result.stdout) if self.show_cpu_usage else None
            
            result_data = {
                'timestamp': datetime.now().isoformat(),
                'benchmark': name,
                'description': config['description'],
                'threads': thread_count,
                'problem_size': problem_size,
                'iteration': iteration,
                'wall_time': wall_time,
                'exit_code': result.returncode,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'cpu_affinity': cpu_info if self.show_cpu_usage else None,
                'cpu_usage': cpu_usage,
                **timing_info
            }
            
            if result.returncode == 0:
                if self.show_cpu_usage and cpu_usage:
                    print(f"    ✓ Completed in {wall_time:.3f}s - Cores used: {cpu_usage}")
                else:
                    print(f"    ✓ Completed in {wall_time:.3f}s")
            else:
                print(f"    ✗ Failed with exit code {result.returncode}")
            
            # Update progress counter
            self.completed_runs += 1
            self.update_progress()
                
            return result_data
            
        except subprocess.TimeoutExpired:
            print(f"    ⏱️  Timeout after 30 minutes")
            self.completed_runs += 1
            self.update_progress()
            return {
                'timestamp': datetime.now().isoformat(),
                'benchmark': name,
                'description': config['description'],
                'threads': thread_count,
                'problem_size': problem_size,
                'iteration': iteration,
                'wall_time': 1800.0,
                'exit_code': -1,
                'success': False,
                'stdout': '',
                'stderr': 'Timeout',
                'timeout': True
            }
            
        except Exception as e:
            print(f"    💥 Exception: {str(e)}")
            self.completed_runs += 1
            self.update_progress()
            return {
                'timestamp': datetime.now().isoformat(),
                'benchmark': name,
                'description': config['description'],
                'threads': thread_count,
                'problem_size': problem_size,
                'iteration': iteration,
                'wall_time': 0.0,
                'exit_code': -999,
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'exception': True
            }
    
    def extract_timing_info(self, output):
        """Extract timing and performance information from benchmark output"""
        info = {}
        
        # Look for common timing patterns
        patterns = {
            'cpu_time': r'TIME.*?(\d+\.?\d*)',
            'timer_total': r'Timer\s+Total_time\s+(\d+\.?\d*)',
            'elapsed_time': r'elapsed time.*?(\d+\.?\d*)',
            'mflops': r'MFlops.*?(\d+\.?\d*)',
            'pi_error': r'ERROR\s*(\d+\.?\d*e?[+-]?\d*)',
            'solution_error': r'Solution Error.*?(\d+\.?\d*e?[+-]?\d*)',
            'pi_value': r'PI\s*=\s*(\d+\.?\d*)',
            'checksum': r'checksum.*?(\d+)',
            'verification': r'verification.*?(SUCCESSFUL|FAILED|passed|failed)',
            'result_value': r'Result.*?(\d+\.?\d*e?[+-]?\d*)',
            'iterations_done': r'iterations.*?(\d+)',
            'convergence': r'converged.*?(\d+\.?\d*e?[+-]?\d*)',
            'final_error': r'Final.*?error.*?(\d+\.?\d*e?[+-]?\d*)',
            'array_sum': r'Sum.*?(\d+\.?\d*e?[+-]?\d*)',
            'matrix_norm': r'Norm.*?(\d+\.?\d*e?[+-]?\d*)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    if key == 'verification':
                        info[key] = match.group(1).upper()
                    else:
                        info[key] = float(match.group(1))
                except ValueError:
                    info[key] = match.group(1)
        
        # Store full output for integrity checking
        info['full_output'] = output
        
        return info
    
    def check_result_integrity(self, results, benchmark_name):
        """Check integrity of results across different configurations"""
        integrity_report = {
            'benchmark': benchmark_name,
            'consistent': True,
            'issues': [],
            'reference_values': {},
            'value_ranges': {},
            'verification_status': {}
        }
        
        # Filter results for this benchmark
        benchmark_results = [r for r in results if r.get('benchmark') == benchmark_name and r.get('success')]
        
        if len(benchmark_results) < 2:
            integrity_report['issues'].append("Insufficient results for integrity check")
            return integrity_report
        
        # Check consistency of numerical results
        numerical_fields = ['pi_value', 'result_value', 'array_sum', 'matrix_norm', 'checksum']
        
        for field in numerical_fields:
            values = [r.get(field) for r in benchmark_results if r.get(field) is not None]
            if values:
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                
                # Calculate relative variance
                if avg_val != 0:
                    variance = ((max_val - min_val) / avg_val) * 100
                else:
                    variance = 0
                
                integrity_report['reference_values'][field] = avg_val
                integrity_report['value_ranges'][field] = {
                    'min': min_val,
                    'max': max_val,
                    'variance_pct': variance
                }
                
                # Flag if variance > threshold (may indicate race conditions or numerical instability)
                if variance > self.integrity_threshold:
                    integrity_report['consistent'] = False
                    integrity_report['issues'].append(f"{field}: {variance:.3f}% variance (min={min_val}, max={max_val})")
        
        # Check verification status consistency
        verifications = [r.get('verification') for r in benchmark_results if r.get('verification') is not None]
        if verifications:
            unique_verifications = set(verifications)
            integrity_report['verification_status'] = {
                'values': list(unique_verifications),
                'count': len(verifications),
                'consistent': len(unique_verifications) == 1
            }
            
            if len(unique_verifications) > 1:
                integrity_report['consistent'] = False
                integrity_report['issues'].append(f"Inconsistent verification status: {unique_verifications}")
        
        # Check for abnormal execution patterns
        execution_times = [r.get('wall_time') for r in benchmark_results if r.get('wall_time') is not None]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            outliers = [t for t in execution_times if abs(t - avg_time) > avg_time * 2]  # 200% deviation
            
            if outliers:
                integrity_report['issues'].append(f"Execution time outliers detected: {len(outliers)} runs significantly different")
        
        return integrity_report
    
    def generate_integrity_report(self, results):
        """Generate comprehensive integrity report for all benchmarks"""
        print("\n🔍 GENERATING INTEGRITY REPORT...")
        
        # Group results by benchmark
        benchmarks = set(r.get('benchmark') for r in results if r.get('success'))
        
        integrity_reports = {}
        overall_issues = []
        
        for benchmark in benchmarks:
            report = self.check_result_integrity(results, benchmark)
            integrity_reports[benchmark] = report
            
            if not report['consistent']:
                overall_issues.extend([f"{benchmark}: {issue}" for issue in report['issues']])
        
        # Generate summary
        consistent_benchmarks = sum(1 for r in integrity_reports.values() if r['consistent'])
        total_benchmarks = len(integrity_reports)
        
        print(f"📊 INTEGRITY SUMMARY:")
        print(f"   Benchmarks analyzed: {total_benchmarks}")
        print(f"   Consistent results: {consistent_benchmarks}/{total_benchmarks}")
        
        if overall_issues:
            print(f"   ⚠️  Issues found: {len(overall_issues)}")
            print("\n🚨 INTEGRITY ISSUES:")
            for issue in overall_issues[:10]:  # Show first 10 issues
                print(f"   • {issue}")
            if len(overall_issues) > 10:
                print(f"   ... and {len(overall_issues) - 10} more issues")
        else:
            print("   ✅ All benchmarks show consistent results!")
        
        return integrity_reports
    
    def show_detailed_integrity_report(self, integrity_reports):
        """Show detailed integrity report in console"""
        print("\n" + "=" * 80)
        print("🔍 DETAILED INTEGRITY VERIFICATION REPORT")
        print("=" * 80)
        
        for benchmark, report in integrity_reports.items():
            print(f"\n📊 {benchmark.upper()}:")
            print(f"   Status: {'✅ CONSISTENT' if report['consistent'] else '⚠️  INCONSISTENT'}")
            
            if report['reference_values']:
                print(f"   📈 Reference Values:")
                for field, value in report['reference_values'].items():
                    range_info = report['value_ranges'][field]
                    variance = range_info['variance_pct']
                    status = "✅" if variance <= self.integrity_threshold else "⚠️"
                    print(f"      {status} {field}: {value:.6g} (variance: {variance:.3f}%)")
            
            if report['verification_status']:
                vs = report['verification_status']
                status = "✅" if vs['consistent'] else "⚠️"
                print(f"   {status} Verification: {vs['values']} ({vs['count']} runs)")
            
            if report['issues']:
                print(f"   🚨 Issues ({len(report['issues'])}):")
                for issue in report['issues']:
                    print(f"      • {issue}")
        
        print(f"\n📋 Summary:")
        consistent = sum(1 for r in integrity_reports.values() if r['consistent'])
        total = len(integrity_reports)
        print(f"   Consistent benchmarks: {consistent}/{total}")
        
        if consistent < total:
            print(f"   ⚠️  Recommendations:")
            print(f"      • Check for race conditions in inconsistent benchmarks")
            print(f"      • Verify numerical stability across thread counts")
            print(f"      • Consider increasing problem sizes for better thread scaling")
        
        print("=" * 80)
    
    def run_benchmarks(self, thread_counts=None, iterations=1, benchmarks=None, problem_sizes=None):
        """Run all benchmarks with specified parameters"""
        if thread_counts is None:
            thread_counts = self.default_threads
            
        if benchmarks is None:
            benchmarks = list(self.benchmarks.keys())
        
        if problem_sizes is None:
            problem_sizes = ['tiny', 'small', 'medium', 'large', 'huge', 'extreme', 'massive', 'colossal', 'gigantic']
        
        # Show CPU topology information if requested
        if self.show_cpu_usage:
            self.show_cpu_topology()
        
        # Filter benchmarks to only include available ones
        available_benchmarks = []
        for name in benchmarks:
            if name in self.benchmarks:
                binary = self.benchmarks[name]['binary']
                if self.check_binary_exists(binary):
                    available_benchmarks.append(name)
                else:
                    print(f"⚠️  Skipping {name}: binary not found at {binary}")
            else:
                print(f"⚠️  Unknown benchmark: {name}")
        
        if not available_benchmarks:
            print("❌ No available benchmarks found!")
            return
        
        print(f"🚀 Running {len(available_benchmarks)} benchmarks")
        print(f"📊 Thread counts: {thread_counts}")
        print(f"📏 Problem sizes: {problem_sizes}")
        print(f"🔄 Iterations per configuration: {iterations}")
        
        # Initialize progress tracking
        self.total_runs = len(available_benchmarks) * len(thread_counts) * len(problem_sizes) * iterations
        self.completed_runs = 0
        self.start_time = time.time()
        
        # Create progress file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_file = self.output_dir / f"progress_{timestamp}.json"
        
        print(f"📁 Progress tracking: {self.progress_file.name}")
        print(f"📈 Total configurations: {self.total_runs}")
        print("=" * 60)
        
        current_run = 0
        
        for name in available_benchmarks:
            config = self.benchmarks[name]
            print(f"\n📈 {name}: {config['description']}")
            
            for problem_size in problem_sizes:
                print(f"  🔧 Problem size: {problem_size}")
                
                for thread_count in thread_counts:
                    for iteration in range(1, iterations + 1):
                        current_run += 1
                        print(f"[{current_run}/{self.total_runs}]", end=" ")
                        
                        result = self.run_single_benchmark(name, config, thread_count, problem_size, iteration)
                        if result:
                            self.results.append(result)
        
        # Final progress update
        self.update_progress(force_save=True)
        
        print("\n" + "=" * 60)
        print("✅ Benchmark execution completed!")
        
    def save_results(self, timestamp=None):
        """Save results to CSV and JSON files, with optional analysis and email notification"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON (detailed)
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV (summary)
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        if self.results:
            with open(csv_file, 'w', newline='') as f:
                fieldnames = [
                    'timestamp', 'benchmark', 'description', 'threads', 'problem_size', 'iteration',
                    'wall_time', 'exit_code', 'success', 'cpu_time', 'timer_total',
                    'elapsed_time', 'mflops', 'pi_error', 'solution_error'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    # Create a filtered row with only the CSV fields
                    row = {field: result.get(field, '') for field in fieldnames}
                    writer.writerow(row)
        
        # Generate summary report
        summary_file = self.generate_summary_report(timestamp)
        
        print(f"📁 Results saved:")
        print(f"   CSV: {csv_file}")
        print(f"   JSON: {json_file}")
        print(f"   Summary: {summary_file}")
        
        # Perform automatic analysis if enabled
        analysis_result = None
        if self.auto_analyze:
            print(f"\n🔍 Running automatic analysis...")
            analysis_dir = self.analysis_output_dir or f"analysis_output_{timestamp}"
            analysis_result = self.analyzer.analyze_performance(self.results, analysis_dir)
            
            if analysis_result:
                print(f"📊 Analysis completed:")
                if analysis_result['plots']:
                    print(f"   📈 Plots: {len(analysis_result['plots'])} generated")
                print(f"   📋 Report: {analysis_result['report']}")
        
        # Generate integrity report
        integrity_reports = self.generate_integrity_report(self.results)
        integrity_file = self.output_dir / f"integrity_report_{timestamp}.json"
        with open(integrity_file, 'w') as f:
            json.dump(integrity_reports, f, indent=2)
        
        print(f"   🔍 Integrity: {integrity_file}")
        
        # Show detailed integrity report if requested
        if self.check_integrity_detailed:
            self.show_detailed_integrity_report(integrity_reports)
        
        # Send email notification if configured
        if self.email_notifier.enabled:
            self.send_email_notification(timestamp, csv_file, json_file, summary_file, analysis_result)
        
        return {
            'csv': csv_file,
            'json': json_file, 
            'summary': summary_file,
            'analysis': analysis_result,
            'integrity': integrity_file
        }
    
    def send_email_notification(self, timestamp, csv_file, json_file, summary_file, analysis_result):
        """Send email notification with benchmark results"""
        try:
            # Calculate basic statistics
            total_runs = len(self.results)
            successful_runs = sum(1 for r in self.results if r.get('success', False))
            success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
            
            if successful_runs > 0:
                avg_time = sum(r.get('wall_time', 0) for r in self.results if r.get('success', False)) / successful_runs
            else:
                avg_time = 0
            
            # Execution duration
            if self.start_time:
                execution_duration = time.time() - self.start_time
                duration_str = f"{execution_duration/60:.1f} minutes"
            else:
                duration_str = "Unknown"
            
            # Create email body
            subject = f"OmpSCR Benchmark Results - {timestamp}"
            
            body = f"""
OmpSCR Benchmark Execution Completed
=====================================

Execution Summary:
- Timestamp: {timestamp}
- Total runs: {total_runs}
- Successful runs: {successful_runs}
- Success rate: {success_rate:.1f}%
- Average execution time: {avg_time:.3f}s
- Total execution duration: {duration_str}

Benchmarks tested: {len(set(r.get('benchmark', 'Unknown') for r in self.results))}
Thread counts: {sorted(set(r.get('threads', 0) for r in self.results))}
Problem sizes: {sorted(set(r.get('problem_size', 'Unknown') for r in self.results))}

"""
            
            if analysis_result:
                body += f"""
Analysis Results:
- Plots generated: {len(analysis_result['plots'])}
- Analysis report: {analysis_result['report'].name}
- Summary: {analysis_result['summary'].strip()}
"""
            
            # Prepare attachments
            attachments = [str(csv_file), str(json_file), str(summary_file)]
            
            if analysis_result:
                if analysis_result['report']:
                    attachments.append(str(analysis_result['report']))
                if analysis_result['plots']:
                    attachments.extend([str(plot) for plot in analysis_result['plots']])
            
            # Send email
            success = self.email_notifier.send_notification(subject, body, attachments)
            
            if success:
                print(f"📧 Email notification sent successfully!")
            else:
                print(f"⚠️  Email notification failed")
                
        except Exception as e:
            print(f"❌ Error sending email notification: {e}")
    
    def generate_summary_report(self, timestamp):
        """Generate a text summary of the benchmark results"""
        if not self.results:
            return None
        
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("OmpSCR Benchmark Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total runs: {len(self.results)}\n\n")
            
            # Success rate by benchmark
            f.write("Success Rate by Benchmark:\n")
            f.write("-" * 30 + "\n")
            
            benchmark_stats = {}
            for result in self.results:
                name = result['benchmark']
                if name not in benchmark_stats:
                    benchmark_stats[name] = {'total': 0, 'success': 0}
                
                benchmark_stats[name]['total'] += 1
                if result['success']:
                    benchmark_stats[name]['success'] += 1
            
            for name, stats in sorted(benchmark_stats.items()):
                success_rate = (stats['success'] / stats['total']) * 100
                f.write(f"{name:20}: {stats['success']:3}/{stats['total']:3} ({success_rate:5.1f}%)\n")
            
            # Performance summary for successful runs
            f.write("\nPerformance Summary (Successful Runs):\n")
            f.write("-" * 40 + "\n")
            
            successful_results = [r for r in self.results if r['success']]
            
            if successful_results:
                # Group by benchmark, problem_size and threads
                perf_data = {}
                for result in successful_results:
                    key = (result['benchmark'], result['problem_size'], result['threads'])
                    if key not in perf_data:
                        perf_data[key] = []
                    perf_data[key].append(result['wall_time'])
                
                f.write("Benchmark                Size     Threads  Avg Time (s)  Std Dev (s)  Min Time (s)  Max Time (s)\n")
                f.write("-" * 95 + "\n")
                
                for (benchmark, problem_size, threads), times in sorted(perf_data.items()):
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    
                    # Calculate standard deviation
                    if len(times) > 1:
                        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
                        std_dev = variance ** 0.5
                    else:
                        std_dev = 0.0
                    
                    f.write(f"{benchmark:20} {problem_size:8} {threads:8d} {avg_time:11.3f} {std_dev:11.3f} {min_time:11.3f} {max_time:11.3f}\n")
        
        return summary_file

def main():
    parser = argparse.ArgumentParser(description='Run OmpSCR benchmarks with varying parameters')
    parser.add_argument('--threads', type=str, default='1,2,4,8,16,24',
                        help='Comma-separated list of thread counts (default: 1,2,4,8,16,24)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations per configuration (default: 3)')
    parser.add_argument('--benchmarks', type=str, default='all',
                        help='Comma-separated list of benchmarks or "all" (default: all)')
    parser.add_argument('--problem-sizes', type=str, default='tiny,small,medium,large,huge,extreme,massive,colossal,gigantic',
                        help='Comma-separated list of problem sizes: tiny,small,medium,large,huge,extreme,massive,colossal,gigantic (default: all)')
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory (default: benchmark_results)')
    parser.add_argument('--list', action='store_true',
                        help='List available benchmarks and exit')
    parser.add_argument('--full-test', action='store_true',
                        help='Run comprehensive test with 1,2,4,8,12,16,24 threads, ALL sizes (tiny→gigantic), 10 iterations')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with 1,2,4 threads, tiny,small,medium sizes, 3 iterations')
    parser.add_argument('--stress-test', action='store_true',
                        help='Run stress test with 1,2,4,8,16,24,32 threads, all sizes including gigantic, 15 iterations')
    parser.add_argument('--extreme-test', action='store_true',
                        help='Run extreme test with massive,colossal,gigantic sizes only, 1,2,4,8,16,32,48,64 threads, 5 iterations')
    parser.add_argument('--auto-analyze', action='store_true',
                        help='Automatically run analysis and generate plots after benchmark completion')
    parser.add_argument('--analysis-output', type=str, default='analysis_output',
                        help='Output directory for analysis results (default: analysis_output)')
    parser.add_argument('--check-integrity', action='store_true',
                        help='Show detailed integrity report for result verification across configurations')
    parser.add_argument('--integrity-threshold', type=float, default=0.1,
                        help='Variance threshold percentage for integrity checks (default: 0.1)')
    parser.add_argument('--show-cpu-usage', action='store_true',
                        help='Show detailed CPU core usage information during execution')
    parser.add_argument('--email-notification', action='store_true',
                        help='Send email notification with results (requires email configuration)')
    parser.add_argument('--email-config', type=str,
                        help='Email configuration file (JSON format with sender, password, recipients)')
    parser.add_argument('--email-sender', type=str,
                        help='Sender email address')
    parser.add_argument('--email-recipients', type=str,
                        help='Comma-separated list of recipient email addresses')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.output, args.check_integrity, args.integrity_threshold, args.show_cpu_usage)
    
    # Configure automatic analysis if requested
    if args.auto_analyze:
        runner.enable_auto_analysis(args.analysis_output)
        print(f"🔍 Automatic analysis enabled - output: {args.analysis_output}")
    
    # Configure integrity checking if requested
    if args.check_integrity:
        print(f"🔍 Integrity checking enabled - variance threshold: {args.integrity_threshold}")
    
    # Configure CPU usage monitoring if requested
    if args.show_cpu_usage:
        print(f"💻 CPU usage monitoring enabled")
    
    # Configure email notifications if requested
    if args.email_notification:
        email_configured = False
        
        # Try to load from config file first
        if args.email_config:
            try:
                with open(args.email_config) as f:
                    email_config = json.load(f)
                
                runner.configure_email_notifications(
                    sender_email=email_config['sender'],
                    sender_password=email_config['password'],
                    recipient_emails=email_config['recipients']
                )
                email_configured = True
                print(f"📧 Email notifications enabled from config: {args.email_config}")
                
            except Exception as e:
                print(f"❌ Failed to load email config: {e}")
        
        # Try command line arguments
        elif args.email_sender and args.email_recipients:
            import getpass
            
            password = getpass.getpass("Enter sender email password: ")
            recipients = [r.strip() for r in args.email_recipients.split(',')]
            
            runner.configure_email_notifications(
                sender_email=args.email_sender,
                sender_password=password,
                recipient_emails=recipients
            )
            email_configured = True
            print(f"📧 Email notifications enabled for: {', '.join(recipients)}")
        
        if not email_configured:
            print("⚠️  Email notification requested but not properly configured!")
            print("   Use --email-config file.json or --email-sender + --email-recipients")
            print("   Continuing without email notifications...")
    
    if args.list:
        print("Available benchmarks:")
        print("=" * 50)
        for name, config in runner.benchmarks.items():
            status = "✓" if runner.check_binary_exists(config['binary']) else "✗"
            print(f"{status} {name:20}: {config['description']}")
        print("\nProblem sizes:")
        print("=" * 120)
        for size, config in runner.problem_sizes.items():
            print(f"• {size:9}: grid={config['grid_size']:5d}, iter={config['iterations']:5d}, "
                  f"array={config['array_size']:9d}, fft={config['fft_size']:7d}")
            print(f"            md_parts={config['md_particles']:6d}, md_steps={config['md_steps']:4d}, "
                  f"qsort_kb={config['qsort_size']:6d}, fft_kb={config['fft_size_kb']:4d}, lu={config['lu_size']:4d}")
        return
    
    # Handle test modes
    if args.full_test:
        thread_counts = [1, 2, 4, 8, 12, 16, 24]
        iterations = 5
        benchmarks = None  # All benchmarks
        problem_sizes = ['small', 'medium', 'large', 'huge', 'extreme']
        print("🔬 Running FULL COMPREHENSIVE TEST (ALL NEW SIZES)")
        print(f"   Threads: {thread_counts}")
        print(f"   Sizes: {problem_sizes}")
        print(f"   Iterations: {iterations}")
        total_configs = len(runner.benchmarks) * len(thread_counts) * len(problem_sizes) * iterations
        print(f"   Total runs: {total_configs}")
        print("⚠️  WARNING: This test includes EXTREME problems (1GB) and may take VERY long!")
        print("💡 Use 'python3 monitor_progress.py' in another terminal to monitor progress")
        print("")
    elif args.quick_test:
        thread_counts = [1, 2, 4, 8]
        iterations = 3
        benchmarks = None  # All benchmarks
        problem_sizes = ['small', 'medium']
        print("⚡ Running QUICK TEST")
        print(f"   Threads: {thread_counts}")
        print(f"   Sizes: {problem_sizes}")
        print(f"   Iterations: {iterations}")
        total_configs = len(runner.benchmarks) * len(thread_counts) * len(problem_sizes) * iterations
        print(f"   Total runs: {total_configs}")
        print("")
    elif args.stress_test:
        thread_counts = [1, 2, 4, 8, 16, 24]
        iterations = 10
        benchmarks = None  # All benchmarks
        problem_sizes = ['small', 'medium', 'large', 'huge']
        print("💪 Running STRESS TEST")
        print(f"   Threads: {thread_counts}")
        print(f"   Sizes: {problem_sizes}")
        print(f"   Iterations: {iterations}")
        total_configs = len(runner.benchmarks) * len(thread_counts) * len(problem_sizes) * iterations
        print(f"   Total runs: {total_configs}")
        print("⚠️  WARNING: This stress test can take several hours to complete!")
        print("💡 Use 'python3 monitor_progress.py' in another terminal to monitor progress")
        print("")
    elif args.extreme_test:
        thread_counts = [1, 2, 4, 8, 16, 24]
        iterations = 3
        benchmarks = None  # All benchmarks
        problem_sizes = ['huge', 'extreme']
        print("🚀 Running EXTREME SCALABILITY TEST")
        print(f"   Threads: {thread_counts}")
        print(f"   Sizes: {problem_sizes}")
        print(f"   Iterations: {iterations}")
        total_configs = len(runner.benchmarks) * len(thread_counts) * len(problem_sizes) * iterations
        print(f"   Total runs: {total_configs}")
        print("🔥 WARNING: This test uses HUGE (256MB) and EXTREME (1GB) problems!")
        print("💾 Ensure you have sufficient RAM and disk space")
        print("💡 Use 'python3 monitor_progress.py' in another terminal to monitor progress")
        print("")
    else:
        # Parse thread counts
        if args.threads.lower() == 'auto':
            import multiprocessing
            max_threads = multiprocessing.cpu_count()
            thread_counts = [1, 2, 4, 8, min(16, max_threads), min(24, max_threads)]
            thread_counts = sorted(list(set(thread_counts)))  # Remove duplicates
        else:
            thread_counts = [int(x.strip()) for x in args.threads.split(',')]
        
        # Parse benchmark selection
        if args.benchmarks.lower() == 'all':
            benchmarks = None  # Will use all available
        else:
            benchmarks = [x.strip() for x in args.benchmarks.split(',')]
        
        # Parse problem sizes
        problem_sizes = [x.strip() for x in args.problem_sizes.split(',')]
        iterations = args.iterations
    
    # Run benchmarks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🧮 OmpSCR Benchmark Runner")
    print("=" * 50)
    
    try:
        runner.run_benchmarks(
            thread_counts=thread_counts,
            iterations=iterations,
            benchmarks=benchmarks,
            problem_sizes=problem_sizes
        )
        
        runner.save_results(timestamp)
        
        # Quick analysis
        if runner.results:
            successful = sum(1 for r in runner.results if r['success'])
            total = len(runner.results)
            print(f"\n📊 Quick Stats:")
            print(f"   Successful runs: {successful}/{total} ({(successful/total)*100:.1f}%)")
            
            if successful > 0:
                avg_time = sum(r['wall_time'] for r in runner.results if r['success']) / successful
                print(f"   Average runtime: {avg_time:.3f}s")
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark run interrupted by user")
        if runner.results:
            runner.save_results(f"{timestamp}_interrupted")
            print("💾 Partial results saved")
    except Exception as e:
        print(f"\n❌ Error during benchmark execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
