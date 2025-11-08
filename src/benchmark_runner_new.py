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

class BenchmarkRunner:
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Problem sizes: small, medium, large
        self.problem_sizes = {
            'small': {'grid_size': 50, 'iterations': 25},
            'medium': {'grid_size': 200, 'iterations': 100}, 
            'large': {'grid_size': 500, 'iterations': 200}
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
                    'small': ['-test'],
                    'medium': ['-test'], 
                    'large': ['-test']
                },
                'description': 'Pi calculation using numerical integration'
            },
            'c_mandel': {
                'binary': 'bin/c_mandel.par.gnu', 
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Mandelbrot set generator'
            },
            'c_qsort': {
                'binary': 'bin/c_qsort.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Parallel quicksort'
            },
            'c_fft': {
                'binary': 'bin/c_fft.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Fast Fourier Transform'
            },
            'c_fft6': {
                'binary': 'bin/c_fft6.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': '6-point FFT implementation'
            },
            'c_md': {
                'binary': 'bin/c_md.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Molecular Dynamics simulation'
            },
            'c_lu': {
                'binary': 'bin/c_lu.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'LU decomposition'
            },
            
            # Jacobi solver variants (with configurable problem sizes)
            'c_jacobi01': {
                'binary': 'bin/c_jacobi01.par.gnu',
                'args_template': {
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver v1'
            },
            'c_jacobi02': {
                'binary': 'bin/c_jacobi02.par.gnu',
                'args_template': {
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver v2'
            },
            'c_jacobi03': {
                'binary': 'bin/c_jacobi03.par.gnu',
                'args_template': {
                    'small': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'medium': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}'],
                    'large': ['{grid_size}', '{grid_size}', '0.8', '1.0', '1e-6', '{iterations}']
                },
                'description': 'Jacobi iterative solver v3'
            },
            
            # Loop dependency examples (correct implementations)
            'c_loopA_sol1': {
                'binary': 'bin/c_loopA.solution1.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Loop A dependency - Solution 1'
            },
            'c_loopA_sol2': {
                'binary': 'bin/c_loopA.solution2.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Loop A dependency - Solution 2'
            },
            'c_loopA_sol3': {
                'binary': 'bin/c_loopA.solution3.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Loop A dependency - Solution 3'
            },
            'c_loopB_pipeline': {
                'binary': 'bin/c_loopB.pipelineSolution.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Loop B dependency - Pipeline Solution'
            },
            
            # Bad implementations (for race detection studies)
            'c_loopA_bad': {
                'binary': 'bin/c_loopA.badSolution.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Loop A dependency - Bad Solution (has races)'
            },
            'c_loopB_bad1': {
                'binary': 'bin/c_loopB.badSolution1.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Loop B dependency - Bad Solution 1 (has races)'
            },
            'c_loopB_bad2': {
                'binary': 'bin/c_loopB.badSolution2.par.gnu',
                'args_template': {
                    'small': ['-test'],
                    'medium': ['-test'],
                    'large': ['-test']
                },
                'description': 'Loop B dependency - Bad Solution 2 (has races)'
            }
        }
        
        # Default thread counts to test
        self.default_threads = [1, 2, 4, 8, 16, 32]
        
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
            else:
                args.append(arg)
        
        return args
    
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
            
        # Set environment
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(thread_count)
        
        print(f"  Running {name} ({problem_size}) with {thread_count} threads (iteration {iteration})...")
        
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
                timeout=600,  # 10 minute timeout for larger problems
                env=env
            )
            
            end_time = time.time()
            wall_time = end_time - start_time
            
            # Parse output for additional metrics
            timing_info = self.extract_timing_info(result.stdout)
            
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
                **timing_info
            }
            
            if result.returncode == 0:
                print(f"    ✓ Completed in {wall_time:.3f}s")
            else:
                print(f"    ✗ Failed with exit code {result.returncode}")
            
            # Update progress counter
            self.completed_runs += 1
            self.update_progress()
                
            return result_data
            
        except subprocess.TimeoutExpired:
            print(f"    ⏱️  Timeout after 10 minutes")
            self.completed_runs += 1
            self.update_progress()
            return {
                'timestamp': datetime.now().isoformat(),
                'benchmark': name,
                'description': config['description'],
                'threads': thread_count,
                'problem_size': problem_size,
                'iteration': iteration,
                'wall_time': 600.0,
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
            'solution_error': r'Solution Error.*?(\d+\.?\d*e?[+-]?\d*)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    info[key] = float(match.group(1))
                except ValueError:
                    info[key] = match.group(1)
        
        return info
    
    def run_benchmarks(self, thread_counts=None, iterations=1, benchmarks=None, problem_sizes=None):
        """Run all benchmarks with specified parameters"""
        if thread_counts is None:
            thread_counts = self.default_threads
            
        if benchmarks is None:
            benchmarks = list(self.benchmarks.keys())
        
        if problem_sizes is None:
            problem_sizes = ['small', 'medium', 'large']
        
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
        """Save results to CSV and JSON files"""
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
        self.generate_summary_report(timestamp)
        
        print(f"📁 Results saved:")
        print(f"   CSV: {csv_file}")
        print(f"   JSON: {json_file}")
        print(f"   Summary: {self.output_dir}/benchmark_summary_{timestamp}.txt")
    
    def generate_summary_report(self, timestamp):
        """Generate a text summary of the benchmark results"""
        if not self.results:
            return
        
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
                
                f.write("Benchmark                Size     Threads  Avg Time (s)  Min Time (s)  Max Time (s)\n")
                f.write("-" * 80 + "\n")
                
                for (benchmark, problem_size, threads), times in sorted(perf_data.items()):
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    f.write(f"{benchmark:20} {problem_size:8} {threads:8d} {avg_time:11.3f} {min_time:11.3f} {max_time:11.3f}\n")

def main():
    parser = argparse.ArgumentParser(description='Run OmpSCR benchmarks with varying parameters')
    parser.add_argument('--threads', type=str, default='1,2,4,8,16,32',
                        help='Comma-separated list of thread counts (default: 1,2,4,8,16,32)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations per configuration (default: 3)')
    parser.add_argument('--benchmarks', type=str, default='all',
                        help='Comma-separated list of benchmarks or "all" (default: all)')
    parser.add_argument('--problem-sizes', type=str, default='small,medium,large',
                        help='Comma-separated list of problem sizes: small,medium,large (default: all)')
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory (default: benchmark_results)')
    parser.add_argument('--list', action='store_true',
                        help='List available benchmarks and exit')
    parser.add_argument('--full-test', action='store_true',
                        help='Run comprehensive test with 1,2,4,8,12,16,24 threads, all sizes, 10 iterations')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.output)
    
    if args.list:
        print("Available benchmarks:")
        print("=" * 50)
        for name, config in runner.benchmarks.items():
            status = "✓" if runner.check_binary_exists(config['binary']) else "✗"
            print(f"{status} {name:20}: {config['description']}")
        print("\nProblem sizes:")
        print("=" * 30)
        for size, config in runner.problem_sizes.items():
            print(f"• {size:8}: grid={config['grid_size']:3d}, iterations={config['iterations']:3d}")
        return
    
    # Handle full test mode
    if args.full_test:
        thread_counts = [1, 2, 4, 8, 12, 16, 24]
        iterations = 10
        benchmarks = None  # All benchmarks
        problem_sizes = ['small', 'medium', 'large']
        print("🔬 Running FULL COMPREHENSIVE TEST")
        print(f"   Threads: {thread_counts}")
        print(f"   Sizes: {problem_sizes}")
        print(f"   Iterations: {iterations}")
        print(f"   Total runs: {len(runner.benchmarks) * len(thread_counts) * len(problem_sizes) * iterations}")
        print("💡 Use 'python3 monitor_progress.py' in another terminal to monitor progress")
        print("")
    else:
        # Parse thread counts
        if args.threads.lower() == 'auto':
            import multiprocessing
            max_threads = multiprocessing.cpu_count()
            thread_counts = [1, 2, 4, 8, min(16, max_threads), min(32, max_threads)]
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
