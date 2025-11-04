#!/usr/bin/env python3
"""
OpenMP Benchmark Suite Runner
============================

This script automatically runs the OmpSCR v2.0 benchmark suite with different thread configurations.
It handles dependency installation, compilation, and execution with multiple iterations per configuration.
Results are stored in CSV format for analysis.

Requirements:
- Linux Mint or compatible Ubuntu-based system
- Python 3.6+
- sudo privileges for package installation

Author: Automated Benchmark Runner
Date: November 2024
"""

import os
import sys
import subprocess
import shutil
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import logging
import re


class BenchmarkRunner:
    """Main class for running OpenMP benchmarks with different thread configurations."""
    
    def __init__(self, base_dir: str = None, output_dir: str = None):
        """Initialize the benchmark runner."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / "benchmark_results"
        self.log_dir = self.base_dir / "log"
        self.bin_dir = self.base_dir / "bin"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.bin_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Benchmark applications (excluding GraphSearch generators)
        self.applications = [
            "c_FFT",
            "c_FFT6", 
            "c_Jacobi",
            "c_LoopsWithDependencies",
            "c_LUreduction",
            "c_Mandelbrot",
            "c_MolecularDynamic",
            "c_Pi",
            "c_QuickSort"
        ]
        
        # Thread configurations to test
        self.thread_configs = [1, 2, 4, 8, 16, 32]
        
        # Number of iterations per configuration
        self.iterations = 10
        
        # Results storage
        self.results = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_dependencies(self) -> bool:
        """Check and install required system dependencies for Linux Mint."""
        self.logger.info("Checking system dependencies...")
        
        required_packages = [
            "build-essential",
            "gcc",
            "g++",
            "gfortran", 
            "libomp-dev",
            "make",
            "git",
            "time"
        ]
        
        try:
            # Update package list
            self.logger.info("Updating package list...")
            subprocess.run(["sudo", "apt", "update"], check=True, capture_output=True)
            
            # Check which packages are missing
            missing_packages = []
            for package in required_packages:
                result = subprocess.run(
                    ["dpkg", "-l", package], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode != 0:
                    missing_packages.append(package)
            
            # Install missing packages
            if missing_packages:
                self.logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
                cmd = ["sudo", "apt", "install", "-y"] + missing_packages
                subprocess.run(cmd, check=True)
            else:
                self.logger.info("All required packages are already installed.")
                
            # Verify critical tools
            critical_tools = ["gcc", "make", "time"]
            for tool in critical_tools:
                if not shutil.which(tool):
                    self.logger.error(f"Critical tool '{tool}' not found after installation!")
                    return False
                    
            self.logger.info("All dependencies satisfied.")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error checking dependencies: {e}")
            return False
    
    def setup_compiler_config(self) -> bool:
        """Setup compiler configuration for the benchmark suite."""
        self.logger.info("Setting up compiler configuration...")
        
        try:
            config_file = self.base_dir / "config" / "templates" / "user.cf.mk"
            gnu_config = self.base_dir / "config" / "templates" / "gnu.cf.mk"
            
            if not gnu_config.exists():
                self.logger.error(f"GNU config template not found: {gnu_config}")
                return False
                
            # Copy GNU config as user config
            shutil.copy2(gnu_config, config_file)
            self.logger.info(f"Copied GNU configuration to {config_file}")
            
            # Verify GCC supports OpenMP
            result = subprocess.run(
                ["gcc", "--help=target"],
                capture_output=True, text=True
            )
            
            if "fopenmp" not in result.stdout:
                self.logger.warning("GCC may not support OpenMP. Checking libomp-dev...")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup compiler configuration: {e}")
            return False
    
    def build_benchmarks(self) -> bool:
        """Build all benchmark applications."""
        self.logger.info("Building benchmark applications...")
        
        try:
            # Change to base directory
            original_dir = os.getcwd()
            os.chdir(self.base_dir)
            
            # Clean previous builds
            self.logger.info("Cleaning previous builds...")
            subprocess.run(["make", "clean"], check=True, capture_output=True)
            
            # Build all applications (parallel versions)
            self.logger.info("Building parallel versions...")
            result = subprocess.run(
                ["make", "par"], 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"Build failed: {result.stderr}")
                return False
                
            # Verify executables were created
            built_apps = []
            for app in self.applications:
                exe_pattern = f"{app}*.par.gnu"
                exe_files = list(self.bin_dir.glob(exe_pattern))
                if exe_files:
                    built_apps.append(app)
                    self.logger.info(f"Built: {exe_files[0].name}")
                else:
                    self.logger.warning(f"No executable found for {app}")
            
            if len(built_apps) == 0:
                self.logger.error("No applications were successfully built!")
                return False
                
            self.applications = built_apps  # Update to only successfully built apps
            self.logger.info(f"Successfully built {len(built_apps)} applications")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Build process timed out after 5 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Build process failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def get_executable_path(self, app_name: str) -> Optional[Path]:
        """Get the path to the executable for a given application."""
        exe_pattern = f"{app_name}*.par.gnu"
        exe_files = list(self.bin_dir.glob(exe_pattern))
        return exe_files[0] if exe_files else None
    
    def run_single_benchmark(self, app_name: str, threads: int, iteration: int) -> Dict:
        """Run a single benchmark iteration and return timing results."""
        exe_path = self.get_executable_path(app_name)
        if not exe_path:
            raise FileNotFoundError(f"Executable not found for {app_name}")
        
        # Set environment variables
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_DISPLAY_ENV"] = "false"
        
        # Run the benchmark with timing
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["/usr/bin/time", "-v", str(exe_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per run
            )
            
            end_time = time.time()
            
            # Parse timing information from /usr/bin/time output
            timing_info = self.parse_time_output(result.stderr)
            
            return {
                "application": app_name,
                "threads": threads,
                "iteration": iteration,
                "wall_time": end_time - start_time,
                "cpu_time": timing_info.get("user_time", 0) + timing_info.get("system_time", 0),
                "user_time": timing_info.get("user_time", 0),
                "system_time": timing_info.get("system_time", 0),
                "max_memory_kb": timing_info.get("max_memory_kb", 0),
                "exit_code": result.returncode,
                "stdout": result.stdout[:500],  # Limit output size
                "stderr": result.stderr[:500] if result.returncode != 0 else "",
                "timestamp": datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout for {app_name} with {threads} threads (iteration {iteration})")
            return {
                "application": app_name,
                "threads": threads,
                "iteration": iteration,
                "wall_time": 120.0,  # Timeout value
                "cpu_time": -1,
                "user_time": -1,
                "system_time": -1,
                "max_memory_kb": -1,
                "exit_code": -1,
                "stdout": "",
                "stderr": "TIMEOUT",
                "timestamp": datetime.now().isoformat()
            }
    
    def parse_time_output(self, time_stderr: str) -> Dict:
        """Parse output from /usr/bin/time -v command."""
        timing_info = {}
        
        patterns = {
            "user_time": r"User time \(seconds\): ([\d.]+)",
            "system_time": r"System time \(seconds\): ([\d.]+)",
            "max_memory_kb": r"Maximum resident set size \(kbytes\): (\d+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, time_stderr)
            if match:
                try:
                    timing_info[key] = float(match.group(1))
                except ValueError:
                    timing_info[key] = 0
            else:
                timing_info[key] = 0
                
        return timing_info
    
    def run_benchmarks(self) -> bool:
        """Run all benchmarks with different thread configurations."""
        self.logger.info("Starting benchmark execution...")
        
        total_runs = len(self.applications) * len(self.thread_configs) * self.iterations
        current_run = 0
        
        for app in self.applications:
            self.logger.info(f"Running benchmarks for {app}...")
            
            for threads in self.thread_configs:
                self.logger.info(f"  Testing with {threads} threads...")
                
                for iteration in range(1, self.iterations + 1):
                    current_run += 1
                    progress = (current_run / total_runs) * 100
                    
                    self.logger.info(f"    Iteration {iteration}/{self.iterations} ({progress:.1f}% complete)")
                    
                    try:
                        result = self.run_single_benchmark(app, threads, iteration)
                        self.results.append(result)
                        
                        # Log performance for this run
                        if result["exit_code"] == 0:
                            self.logger.info(f"      Completed in {result['wall_time']:.3f}s")
                        else:
                            self.logger.warning(f"      Failed with exit code {result['exit_code']}")
                            
                    except Exception as e:
                        self.logger.error(f"      Error running {app}: {e}")
                        # Add error result
                        self.results.append({
                            "application": app,
                            "threads": threads,
                            "iteration": iteration,
                            "wall_time": -1,
                            "cpu_time": -1,
                            "user_time": -1,
                            "system_time": -1,
                            "max_memory_kb": -1,
                            "exit_code": -1,
                            "stdout": "",
                            "stderr": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Brief pause between thread configurations
                time.sleep(1)
        
        self.logger.info(f"Completed all {total_runs} benchmark runs")
        return True
    
    def save_results(self) -> bool:
        """Save benchmark results to CSV and JSON files."""
        self.logger.info("Saving benchmark results...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save to CSV
            csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
            with open(csv_file, 'w', newline='') as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.results)
            
            # Save to JSON for more detailed analysis
            json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Generate summary statistics
            self.generate_summary(timestamp)
            
            self.logger.info(f"Results saved to:")
            self.logger.info(f"  CSV: {csv_file}")
            self.logger.info(f"  JSON: {json_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False
    
    def generate_summary(self, timestamp: str):
        """Generate a summary report of the benchmark results."""
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        
        try:
            with open(summary_file, 'w') as f:
                f.write("OpenMP Benchmark Suite Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Runs: {len(self.results)}\n")
                f.write(f"Applications: {', '.join(self.applications)}\n")
                f.write(f"Thread Configurations: {', '.join(map(str, self.thread_configs))}\n")
                f.write(f"Iterations per Configuration: {self.iterations}\n\n")
                
                # Success rate analysis
                successful_runs = [r for r in self.results if r["exit_code"] == 0]
                success_rate = (len(successful_runs) / len(self.results)) * 100
                f.write(f"Success Rate: {success_rate:.1f}% ({len(successful_runs)}/{len(self.results)})\n\n")
                
                # Performance summary by application
                f.write("Performance Summary by Application:\n")
                f.write("-" * 40 + "\n")
                
                for app in self.applications:
                    app_results = [r for r in successful_runs if r["application"] == app]
                    if app_results:
                        avg_time = sum(r["wall_time"] for r in app_results) / len(app_results)
                        min_time = min(r["wall_time"] for r in app_results)
                        max_time = max(r["wall_time"] for r in app_results)
                        
                        f.write(f"{app}:\n")
                        f.write(f"  Average Time: {avg_time:.3f}s\n")
                        f.write(f"  Min Time: {min_time:.3f}s\n")
                        f.write(f"  Max Time: {max_time:.3f}s\n")
                        f.write(f"  Successful Runs: {len(app_results)}\n\n")
                
                # Thread scaling analysis
                f.write("Thread Scaling Analysis:\n")
                f.write("-" * 30 + "\n")
                
                for threads in self.thread_configs:
                    thread_results = [r for r in successful_runs if r["threads"] == threads]
                    if thread_results:
                        avg_time = sum(r["wall_time"] for r in thread_results) / len(thread_results)
                        f.write(f"{threads} threads: {avg_time:.3f}s average\n")
                
            self.logger.info(f"Summary report saved to: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
    
    def run_full_benchmark(self) -> bool:
        """Run the complete benchmark suite."""
        self.logger.info("Starting OpenMP Benchmark Suite Runner")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            self.logger.error("Dependency check failed")
            return False
        
        # Step 2: Setup compiler configuration
        if not self.setup_compiler_config():
            self.logger.error("Compiler configuration failed")
            return False
        
        # Step 3: Build benchmarks
        if not self.build_benchmarks():
            self.logger.error("Build process failed")
            return False
        
        # Step 4: Run benchmarks
        if not self.run_benchmarks():
            self.logger.error("Benchmark execution failed")
            return False
        
        # Step 5: Save results
        if not self.save_results():
            self.logger.error("Failed to save results")
            return False
        
        self.logger.info("Benchmark suite completed successfully!")
        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="OpenMP Benchmark Suite Runner for Linux Mint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 benchmark_runner.py                    # Run with default settings
  python3 benchmark_runner.py --threads 1,2,4,8 # Custom thread configurations
  python3 benchmark_runner.py --iterations 5     # 5 iterations per config
  python3 benchmark_runner.py --output /tmp/results # Custom output directory
        """
    )
    
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of thread counts to test (default: 1,2,4,8,16,32)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per thread configuration (default: 10)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (default: ./benchmark_results)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        help="Base directory of OmpSCR benchmark suite (default: script directory)"
    )
    
    parser.add_argument(
        "--apps",
        type=str,
        help="Comma-separated list of applications to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Parse thread configurations
    try:
        thread_configs = [int(x.strip()) for x in args.threads.split(',')]
    except ValueError:
        print("Error: Invalid thread configuration format")
        return 1
    
    # Create benchmark runner
    runner = BenchmarkRunner(base_dir=args.base_dir, output_dir=args.output)
    
    # Override configurations if specified
    runner.thread_configs = thread_configs
    runner.iterations = args.iterations
    
    if args.apps:
        try:
            custom_apps = [x.strip() for x in args.apps.split(',')]
            runner.applications = custom_apps
        except ValueError:
            print("Error: Invalid application list format")
            return 1
    
    # Run the benchmark suite
    success = runner.run_full_benchmark()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())