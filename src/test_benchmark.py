#!/usr/bin/env python3
"""
Quick Test Script for OpenMP Benchmark Suite
===========================================

This script runs a minimal test to verify the benchmark suite works correctly.
It tests with reduced iterations and limited thread configurations.

Author: Benchmark Test Runner
Date: November 2024
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run a quick test of the benchmark suite."""
    print("OpenMP Benchmark Suite - Quick Test")
    print("=" * 40)
    print()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    benchmark_script = script_dir / "benchmark_runner.py"
    
    if not benchmark_script.exists():
        print(f"Error: Benchmark script not found at {benchmark_script}")
        return 1
    
    print("Running quick test with minimal configuration...")
    print("- Thread configurations: 1, 2, 4")
    print("- Iterations per config: 2")
    print("- Applications: c_Pi only (fastest benchmark)")
    print()
    
    # Run the benchmark with test parameters
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--threads", "1,2,4",
        "--iterations", "2", 
        "--apps", "c_Pi"
    ]
    
    try:
        print("Starting test run...")
        result = subprocess.run(cmd, check=True)
        
        print()
        print("✓ Quick test completed successfully!")
        print()
        print("Next steps:")
        print("1. Check the results in the 'benchmark_results' directory")
        print("2. Run the full benchmark suite:")
        print("   python3 benchmark_runner.py")
        print("3. Analyze results with:")
        print("   python3 analyze_results.py benchmark_results/benchmark_results_*.csv")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Test failed with exit code {e.returncode}")
        print("\nTroubleshooting:")
        print("1. Check that you're running on a Linux system with GCC installed")
        print("2. Ensure you have sudo privileges for package installation")
        print("3. Review the log files in the benchmark_results directory")
        return 1
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())