#!/usr/bin/env python3
"""
Quick test script for granularity variants
Tests all fine/coarse variants with 1, 2, and 4 threads
"""

import subprocess
import sys
import os

def run_test(binary, threads, description):
    """Run a single test with specified threads"""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"Binary: {binary}")
        print(f"Threads: {threads}")
        print(f"{'='*60}")
        
        result = subprocess.run(
            [binary, '-test'],
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✓ PASSED")
            return True
        else:
            print(f"✗ FAILED (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT")
        return False
    except FileNotFoundError:
        print("✗ BINARY NOT FOUND")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def main():
    """Run quick tests for all granularity variants"""
    
    # Test configuration
    tests = [
        # Pi variants
        ('bin/c_pi.par.gnu', 'Pi - Standard'),
        ('bin/c_pi_fine.par.gnu', 'Pi - Fine-grained'),
        ('bin/c_pi_coarse.par.gnu', 'Pi - Coarse-grained'),
        
        # Mandelbrot variants
        ('bin/c_mandel.par.gnu', 'Mandelbrot - Standard'),
        ('bin/c_mandel_fine.par.gnu', 'Mandelbrot - Fine-grained'),
        ('bin/c_mandel_coarse.par.gnu', 'Mandelbrot - Coarse-grained'),
        
        # QuickSort variants
        ('bin/c_qsort.par.gnu', 'QuickSort - Standard'),
        ('bin/c_qsort_fine.par.gnu', 'QuickSort - Fine-grained'),
        ('bin/c_qsort_coarse.par.gnu', 'QuickSort - Coarse-grained'),
        
        # FFT variants
        ('bin/c_fft.par.gnu', 'FFT - Standard'),
        ('bin/c_fft_fine.par.gnu', 'FFT - Fine-grained'),
        ('bin/c_fft_coarse.par.gnu', 'FFT - Coarse-grained'),
        
        # Jacobi variants
        ('bin/c_jacobi01.par.gnu', 'Jacobi - Standard'),
        ('bin/c_jacobi_fine.par.gnu', 'Jacobi - Fine-grained'),
        ('bin/c_jacobi_coarse.par.gnu', 'Jacobi - Coarse-grained'),
        
        # LU variants
        ('bin/c_lu.par.gnu', 'LU - Standard'),
        ('bin/c_lu_fine.par.gnu', 'LU - Fine-grained'),
        ('bin/c_lu_coarse.par.gnu', 'LU - Coarse-grained'),
        
        # Molecular Dynamics variants
        ('bin/c_md.par.gnu', 'MD - Standard'),
        ('bin/c_md_fine.par.gnu', 'MD - Fine-grained'),
        ('bin/c_md_coarse.par.gnu', 'MD - Coarse-grained'),
    ]
    
    threads_to_test = [1, 2, 4]
    
    print("\n" + "="*60)
    print("QUICK GRANULARITY VARIANTS TEST")
    print("="*60)
    print(f"Testing {len(tests)} applications with threads: {threads_to_test}")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for binary, description in tests:
        for threads in threads_to_test:
            total_tests += 1
            if run_test(binary, threads, description):
                passed_tests += 1
            else:
                failed_tests.append(f"{description} (threads={threads})")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Failed: {total_tests - passed_tests}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
