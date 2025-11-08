#!/usr/bin/env python3
"""
Progress Monitor for OmpSCR Benchmark Runner
Real-time monitoring of benchmark execution progress
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def format_time(seconds):
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def monitor_progress(benchmark_dir="benchmark_results", refresh_interval=5):
    """Monitor benchmark progress in real-time"""
    benchmark_dir = Path(benchmark_dir)
    
    print("🔍 OmpSCR Benchmark Progress Monitor")
    print("=" * 50)
    print(f"📁 Monitoring directory: {benchmark_dir}")
    print(f"🔄 Refresh interval: {refresh_interval}s")
    print("📊 Press Ctrl+C to exit")
    print("=" * 50)
    
    last_progress_file = None
    
    try:
        while True:
            # Find the most recent progress file
            progress_files = list(benchmark_dir.glob("progress_*.json"))
            
            if not progress_files:
                print("⏳ Waiting for benchmark to start...")
                time.sleep(refresh_interval)
                continue
            
            # Get the most recent progress file
            current_progress_file = max(progress_files, key=lambda p: p.stat().st_mtime)
            
            if current_progress_file != last_progress_file:
                print(f"\n📈 Monitoring: {current_progress_file.name}")
                last_progress_file = current_progress_file
            
            try:
                with open(current_progress_file, 'r') as f:
                    progress = json.load(f)
                
                # Clear screen (ANSI escape codes)
                print("\033[2J\033[H", end="")
                
                # Display header
                print("🧮 OmpSCR Benchmark Progress Monitor")
                print("=" * 60)
                print(f"📁 File: {current_progress_file.name}")
                print(f"🕐 Last update: {progress['timestamp']}")
                print("=" * 60)
                
                # Progress bar
                pct = progress['progress_pct']
                bar_width = 40
                filled = int(bar_width * pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"📊 Progress: {pct:5.1f}% [{bar}]")
                
                # Stats
                print(f"✅ Completed: {progress['completed_runs']:4d} / {progress['total_runs']:4d}")
                print(f"🎯 Success:   {progress['successful_runs']:4d}")
                print(f"❌ Failed:    {progress['failed_runs']:4d}")
                
                # Time information
                elapsed = format_time(progress['elapsed_time_s'])
                eta = format_time(progress['eta_s'])
                print(f"⏱️  Elapsed:   {elapsed}")
                print(f"🎯 ETA:       {eta}")
                
                # Current status
                print(f"🔄 Current:   {progress['current_benchmark']}")
                print(f"⚙️  Config:    {progress['current_config']}")
                
                # Performance estimate
                if progress['completed_runs'] > 0:
                    rate = progress['completed_runs'] / progress['elapsed_time_s']
                    print(f"🚀 Rate:      {rate:.2f} runs/second")
                
                print("=" * 60)
                print("📊 Press Ctrl+C to exit monitor")
                
                # Check if benchmark is complete
                if progress['completed_runs'] >= progress['total_runs']:
                    print("\n🎉 Benchmark execution completed!")
                    break
                    
            except json.JSONDecodeError:
                print("⚠️  Invalid progress file format")
            except FileNotFoundError:
                print("⚠️  Progress file disappeared")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")

def show_summary(benchmark_dir="benchmark_results"):
    """Show summary of completed benchmarks"""
    benchmark_dir = Path(benchmark_dir)
    
    # Find result files
    result_files = list(benchmark_dir.glob("benchmark_results_*.json"))
    
    if not result_files:
        print("❌ No completed benchmark results found")
        return
    
    print("📊 Benchmark Results Summary")
    print("=" * 50)
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            if not results:
                continue
                
            total_runs = len(results)
            successful = sum(1 for r in results if r['success'])
            
            # Get time range
            timestamps = [r['timestamp'] for r in results]
            start_time = min(timestamps)
            end_time = max(timestamps)
            
            print(f"\n📁 File: {result_file.name}")
            print(f"🕐 Time: {start_time} to {end_time}")
            print(f"✅ Success: {successful}/{total_runs} ({successful/total_runs*100:.1f}%)")
            
            # Show benchmark breakdown
            benchmarks = {}
            for r in results:
                name = r['benchmark']
                if name not in benchmarks:
                    benchmarks[name] = {'total': 0, 'success': 0}
                benchmarks[name]['total'] += 1
                if r['success']:
                    benchmarks[name]['success'] += 1
            
            print("📈 By benchmark:")
            for name, stats in sorted(benchmarks.items()):
                rate = stats['success'] / stats['total'] * 100
                print(f"   {name:15}: {stats['success']:3d}/{stats['total']:3d} ({rate:5.1f}%)")
                
        except Exception as e:
            print(f"⚠️  Error reading {result_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Monitor OmpSCR benchmark progress')
    parser.add_argument('--dir', type=str, default='benchmark_results',
                        help='Benchmark results directory (default: benchmark_results)')
    parser.add_argument('--interval', type=int, default=5,
                        help='Refresh interval in seconds (default: 5)')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary of completed benchmarks and exit')
    
    args = parser.parse_args()
    
    if args.summary:
        show_summary(args.dir)
    else:
        monitor_progress(args.dir, args.interval)

if __name__ == "__main__":
    main()
