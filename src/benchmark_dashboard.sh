#!/bin/bash

# OmpSCR Benchmark Progress Dashboard
# Shows live progress and allows background execution monitoring

echo "🧮 OmpSCR Benchmark Progress Dashboard"
echo "====================================="

# Check if benchmark is running
if pgrep -f "benchmark_runner.py" > /dev/null; then
    echo "✅ Benchmark is currently running"
    echo ""
    
    # Show options
    echo "Select monitoring option:"
    echo "1) Live progress monitor (real-time updates)"
    echo "2) Quick status check" 
    echo "3) Show summary of completed runs"
    echo "4) Show latest results"
    echo "5) Kill running benchmark (emergency stop)"
    echo ""
    
    read -p "Choose option (1-5): " choice
    
    case $choice in
        1)
            echo "🔄 Starting live monitor (Press Ctrl+C to exit)..."
            sleep 1
            python3 monitor_progress.py
            ;;
        2)
            echo "📊 Current status:"
            if ls benchmark_results/progress_*.json 1> /dev/null 2>&1; then
                latest_progress=$(ls -t benchmark_results/progress_*.json | head -n1)
                python3 -c "
import json
with open('$latest_progress', 'r') as f:
    p = json.load(f)
print(f\"Progress: {p['completed_runs']}/{p['total_runs']} ({p['progress_pct']:.1f}%)\")
print(f\"Current: {p['current_benchmark']} - {p['current_config']}\")
print(f\"Elapsed: {p['elapsed_time_s']/60:.1f}m, ETA: {p['eta_s']/60:.1f}m\")
print(f\"Success rate: {p['successful_runs']}/{p['completed_runs']} ({p['successful_runs']/max(1,p['completed_runs'])*100:.1f}%)\")
"
            else
                echo "❌ No progress file found"
            fi
            ;;
        3)
            echo "📈 Summary of completed runs:"
            python3 monitor_progress.py --summary
            ;;
        4)
            echo "📋 Latest results:"
            if ls benchmark_results/benchmark_results_*.json 1> /dev/null 2>&1; then
                latest=$(ls -t benchmark_results/benchmark_results_*.json | head -n1)
                echo "File: $(basename $latest)"
                python3 -c "
import json
with open('$latest', 'r') as f:
    results = json.load(f)
print(f'Total runs: {len(results)}')
successful = sum(1 for r in results if r['success'])
print(f'Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)')
if successful > 0:
    avg_time = sum(r['wall_time'] for r in results if r['success']) / successful
    print(f'Average time: {avg_time:.3f}s')
"
            else
                echo "❌ No results found"
            fi
            ;;
        5)
            echo "⚠️  Stopping benchmark execution..."
            pkill -f "benchmark_runner.py"
            echo "✅ Benchmark stopped"
            ;;
        *)
            echo "❌ Invalid option"
            ;;
    esac
    
else
    echo "❌ No benchmark currently running"
    echo ""
    
    # Check for existing results
    if ls benchmark_results/*.json 1> /dev/null 2>&1; then
        echo "📁 Found existing results:"
        echo "1) Show summary of completed runs"
        echo "2) Start new benchmark run" 
        echo "3) Quick test (1,2,4 threads, tiny/small/medium sizes, 3 iterations)"
        echo "4) Full comprehensive test (1,2,4,8,12,16,24 threads, all sizes, 10 iterations)"
        echo "5) Stress test (1,2,4,8,16,24,32 threads, all sizes including extreme, 15 iterations)"
        echo ""
        
        read -p "Choose option (1-5): " choice
        
        case $choice in
            1)
                python3 monitor_progress.py --summary
                ;;
            2)
                echo "🚀 Starting new benchmark..."
                echo "Example commands:"
                echo "• Quick:         python3 benchmark_runner.py --quick-test"
                echo "• Custom:        python3 benchmark_runner.py --threads 1,2,4 --iterations 3 --problem-sizes tiny,small"
                echo "• Comprehensive: python3 benchmark_runner.py --full-test"
                ;;
            3)
                echo "⚡ Starting quick test..."
                echo "This will run ~459 configurations and take ~30-60 minutes."
                read -p "Continue? (y/N): " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    nohup python3 benchmark_runner.py --quick-test > benchmark_quick.log 2>&1 &
                    echo "✅ Quick test started in background"
                    echo "📁 Log: benchmark_quick.log"
                    echo "🔄 Monitor with: python3 monitor_progress.py"
                else
                    echo "❌ Test cancelled"
                fi
                ;;
            4)
                echo "🔬 Starting comprehensive test..."
                echo "This will run ~7140 configurations and may take several hours."
                read -p "Continue? (y/N): " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    nohup python3 benchmark_runner.py --full-test > benchmark_comprehensive.log 2>&1 &
                    echo "✅ Comprehensive test started in background"
                    echo "📁 Log: benchmark_comprehensive.log"
                    echo "🔄 Monitor with: python3 monitor_progress.py"
                else
                    echo "❌ Test cancelled"
                fi
                ;;
            5)
                echo "💪 Starting stress test..."
                echo "This will run ~15750 configurations and may take many hours!"
                read -p "Are you absolutely sure? This is a VERY long test. (y/N): " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    nohup python3 benchmark_runner.py --stress-test > benchmark_stress.log 2>&1 &
                    echo "✅ Stress test started in background"
                    echo "📁 Log: benchmark_stress.log"
                    echo "🔄 Monitor with: python3 monitor_progress.py"
                else
                    echo "❌ Test cancelled"
                fi
                ;;
            *)
                echo "❌ Invalid option"
                ;;
        esac
    else
        echo "📝 No previous results found"
        echo "🚀 Ready to start new benchmark run"
        echo ""
        echo "Example commands:"
        echo "• Quick test:        python3 benchmark_runner.py --threads 1,2,4 --iterations 1"
        echo "• Comprehensive:     python3 benchmark_runner.py --threads 1,2,4,8,12,16,24 --iterations 10"
        echo "• Monitor progress:  python3 monitor_progress.py"
    fi
fi

echo ""
echo "💡 Tip: Use 'python3 monitor_progress.py' to monitor any running benchmark"
