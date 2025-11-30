#!/usr/bin/env python3
"""
Script to update LaTeX tables and figures with latest benchmark results
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import math

def load_benchmark_data(json_file):
    """Load benchmark results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def calculate_speedup_efficiency(data):
    """Calculate speedup and efficiency for each benchmark"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for entry in data:
        if not entry.get('success'):
            continue
            
        benchmark = entry['benchmark']
        threads = entry['threads']
        size = entry['problem_size']
        wall_time = entry['wall_time']
        
        results[benchmark][size][threads].append(wall_time)
    
    # Calculate averages and speedup
    speedup_data = {}
    for benchmark in results:
        speedup_data[benchmark] = {}
        for size in results[benchmark]:
            speedup_data[benchmark][size] = {}
            # Get baseline (1 thread)
            if 1 in results[benchmark][size]:
                baseline = sum(results[benchmark][size][1]) / len(results[benchmark][size][1])
                
                for threads in sorted(results[benchmark][size].keys()):
                    avg_time = sum(results[benchmark][size][threads]) / len(results[benchmark][size][threads])
                    speedup = baseline / avg_time if avg_time > 0 else 0
                    efficiency = (speedup / threads) * 100 if threads > 0 else 0
                    
                    speedup_data[benchmark][size][threads] = {
                        'time': avg_time,
                        'speedup': speedup,
                        'efficiency': efficiency,
                        'baseline': baseline
                    }
    
    return speedup_data

def generate_speedup_table(speedup_data, benchmarks, size='extreme'):
    """Generate LaTeX speedup table for a specific problem size"""
    
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Speedup e Eficiência das Aplicações (problema \\textit{{{size}}})}}")
    latex.append(f"\\label{{tab:speedup_{size}}}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Aplicação} & \\multicolumn{7}{c}{\\textbf{Threads / Speedup (Eficiência \\%)}} \\\\")
    latex.append("\\cmidrule(lr){2-8}")
    latex.append(" & 1 & 2 & 4 & 8 & 12 & 16 & 24 \\\\")
    latex.append("\\midrule")
    
    for bench in benchmarks:
        if bench not in speedup_data or size not in speedup_data[bench]:
            continue
            
        row = [bench.replace('_', '\\_')]
        for threads in [1, 2, 4, 8, 12, 16, 24]:
            if threads in speedup_data[bench][size]:
                data = speedup_data[bench][size][threads]
                speedup = data['speedup']
                efficiency = data['efficiency']
                row.append(f"{speedup:.2f} ({efficiency:.1f}\\%)")
            else:
                row.append("---")
        
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_comparison_table(speedup_data):
    """Generate comparison table with best results for each benchmark"""
    
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\caption{Resumo comparativo das aplicações analisadas (dados atualizados)}")
    latex.append("\\label{tab:resultados-gerais-updated}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lccccr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Aplicação} & \\textbf{Efic. Média} & \\textbf{Speedup Máx.} & \\textbf{Threads} & \\textbf{Tempo (s)} & \\textbf{Avaliação} \\\\")
    latex.append("\\midrule")
    
    benchmarks_summary = {
        'c_mandel': 'Excelente',
        'c_md': 'Excelente',
        'c_pi': 'Boa',
        'c_fft': 'Limitada',
        'c_qsort': 'Limitada',
        'c_fft6': 'Moderada'
    }
    
    for bench in ['c_mandel', 'c_md', 'c_pi', 'c_fft6', 'c_fft', 'c_qsort']:
        if bench not in speedup_data or 'extreme' not in speedup_data[bench]:
            continue
        
        # Find best speedup
        max_speedup = 0
        max_threads = 1
        avg_efficiency = 0
        best_time = 0
        count = 0
        
        for threads in speedup_data[bench]['extreme']:
            data = speedup_data[bench]['extreme'][threads]
            if data['speedup'] > max_speedup:
                max_speedup = data['speedup']
                max_threads = threads
                best_time = data['time']
            
            if threads > 1:  # Don't count baseline in efficiency average
                avg_efficiency += data['efficiency']
                count += 1
        
        if count > 0:
            avg_efficiency /= count
        
        bench_display = bench.replace('_', '\\_')
        rating = benchmarks_summary.get(bench, 'Moderada')
        
        latex.append(f"{bench_display} & {avg_efficiency:.1f}\\% & {max_speedup:.2f}× & {max_threads} & {best_time:.2f} & {rating} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_size_comparison_table(speedup_data, benchmark='c_mandel', threads=24):
    """Generate table comparing different problem sizes"""
    
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Eficiência por tamanho de problema ({benchmark.replace('_', '\\_')}, {threads} threads)}}")
    latex.append(f"\\label{{tab:size_comparison_{benchmark}}}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Tamanho} & \\textbf{Tempo (s)} & \\textbf{Speedup} & \\textbf{Eficiência (\\%)} & \\textbf{Pontos/Iterações} \\\\")
    latex.append("\\midrule")
    
    sizes_info = {
        'small': '200k pontos',
        'medium': '300k pontos',
        'large': '500k pontos',
        'huge': '1M pontos',
        'extreme': '2M pontos'
    }
    
    if benchmark in speedup_data:
        for size in ['small', 'medium', 'large', 'huge', 'extreme']:
            if size in speedup_data[benchmark] and threads in speedup_data[benchmark][size]:
                data = speedup_data[benchmark][size][threads]
                size_display = size.capitalize()
                latex.append(f"{size_display} & {data['time']:.2f} & {data['speedup']:.2f}× & {data['efficiency']:.1f}\\% & {sizes_info[size]} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def main():
    # Load data
    results_file = Path("benchmark_results/benchmark_results_20251129_154041.json")
    data = load_benchmark_data(results_file)
    
    print(f"Loaded {len(data)} benchmark results")
    
    # Calculate speedup and efficiency
    speedup_data = calculate_speedup_efficiency(data)
    
    # Generate LaTeX tables
    output_dir = Path("texto/Conteudo/Generated_Tables")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Speedup table for extreme size
    table1 = generate_speedup_table(
        speedup_data,
        ['c_mandel', 'c_md', 'c_pi', 'c_fft', 'c_fft6', 'c_qsort'],
        'extreme'
    )
    with open(output_dir / "speedup_extreme.tex", 'w') as f:
        f.write(table1)
    print("Generated: speedup_extreme.tex")
    
    # 2. Comparison table
    table2 = generate_comparison_table(speedup_data)
    with open(output_dir / "comparison_summary.tex", 'w') as f:
        f.write(table2)
    print("Generated: comparison_summary.tex")
    
    # 3. Size comparison for c_mandel
    table3 = generate_size_comparison_table(speedup_data, 'c_mandel', 24)
    with open(output_dir / "size_comparison_mandel.tex", 'w') as f:
        f.write(table3)
    print("Generated: size_comparison_mandel.tex")
    
    # 4. Size comparison for c_md
    table4 = generate_size_comparison_table(speedup_data, 'c_md', 24)
    with open(output_dir / "size_comparison_md.tex", 'w') as f:
        f.write(table4)
    print("Generated: size_comparison_md.tex")
    
    print("\n✅ All LaTeX tables generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("\nYou can now include these tables in your LaTeX document using:")
    print("  \\input{Conteudo/Generated_Tables/speedup_extreme.tex}")

if __name__ == "__main__":
    main()
