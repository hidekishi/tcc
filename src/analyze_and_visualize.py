#!/usr/bin/env python3
"""
Consolidated Analysis and Visualization Script
===============================================

Este script unifica toda a análise e visualização de resultados de benchmarks OpenMP.
Substitui 10+ scripts individuais, fornecendo uma interface única para:
  - Cálculo de métricas de desempenho (speedup, eficiência, overhead)
  - Geração de gráficos (speedup, eficiência, overhead, ajuste polinomial)
  - Geração de tabelas LaTeX formatadas
  
Autor: Sistema consolidado de benchmarking OpenMP
Data: 2025-12-01
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

# ============================================================================
# CONFIGURAÇÃO GLOBAL
# ============================================================================

# Diretórios de entrada e saída
RESULTS_DIR = Path("benchmark_results")  # Onde estão os JSONs de benchmark
OUTPUT_DIR = Path("../tcc/Graficos")     # Onde salvar os gráficos PNG
TABLES_DIR = RESULTS_DIR                 # Onde salvar as tabelas LaTeX

# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================================================

def load_benchmark_data(json_file):
    """
    Carrega dados de benchmark de um arquivo JSON.
    
    Args:
        json_file (str|Path): Caminho para o arquivo JSON com resultados
        
    Returns:
        list: Lista de dicionários com resultados de benchmark
        
    Estrutura esperada do JSON:
        [
            {
                "application": "c_mandel",
                "input_size": "large",
                "num_threads": 8,
                "mean_time": 1.234,
                ...
            },
            ...
        ]
    """
    with open(json_file, 'r') as f:
        return json.load(f)

# ============================================================================
# CÁLCULO DE MÉTRICAS
# ============================================================================

def calculate_metrics(data):
    """
    Calcula métricas de desempenho paralelo a partir de dados brutos.
    
    Para cada aplicação e tamanho de problema, calcula:
      - Speedup: S(p) = T₁ / Tₚ
      - Eficiência: E(p) = S(p) / p
      - Overhead: φ = (p × Tₚ - T₁) / T₁
      - Fração Serial: ε = (1/S - 1/p) / (1 - 1/p) [Karp-Flatt]
    
    Args:
        data (list): Lista de resultados de benchmark (output de load_benchmark_data)
        
    Returns:
        dict: Estrutura aninhada {app: {size: {threads: {metrics}}}}
              onde metrics = {'time': float, 'speedup': float, 'efficiency': float,
                             'overhead': float, 'serial_fraction': float}
    """
    # Estrutura temporária para armazenar tempos brutos
    metrics = defaultdict(lambda: defaultdict(dict))
    
    # Primeira passagem: extrair tempos de execução
    for entry in data:
        app = entry['application']
        size = entry['input_size']
        threads = entry['num_threads']
        time = entry['mean_time']
        
        # Armazenar tempo médio para esta configuração
        if threads not in metrics[app][size]:
            metrics[app][size][threads] = time
    
    # Segunda passagem: calcular métricas derivadas
    results = defaultdict(lambda: defaultdict(dict))
    
    for app in metrics:
        for size in metrics[app]:
            # Obter tempo serial (T₁) como baseline
            T1 = metrics[app][size].get(1, None)
            if T1 is None:
                continue  # Pular se não tiver execução serial
                
            # Calcular métricas para cada configuração de threads
            for threads in sorted(metrics[app][size].keys()):
                Tp = metrics[app][size][threads]  # Tempo com p threads
                
                # Speedup: quanto mais rápido ficou comparado ao serial
                speedup = T1 / Tp
                
                # Eficiência: percentual de utilização ideal dos processadores
                efficiency = speedup / threads
                
                # Overhead: tempo extra gasto com sincronização/coordenação
                # φ = (p×Tp - T1) / T1
                overhead = (threads * Tp - T1) / T1
                
                # Fração serial: estimativa de código não paralelizável (Karp-Flatt)
                # ε = (1/S - 1/p) / (1 - 1/p)
                if threads > 1:
                    serial_fraction = (1/speedup - 1/threads) / (1 - 1/threads)
                else:
                    serial_fraction = 0
                
                # Armazenar todas as métricas para esta configuração
                results[app][size][threads] = {
                    'time': Tp,
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'overhead': overhead,
                    'serial_fraction': serial_fraction
                }
    
    return results

# ============================================================================
# GERAÇÃO DE GRÁFICOS
# ============================================================================

def generate_speedup_graphs(metrics, output_dir):
    """
    Gera gráficos de speedup e eficiência para cada aplicação.
    
    Cria figuras com dois painéis lado a lado:
      - Esquerda: Speedup observado vs ideal (linear)
      - Direita: Eficiência (%) vs ideal (100%)
    
    Args:
        metrics (dict): Estrutura de métricas calculadas (output de calculate_metrics)
        output_dir (str|Path): Diretório onde salvar os gráficos PNG
        
    Output:
        Arquivos PNG salvos como: {app}_speedup_efficiency.png
        Resolução: 300 DPI (qualidade publicação)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for app in metrics:
        # Focar no problema "extreme" (maior carga computacional)
        for size in ['extreme']:
            if size not in metrics[app]:
                continue
                
            # Extrair dados para plotagem
            threads_list = sorted(metrics[app][size].keys())
            speedups = [metrics[app][size][t]['speedup'] for t in threads_list]
            efficiencies = [metrics[app][size][t]['efficiency'] * 100 for t in threads_list]
            
            # Criar figura com 2 subplots lado a lado
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # ========== PAINEL ESQUERDO: SPEEDUP ==========
            # Speedup observado
            ax1.plot(threads_list, speedups, 'o-', linewidth=2, markersize=8, 
                    label='Observed', color='blue')
            
            # Speedup ideal (linear: S(p) = p)
            ax1.plot(threads_list, threads_list, '--', linewidth=2, 
                    label='Ideal (linear)', alpha=0.5, color='gray')
            
            ax1.set_xlabel('Number of Threads', fontsize=14)
            ax1.set_ylabel('Speedup', fontsize=14)
            ax1.set_title(f'{app.replace("_", " ").title()} - Speedup', 
                         fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=12)
            ax1.tick_params(labelsize=12)
            
            # ========== PAINEL DIREITO: EFICIÊNCIA ==========
            # Eficiência observada
            ax2.plot(threads_list, efficiencies, 's-', linewidth=2, markersize=8, 
                    color='red', label='Observed')
            
            # Eficiência ideal (100%)
            ax2.axhline(y=100, linestyle='--', linewidth=2, color='gray', 
                       alpha=0.5, label='Ideal (100%)')
            
            ax2.set_xlabel('Number of Threads', fontsize=14)
            ax2.set_ylabel('Efficiency (%)', fontsize=14)
            ax2.set_title(f'{app.replace("_", " ").title()} - Efficiency', 
                         fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=12)
            ax2.tick_params(labelsize=12)
            
            # Salvar figura em alta resolução
            plt.tight_layout()
            plt.savefig(output_dir / f'{app}_speedup_efficiency.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Generated: {app}_speedup_efficiency.png")

def generate_overhead_graphs(metrics, output_dir):
    """
    Gera gráficos de overhead e fração serial para cada aplicação.
    
    Cria figuras com dois painéis:
      - Esquerda: Overhead relativo (φ) - tempo gasto com sincronização
      - Direita: Fração serial (ε) - estimativa de código não paralelizável
    
    Args:
        metrics (dict): Estrutura de métricas calculadas
        output_dir (str|Path): Diretório de saída
        
    Output:
        Arquivos PNG: {app}_overhead_serial.png (300 DPI)
        
    Interpretação:
      - Overhead alto (φ >> 1): muita sincronização/coordenação
      - Fração serial alta (ε → 1): código majoritariamente serial
    """
    output_dir = Path(output_dir)
    
    for app in metrics:
        for size in ['extreme']:
            if size not in metrics[app]:
                continue
                
            # Extrair dados
            threads_list = sorted(metrics[app][size].keys())
            overheads = [metrics[app][size][t]['overhead'] for t in threads_list]
            serial_fractions = [metrics[app][size][t]['serial_fraction'] for t in threads_list]
            
            # Criar figura com 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # ========== PAINEL ESQUERDO: OVERHEAD ==========
            # Overhead relativo: φ = (p×Tp - T1) / T1
            ax1.plot(threads_list, overheads, 'o-', linewidth=2, markersize=8, 
                    color='orange', label='Overhead')
            ax1.set_xlabel('Number of Threads', fontsize=14)
            ax1.set_ylabel('Relative Overhead (φ)', fontsize=14)
            ax1.set_title(f'{app.replace("_", " ").title()} - Overhead', 
                         fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=12)
            
            # ========== PAINEL DIREITO: FRAÇÃO SERIAL ==========
            # Fração serial (Karp-Flatt): ε = (1/S - 1/p) / (1 - 1/p)
            ax2.plot(threads_list, serial_fractions, 's-', linewidth=2, markersize=8, 
                    color='purple', label='Serial Fraction')
            ax2.set_xlabel('Number of Threads', fontsize=14)
            ax2.set_ylabel('Serial Fraction (ε)', fontsize=14)
            ax2.set_title(f'{app.replace("_", " ").title()} - Serial Fraction', 
                         fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=12)
            
            # Salvar figura
            plt.tight_layout()
            plt.savefig(output_dir / f'{app}_overhead_serial.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Generated: {app}_overhead_serial.png")

def generate_polynomial_fit_graphs(metrics, output_dir):
    """
    Gera gráficos de ajuste polinomial para análise de escalabilidade.
    
    Para cada aplicação, ajusta um polinômio de grau 2 aos dados de speedup:
        S(p) = a₀ + a₁p + a₂p²
    
    Onde:
      - a₁: taxa inicial de ganho por thread
      - a₂: taxa de degradação (saturação)
      - R²: qualidade do ajuste (0-1, quanto maior melhor)
    
    Args:
        metrics (dict): Estrutura de métricas calculadas
        output_dir (str|Path): Diretório de saída
        
    Output:
        Arquivos PNG: {app}_polynomial_comparison.png
        Gráfico mostra: dados observados, curva ajustada, speedup ideal, R²
    """
    output_dir = Path(output_dir)
    
    for app in metrics:
        for size in ['extreme']:
            if size not in metrics[app]:
                continue
                
            threads_list = sorted(metrics[app][size].keys())
            speedups = [metrics[app][size][t]['speedup'] for t in threads_list]
            
            # Polynomial fit (degree 2)
            coeffs = np.polyfit(threads_list, speedups, 2)
            poly = np.poly1d(coeffs)
            
            # R² calculation
            y_mean = np.mean(speedups)
            ss_tot = sum((y - y_mean)**2 for y in speedups)
            ss_res = sum((speedups[i] - poly(threads_list[i]))**2 for i in range(len(speedups)))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Generate smooth curve
            threads_smooth = np.linspace(min(threads_list), max(threads_list), 100)
            speedup_fit = poly(threads_smooth)
            
            plt.figure(figsize=(12, 8))
            plt.plot(threads_list, speedups, 'o', markersize=10, label='Observed Data', color='blue')
            plt.plot(threads_smooth, speedup_fit, '-', linewidth=2, label=f'Polynomial Fit (R²={r_squared:.3f})', color='red')
            plt.plot(threads_list, threads_list, '--', linewidth=2, label='Ideal (linear)', alpha=0.5, color='gray')
            
            plt.xlabel('Number of Threads', fontsize=14)
            plt.ylabel('Speedup', fontsize=14)
            plt.title(f'{app.replace("_", " ").title()} - Polynomial Fit Comparison\nS(p) = {coeffs[0]:.4f}p² + {coeffs[1]:.4f}p + {coeffs[2]:.4f}', 
                     fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tick_params(labelsize=12)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{app}_polynomial_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Generated: {app}_polynomial_comparison.png")

def generate_comparison_graph(metrics, output_dir, size='large'):
    """Generate comparison graph for all applications"""
    output_dir = Path(output_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    for app in sorted(metrics.keys()):
        if size not in metrics[app]:
            continue
            
        threads_list = sorted(metrics[app][size].keys())
        speedups = [metrics[app][size][t]['speedup'] for t in threads_list]
        efficiencies = [metrics[app][size][t]['efficiency'] * 100 for t in threads_list]
        
        label = app.replace('c_', '').replace('_', ' ').title()
        ax1.plot(threads_list, speedups, 'o-', linewidth=2, markersize=6, label=label)
        ax2.plot(threads_list, efficiencies, 's-', linewidth=2, markersize=6, label=label)
    
    # Speedup comparison
    ax1.plot([1, 24], [1, 24], '--', linewidth=2, color='black', alpha=0.3, label='Ideal')
    ax1.set_xlabel('Number of Threads', fontsize=14)
    ax1.set_ylabel('Speedup', fontsize=14)
    ax1.set_title(f'Speedup Comparison - {size.capitalize()} Problem', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, ncol=2)
    ax1.tick_params(labelsize=12)
    
    # Efficiency comparison
    ax2.axhline(y=100, linestyle='--', linewidth=2, color='black', alpha=0.3, label='Ideal')
    ax2.set_xlabel('Number of Threads', fontsize=14)
    ax2.set_ylabel('Efficiency (%)', fontsize=14)
    ax2.set_title(f'Efficiency Comparison - {size.capitalize()} Problem', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, ncol=2)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_{size}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated: comparison_{size}.png")

def generate_overhead_table(metrics, output_file, size='extreme', threads=24):
    """Generate LaTeX overhead comparison table"""
    
    data = []
    for app in sorted(metrics.keys()):
        if size in metrics[app] and threads in metrics[app][size]:
            m = metrics[app][size][threads]
            data.append({
                'app': app,
                'overhead': m['overhead'],
                'speedup': m['speedup'],
                'efficiency': m['efficiency'] * 100,
                'serial_fraction': m['serial_fraction']
            })
    
    # Sort by overhead
    data.sort(key=lambda x: x['overhead'])
    
    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Overhead Analysis - {size.capitalize()} Problem with {threads} Threads}}")
    latex.append("\\label{tab:overhead-extreme-24}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Application} & \\textbf{Overhead ($\\phi$)} & \\textbf{Speedup} & \\textbf{Efficiency (\\%)} & \\textbf{Serial Fraction ($\\epsilon$)} \\\\")
    latex.append("\\midrule")
    
    for entry in data:
        app_name = entry['app'].replace('_', '\\_')
        latex.append(f"{app_name} & {entry['overhead']:.2f} & {entry['speedup']:.2f}× & {entry['efficiency']:.1f}\\% & {entry['serial_fraction']:.4f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ Generated: {output_path.name}")

def generate_results_table(metrics, output_file, size='large'):
    """Generate LaTeX results table with best configurations"""
    
    data = []
    for app in sorted(metrics.keys()):
        if size not in metrics[app]:
            continue
        
        # Find best configuration
        best_threads = max(metrics[app][size].keys(), key=lambda t: metrics[app][size][t]['speedup'])
        m = metrics[app][size][best_threads]
        
        data.append({
            'app': app,
            'efficiency': m['efficiency'] * 100,
            'speedup': m['speedup'],
            'threads': best_threads,
            'overhead': m['overhead'],
            'serial_fraction': m['serial_fraction']
        })
    
    # Sort by efficiency descending
    data.sort(key=lambda x: x['efficiency'], reverse=True)
    
    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Best Results Summary - {size.capitalize()} Problem}}")
    latex.append("\\label{tab:resultados-gerais}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Application} & \\textbf{Max Efficiency} & \\textbf{Max Speedup} & \\textbf{Threads} & \\textbf{$\\phi$} & \\textbf{$\\epsilon$} & \\textbf{Rating} \\\\")
    latex.append("\\midrule")
    
    for entry in data:
        app_name = entry['app'].replace('_', '\\_')
        
        # Determine rating
        if entry['efficiency'] > 60:
            rating = "Excellent"
        elif entry['efficiency'] > 40:
            rating = "Good"
        elif entry['efficiency'] > 20:
            rating = "Moderate"
        else:
            rating = "Limited"
        
        latex.append(f"{app_name} & {entry['efficiency']:.1f}\\% & {entry['speedup']:.2f}× & {entry['threads']} & {entry['overhead']:.2f} & {entry['serial_fraction']:.4f} & {rating} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ Generated: {output_path.name}")

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Ponto de entrada principal do script.
    
    Fluxo de execução:
      1. Parse de argumentos de linha de comando
      2. Carregamento de dados do JSON
      3. Cálculo de métricas de desempenho
      4. Geração de gráficos (se solicitado)
      5. Geração de tabelas LaTeX (se solicitado)
      
    Uso típico:
        python3 analyze_and_visualize.py                    # Gerar tudo
        python3 analyze_and_visualize.py --graphs-only      # Só gráficos
        python3 analyze_and_visualize.py --tables-only      # Só tabelas
        python3 analyze_and_visualize.py --apps c_mandel    # App específica
    """
    # ========== CONFIGURAÇÃO DE ARGUMENTOS ==========
    parser = argparse.ArgumentParser(
        description='Analyze and visualize benchmark results',
        epilog='Example: python3 %(prog)s --graphs-only --apps c_mandel c_md'
    )
    
    parser.add_argument('--json', 
                       default='benchmark_results/benchmark_results_20251129_154041.json',
                       help='Path to benchmark JSON file')
    parser.add_argument('--graphs-only', action='store_true', 
                       help='Generate only graphs (skip tables)')
    parser.add_argument('--tables-only', action='store_true', 
                       help='Generate only tables (skip graphs)')
    parser.add_argument('--apps', nargs='+', 
                       help='Specific applications to process (default: all)')
    
    args = parser.parse_args()
    
    # ========== BANNER INICIAL ==========
    print("=" * 60)
    print("CONSOLIDATED BENCHMARK ANALYSIS AND VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print(f"\n📊 Loading data from: {args.json}")
    data = load_benchmark_data(args.json)
    print(f"✓ Loaded {len(data)} benchmark results")
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    metrics = calculate_metrics(data)
    
    # Filter applications if specified
    if args.apps:
        metrics = {app: metrics[app] for app in args.apps if app in metrics}
    
    print(f"✓ Processed {len(metrics)} applications")
    
    # Generate graphs
    if not args.tables_only:
        print("\n🎨 Generating graphs...")
        generate_speedup_graphs(metrics, OUTPUT_DIR)
        generate_overhead_graphs(metrics, OUTPUT_DIR)
        generate_polynomial_fit_graphs(metrics, OUTPUT_DIR)
        generate_comparison_graph(metrics, OUTPUT_DIR, 'large')
        print("✓ All graphs generated")
    
    # Generate tables
    if not args.graphs_only:
        print("\n📋 Generating LaTeX tables...")
        generate_overhead_table(metrics, TABLES_DIR / "tabela_overhead_extreme_24.tex")
        generate_results_table(metrics, TABLES_DIR / "tabela_resultados_summary.tex")
        print("✓ All tables generated")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGraphs saved to: {OUTPUT_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")

if __name__ == '__main__':
    main()
