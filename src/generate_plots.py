#!/usr/bin/env python3
"""
Script para geração de gráficos de análise de desempenho dos benchmarks OpenMP
Gera gráficos de speedup por aplicação e comparações entre aplicações
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import sys

class BenchmarkPlotter:
    """Gerador de gráficos para análise de benchmarks OpenMP"""
    
    def __init__(self, results_file, output_dir="plots"):
        """
        Inicializa o plotador com arquivo de resultados
        
        Args:
            results_file: Caminho para o arquivo JSON com resultados
            output_dir: Diretório de saída para os gráficos
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Carregar resultados
        print(f"📁 Carregando resultados de: {self.results_file}")
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # Converter para DataFrame
        self.df = self._prepare_dataframe()
        
        # Configurar estilo dos gráficos
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _prepare_dataframe(self):
        """Prepara DataFrame a partir dos resultados JSON"""
        data = []
        for result in self.results:
            if result.get('success', False):
                data.append({
                    'benchmark': result['benchmark'],
                    'threads': result['threads'],
                    'problem_size': result['problem_size'],
                    'iteration': result['iteration'],
                    'wall_time': result['wall_time']
                })
        
        if not data:
            print("❌ Nenhum resultado válido encontrado!")
            sys.exit(1)
        
        df = pd.DataFrame(data)
        
        # Calcular tempo médio por configuração
        df_avg = df.groupby(['benchmark', 'threads', 'problem_size'])['wall_time'].mean().reset_index()
        df_avg.columns = ['benchmark', 'threads', 'problem_size', 'avg_time']
        
        # Calcular speedup (tempo com 1 thread / tempo com N threads)
        speedups = []
        for benchmark in df_avg['benchmark'].unique():
            for size in df_avg['problem_size'].unique():
                # Filtrar dados deste benchmark e tamanho
                subset = df_avg[(df_avg['benchmark'] == benchmark) & 
                               (df_avg['problem_size'] == size)]
                
                if len(subset) == 0:
                    continue
                
                # Tempo base (1 thread)
                base_time = subset[subset['threads'] == 1]['avg_time'].values
                
                if len(base_time) == 0:
                    continue
                
                base_time = base_time[0]
                
                # Calcular speedup para cada configuração de threads
                for _, row in subset.iterrows():
                    speedup = base_time / row['avg_time']
                    efficiency = (speedup / row['threads']) * 100
                    
                    speedups.append({
                        'benchmark': row['benchmark'],
                        'threads': row['threads'],
                        'problem_size': row['problem_size'],
                        'avg_time': row['avg_time'],
                        'speedup': speedup,
                        'efficiency': efficiency
                    })
        
        return pd.DataFrame(speedups)
    
    def plot_speedup_per_application(self):
        """
        Gera gráficos individuais de speedup para cada aplicação
        Cada gráfico mostra todas as dimensões (small, medium, large, etc.)
        """
        print("\n📊 Gerando gráficos individuais por aplicação...")
        
        benchmarks = self.df['benchmark'].unique()
        problem_sizes = sorted(self.df['problem_size'].unique())
        
        # Cores e estilos para cada tamanho
        size_colors = {
            'small': '#2ecc71',
            'medium': '#3498db', 
            'large': '#e74c3c',
            'huge': '#9b59b6',
            'extreme': '#e67e22'
        }
        
        size_markers = {
            'small': 'o',
            'medium': 's',
            'large': '^',
            'huge': 'D',
            'extreme': 'v'
        }
        
        for benchmark in benchmarks:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{benchmark} - Análise de Escalabilidade', fontsize=14, fontweight='bold')
            
            # Filtrar dados deste benchmark
            bench_data = self.df[self.df['benchmark'] == benchmark]
            
            # Plot 1: Speedup vs Threads
            for size in problem_sizes:
                size_data = bench_data[bench_data['problem_size'] == size].sort_values('threads')
                
                if len(size_data) == 0:
                    continue
                
                ax1.plot(size_data['threads'], size_data['speedup'], 
                        marker=size_markers.get(size, 'o'),
                        color=size_colors.get(size, 'gray'),
                        linewidth=2, markersize=8,
                        label=f'{size}')
            
            # Linha de speedup ideal
            max_threads = bench_data['threads'].max()
            ideal_threads = sorted(bench_data['threads'].unique())
            ax1.plot(ideal_threads, ideal_threads, 'k--', 
                    linewidth=1.5, alpha=0.7, label='Speedup Ideal')
            
            ax1.set_xlabel('Número de Threads', fontsize=11)
            ax1.set_ylabel('Speedup', fontsize=11)
            ax1.set_title('Speedup vs Threads', fontsize=12)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)
            ax1.set_yscale('log', base=2)
            
            # Plot 2: Eficiência Paralela
            for size in problem_sizes:
                size_data = bench_data[bench_data['problem_size'] == size].sort_values('threads')
                
                if len(size_data) == 0:
                    continue
                
                ax2.plot(size_data['threads'], size_data['efficiency'],
                        marker=size_markers.get(size, 'o'),
                        color=size_colors.get(size, 'gray'),
                        linewidth=2, markersize=8,
                        label=f'{size}')
            
            # Linha de 100% eficiência
            ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='100% Eficiência')
            
            ax2.set_xlabel('Número de Threads', fontsize=11)
            ax2.set_ylabel('Eficiência Paralela (%)', fontsize=11)
            ax2.set_title('Eficiência vs Threads', fontsize=12)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)
            ax2.set_ylim(0, 110)
            
            plt.tight_layout()
            
            # Salvar gráfico
            output_file = self.output_dir / f'{benchmark}_speedup.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ {benchmark}: {output_file}")
            plt.close()
    
    def plot_standard_applications_comparison(self):
        """
        Gera gráficos comparando todas as versões 'standard' das aplicações
        Um gráfico por tamanho de problema (small, medium, large, etc.)
        """
        print("\n📊 Gerando gráficos comparativos das versões standard...")
        
        # Filtrar apenas versões standard (sem _fine ou _coarse no nome)
        standard_benchmarks = [b for b in self.df['benchmark'].unique() 
                              if not ('_fine' in b or '_coarse' in b)]
        
        problem_sizes = sorted(self.df['problem_size'].unique())
        
        # Cores diferentes para cada aplicação
        colors = sns.color_palette("husl", len(standard_benchmarks))
        color_map = dict(zip(standard_benchmarks, colors))
        
        for size in problem_sizes:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Comparação de Aplicações Standard - Problema Size: {size.upper()}', 
                        fontsize=14, fontweight='bold')
            
            # Plot 1: Speedup
            for i, benchmark in enumerate(standard_benchmarks):
                bench_data = self.df[(self.df['benchmark'] == benchmark) & 
                                    (self.df['problem_size'] == size)].sort_values('threads')
                
                if len(bench_data) == 0:
                    continue
                
                ax1.plot(bench_data['threads'], bench_data['speedup'],
                        marker='o', color=color_map[benchmark],
                        linewidth=2, markersize=7,
                        label=benchmark)
            
            # Speedup ideal
            max_threads = self.df['threads'].max()
            ideal_threads = sorted(self.df['threads'].unique())
            ax1.plot(ideal_threads, ideal_threads, 'k--',
                    linewidth=1.5, alpha=0.7, label='Speedup Ideal')
            
            ax1.set_xlabel('Número de Threads', fontsize=11)
            ax1.set_ylabel('Speedup', fontsize=11)
            ax1.set_title('Speedup vs Threads', fontsize=12)
            ax1.legend(loc='upper left', fontsize=8, ncol=2)
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)
            ax1.set_yscale('log', base=2)
            
            # Plot 2: Eficiência
            for i, benchmark in enumerate(standard_benchmarks):
                bench_data = self.df[(self.df['benchmark'] == benchmark) & 
                                    (self.df['problem_size'] == size)].sort_values('threads')
                
                if len(bench_data) == 0:
                    continue
                
                ax2.plot(bench_data['threads'], bench_data['efficiency'],
                        marker='o', color=color_map[benchmark],
                        linewidth=2, markersize=7,
                        label=benchmark)
            
            ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='100% Eficiência')
            
            ax2.set_xlabel('Número de Threads', fontsize=11)
            ax2.set_ylabel('Eficiência Paralela (%)', fontsize=11)
            ax2.set_title('Eficiência vs Threads', fontsize=12)
            ax2.legend(loc='upper right', fontsize=8, ncol=2)
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)
            ax2.set_ylim(0, 110)
            
            plt.tight_layout()
            
            # Salvar gráfico
            output_file = self.output_dir / f'comparison_standard_{size}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ {size}: {output_file}")
            plt.close()
    
    def plot_granularity_comparison(self):
        """
        Gera gráficos comparando as três versões de cada aplicação
        (standard, fine-grained, coarse-grained)
        """
        print("\n📊 Gerando gráficos de comparação de granularidade...")
        
        # Identificar aplicações base (sem sufixo)
        base_apps = set()
        for bench in self.df['benchmark'].unique():
            if '_fine' in bench:
                base_apps.add(bench.replace('_fine', ''))
            elif '_coarse' in bench:
                base_apps.add(bench.replace('_coarse', ''))
            else:
                # Verificar se existem variantes fine/coarse
                bench_base = bench
                if f"{bench}_fine" in self.df['benchmark'].values or f"{bench}_coarse" in self.df['benchmark'].values:
                    base_apps.add(bench_base)
        
        problem_sizes = sorted(self.df['problem_size'].unique())
        
        for base_app in sorted(base_apps):
            # Identificar as três versões
            variants = {
                'standard': base_app,
                'fine': f"{base_app}_fine",
                'coarse': f"{base_app}_coarse"
            }
            
            # Verificar se todas as versões existem
            available_variants = {k: v for k, v in variants.items() 
                                 if v in self.df['benchmark'].values}
            
            if len(available_variants) < 2:
                continue
            
            # Criar subplots por tamanho de problema
            n_sizes = len(problem_sizes)
            fig, axes = plt.subplots(1, n_sizes, figsize=(6*n_sizes, 5))
            if n_sizes == 1:
                axes = [axes]
            
            fig.suptitle(f'{base_app} - Comparação de Granularidade', 
                        fontsize=14, fontweight='bold')
            
            variant_colors = {
                'standard': '#3498db',
                'fine': '#2ecc71',
                'coarse': '#e74c3c'
            }
            
            variant_markers = {
                'standard': 'o',
                'fine': '^',
                'coarse': 's'
            }
            
            for idx, size in enumerate(problem_sizes):
                ax = axes[idx]
                
                for variant_name, variant_bench in available_variants.items():
                    variant_data = self.df[(self.df['benchmark'] == variant_bench) & 
                                          (self.df['problem_size'] == size)].sort_values('threads')
                    
                    if len(variant_data) == 0:
                        continue
                    
                    ax.plot(variant_data['threads'], variant_data['speedup'],
                           marker=variant_markers[variant_name],
                           color=variant_colors[variant_name],
                           linewidth=2, markersize=8,
                           label=variant_name)
                
                # Speedup ideal
                ideal_threads = sorted(self.df['threads'].unique())
                ax.plot(ideal_threads, ideal_threads, 'k--',
                       linewidth=1.5, alpha=0.7, label='ideal')
                
                ax.set_xlabel('Threads', fontsize=10)
                ax.set_ylabel('Speedup', fontsize=10)
                ax.set_title(f'{size}', fontsize=11)
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log', base=2)
                ax.set_yscale('log', base=2)
            
            plt.tight_layout()
            
            # Salvar gráfico
            output_file = self.output_dir / f'{base_app}_granularity_comparison.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ {base_app}: {output_file}")
            plt.close()
    
    def plot_execution_time_heatmap(self):
        """
        Gera heatmaps de tempo de execução para cada tamanho de problema
        """
        print("\n📊 Gerando heatmaps de tempo de execução...")
        
        # Filtrar apenas versões standard
        standard_benchmarks = [b for b in self.df['benchmark'].unique() 
                              if not ('_fine' in b or '_coarse' in b)]
        
        problem_sizes = sorted(self.df['problem_size'].unique())
        
        for size in problem_sizes:
            # Preparar dados para o heatmap
            pivot_data = self.df[(self.df['benchmark'].isin(standard_benchmarks)) & 
                                (self.df['problem_size'] == size)].pivot_table(
                                    index='benchmark',
                                    columns='threads',
                                    values='avg_time',
                                    aggfunc='mean'
                                )
            
            if pivot_data.empty:
                continue
            
            fig, ax = plt.subplots(figsize=(12, len(standard_benchmarks) * 0.6))
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': 'Tempo (segundos)'},
                       linewidths=0.5, ax=ax)
            
            ax.set_title(f'Tempo de Execução - {size.upper()}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Número de Threads', fontsize=11)
            ax.set_ylabel('Aplicação', fontsize=11)
            
            plt.tight_layout()
            
            # Salvar gráfico
            output_file = self.output_dir / f'heatmap_time_{size}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ {size}: {output_file}")
            plt.close()
    
    def generate_all_plots(self):
        """Gera todos os gráficos disponíveis"""
        print("\n" + "="*60)
        print("🎨 GERADOR DE GRÁFICOS - BENCHMARK OPENMP")
        print("="*60)
        print(f"\n📊 Total de resultados carregados: {len(self.df)}")
        print(f"📋 Aplicações encontradas: {len(self.df['benchmark'].unique())}")
        print(f"🔢 Contagens de threads: {sorted(self.df['threads'].unique())}")
        print(f"📏 Tamanhos de problema: {sorted(self.df['problem_size'].unique())}")
        
        self.plot_speedup_per_application()
        self.plot_standard_applications_comparison()
        self.plot_granularity_comparison()
        self.plot_execution_time_heatmap()
        
        print("\n" + "="*60)
        print(f"✅ Todos os gráficos foram salvos em: {self.output_dir.absolute()}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Gerador de gráficos para análise de benchmarks OpenMP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Gerar todos os gráficos
  python generate_plots.py benchmark_results/benchmark_results_20241125_120000.json
  
  # Especificar diretório de saída
  python generate_plots.py results.json --output plots_final
  
  # Gerar apenas gráficos específicos
  python generate_plots.py results.json --speedup-only
  python generate_plots.py results.json --comparison-only
        """
    )
    
    parser.add_argument('results_file', 
                       help='Arquivo JSON com resultados dos benchmarks')
    parser.add_argument('--output', '-o', default='plots',
                       help='Diretório de saída para os gráficos (default: plots)')
    parser.add_argument('--speedup-only', action='store_true',
                       help='Gerar apenas gráficos individuais de speedup')
    parser.add_argument('--comparison-only', action='store_true',
                       help='Gerar apenas gráficos comparativos')
    parser.add_argument('--granularity-only', action='store_true',
                       help='Gerar apenas gráficos de comparação de granularidade')
    
    args = parser.parse_args()
    
    # Verificar se arquivo existe
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"❌ Erro: Arquivo não encontrado: {results_path}")
        sys.exit(1)
    
    # Criar plotador
    plotter = BenchmarkPlotter(args.results_file, args.output)
    
    # Gerar gráficos conforme opções
    if args.speedup_only:
        plotter.plot_speedup_per_application()
    elif args.comparison_only:
        plotter.plot_standard_applications_comparison()
    elif args.granularity_only:
        plotter.plot_granularity_comparison()
    else:
        plotter.generate_all_plots()

if __name__ == "__main__":
    main()
