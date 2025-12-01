#!/usr/bin/env python3
"""
Utility Functions for Benchmark Analysis
=========================================

Biblioteca de funções auxiliares compartilhadas entre scripts de análise.
Fornece utilitários para:
  - Carregamento e salvamento de dados
  - Cálculos de métricas de desempenho paralelo
  - Formatação de nomes e saída
  - Análise estatística (ajuste polinomial, R²)
  - Geração de estruturas LaTeX

Autor: Sistema consolidado de benchmarking OpenMP
Data: 2025-12-01
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================================
# FUNÇÕES DE I/O
# ============================================================================

def load_json(filepath):
    """
    Carrega dados de um arquivo JSON.
    
    Args:
        filepath (str|Path): Caminho para o arquivo JSON
        
    Returns:
        dict|list: Dados desserializados do JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """
    Salva dados em um arquivo JSON formatado.
    
    Args:
        data (dict|list): Dados a serem salvos
        filepath (str|Path): Caminho do arquivo de saída
        
    Output:
        Arquivo JSON com indentação de 2 espaços
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# ============================================================================
# FORMATAÇÃO E EXIBIÇÃO
# ============================================================================

def format_app_name(app_name, latex=False):
    """
    Formata nome de aplicação para exibição legível.
    
    Args:
        app_name (str): Nome técnico (ex: 'c_mandel', 'c_pi_fine')
        latex (bool): Se True, escapa underscores para LaTeX
        
    Returns:
        str: Nome formatado
        
    Examples:
        format_app_name('c_mandel') → 'Mandel'
        format_app_name('c_pi_fine') → 'Pi Fine'
        format_app_name('c_mandel', latex=True) → 'c\\_mandel'
    """
    if latex:
        return app_name.replace('_', '\\_')
    else:
        return app_name.replace('c_', '').replace('_', ' ').title()

# ============================================================================
# CÁLCULOS DE MÉTRICAS DE DESEMPENHO
# ============================================================================

def calculate_speedup(T1, Tp):
    """
    Calcula speedup: quanto mais rápido ficou com paralelização.
    
    Fórmula: S(p) = T₁ / Tₚ
    
    Args:
        T1 (float): Tempo de execução serial (1 thread)
        Tp (float): Tempo de execução paralelo (p threads)
        
    Returns:
        float: Speedup (valores > 1 indicam aceleração)
        
    Interpretação:
        S = 1.0 → Sem ganho
        S = p → Speedup ideal (linear)
        S < p → Speedup sublinear (usual)
        S > p → Speedup superlinear (raro, efeito de cache)
    """
    return T1 / Tp if Tp > 0 else 0

def calculate_efficiency(speedup, p):
    """
    Calcula eficiência: percentual de uso ideal dos processadores.
    
    Fórmula: E(p) = S(p) / p
    
    Args:
        speedup (float): Speedup observado
        p (int): Número de threads/processadores
        
    Returns:
        float: Eficiência (0.0 a 1.0, onde 1.0 = 100%)
        
    Interpretação:
        E = 1.0 → Uso ideal (todos processadores 100% úteis)
        E = 0.5 → 50% de eficiência (muito overhead)
        E → 0 → Paralelização ineficaz
    """
    return speedup / p if p > 0 else 0

def calculate_overhead(T1, Tp, p):
    """
    Calcula overhead: tempo extra gasto com sincronização/coordenação.
    
    Fórmula: φ = (p × Tₚ - T₁) / T₁
    
    Args:
        T1 (float): Tempo serial
        Tp (float): Tempo paralelo
        p (int): Número de threads
        
    Returns:
        float: Overhead relativo
        
    Interpretação:
        φ = 0 → Sem overhead (ideal impossível)
        φ < 1 → Overhead baixo (bom)
        φ = 1 → Overhead igual ao trabalho útil
        φ >> 1 → Overhead dominante (ruim)
        
    Exemplo:
        T1 = 10s, Tp = 2s, p = 8
        φ = (8×2 - 10)/10 = 0.6 → 60% de overhead
    """
    return (p * Tp - T1) / T1 if T1 > 0 else 0

def calculate_serial_fraction(speedup, p):
    """
    Calcula fração serial usando métrica de Karp-Flatt.
    
    Fórmula: ε = (1/S - 1/p) / (1 - 1/p)
    
    Estima a porção de código que não pode ser paralelizada.
    Baseado na Lei de Amdahl: S(p) ≤ 1 / (ε + (1-ε)/p)
    
    Args:
        speedup (float): Speedup observado
        p (int): Número de threads
        
    Returns:
        float: Fração serial (0.0 a 1.0)
        
    Interpretação:
        ε = 0 → Código 100% paralelizável (ideal)
        ε = 0.1 → 10% serial, 90% paralelo (bom)
        ε = 0.5 → 50% serial (limitado)
        ε → 1 → Código essencialmente serial
        ε > 1 → Overhead excede tempo serial (muito ruim)
    """
    if p <= 1:
        return 0
    
    denominator = 1 - 1/p
    if denominator == 0:
        return 0
    
    return (1/speedup - 1/p) / denominator if speedup > 0 else 1

# ============================================================================
# ANÁLISE ESTATÍSTICA
# ============================================================================

def polynomial_fit(x_data, y_data, degree=2):
    """
    Ajusta polinômio aos dados e calcula qualidade do ajuste (R²).
    
    Usa regressão por mínimos quadrados para encontrar coeficientes
    do polinômio que melhor se ajusta aos dados observados.
    
    Args:
        x_data (list): Valores de x (ex: número de threads)
        y_data (list): Valores de y (ex: speedup observado)
        degree (int): Grau do polinômio (default: 2 = quadrático)
        
    Returns:
        tuple: (coeffs, r_squared, poly)
            - coeffs: Coeficientes [a_n, ..., a_1, a_0]
            - r_squared: Qualidade do ajuste (0-1)
            - poly: Função polinomial ajustada
            
    Exemplo:
        threads = [1, 2, 4, 8, 16]
        speedups = [1.0, 1.9, 3.6, 6.8, 11.2]
        coeffs, r2, poly = polynomial_fit(threads, speedups, degree=2)
        # coeffs = [a2, a1, a0] onde S(p) = a2*p² + a1*p + a0
        # r2 = 0.995 (ajuste excelente)
        
    Interpretação de R²:
        R² > 0.95 → Ajuste excelente
        0.70 < R² < 0.95 → Ajuste moderado
        R² < 0.70 → Ajuste pobre
    """
    # Ajustar polinômio aos dados
    coeffs = np.polyfit(x_data, y_data, degree)
    poly = np.poly1d(coeffs)
    
    # Calcular R² (coeficiente de determinação)
    # R² = 1 - (SS_res / SS_tot)
    # SS_tot = soma dos quadrados total
    # SS_res = soma dos quadrados dos resíduos
    y_mean = np.mean(y_data)
    ss_tot = sum((y - y_mean)**2 for y in y_data)
    ss_res = sum((y_data[i] - poly(x_data[i]))**2 for i in range(len(y_data)))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return coeffs, r_squared, poly

# ============================================================================
# ANÁLISE E SELEÇÃO DE DADOS
# ============================================================================

def get_best_configuration(metrics, app, size):
    """
    Encontra a melhor configuração de threads para uma aplicação.
    
    "Melhor" é definido como a configuração com maior speedup.
    
    Args:
        metrics (dict): Estrutura de métricas calculadas
        app (str): Nome da aplicação
        size (str): Tamanho do problema
        
    Returns:
        tuple: (best_threads, best_metrics) ou None se não encontrado
            - best_threads: Número ótimo de threads
            - best_metrics: Dict com todas as métricas dessa config
            
    Exemplo:
        threads, metrics = get_best_configuration(data, 'c_mandel', 'large')
        # threads = 24
        # metrics = {'speedup': 15.99, 'efficiency': 0.666, ...}
    """
    if app not in metrics or size not in metrics[app]:
        return None
    
    # Encontrar configuração com maior speedup
    best_threads = max(metrics[app][size].keys(), 
                      key=lambda t: metrics[app][size][t]['speedup'])
    return best_threads, metrics[app][size][best_threads]

def filter_main_apps(metrics):
    """
    Filtra apenas aplicações principais (sem variações de granularidade).
    
    Remove variantes _fine, _coarse, mantendo apenas versões standard.
    
    Args:
        metrics (dict): Estrutura completa de métricas
        
    Returns:
        dict: Métricas filtradas apenas para apps principais
        
    Apps principais incluídas:
        c_pi, c_mandel, c_qsort, c_fft, c_fft6, c_md, c_lu,
        c_jacobi01, c_jacobi02, c_jacobi03
    """
    main_apps = ['c_pi', 'c_mandel', 'c_qsort', 'c_fft', 'c_fft6', 'c_md', 'c_lu',
                 'c_jacobi01', 'c_jacobi02', 'c_jacobi03']
    return {app: metrics[app] for app in main_apps if app in metrics}

def get_size_progression(metrics, app):
    """
    Obtém progressão de métricas através dos tamanhos de problema.
    
    Útil para análise de escalabilidade conforme Lei de Gustafson.
    
    Args:
        metrics (dict): Estrutura de métricas
        app (str): Nome da aplicação
        
    Returns:
        dict: {size: {threads: {metrics}}} ou None se app não existe
        
    Exemplo:
        prog = get_size_progression(data, 'c_mandel')
        # prog = {
        #   'small': {1: {...}, 2: {...}, ...},
        #   'medium': {1: {...}, 2: {...}, ...},
        #   ...
        # }
    """
    if app not in metrics:
        return None
    
    sizes = ['small', 'medium', 'large', 'huge', 'extreme']
    progression = {}
    
    for size in sizes:
        if size in metrics[app]:
            progression[size] = metrics[app][size]
    
    return progression

# ============================================================================
# FORMATAÇÃO LaTeX
# ============================================================================

def create_latex_table_header(caption, label, columns):
    """
    Cria cabeçalho de tabela LaTeX com formatação booktabs.
    
    Args:
        caption (str): Legenda da tabela
        label (str): Rótulo para referência cruzada (\ref{label})
        columns (list): Lista de nomes das colunas
        
    Returns:
        list: Linhas de código LaTeX do cabeçalho
        
    Formato gerado:
        - Primeira coluna: alinhada à esquerda ('l')
        - Demais colunas: centralizadas ('c')
        - Usa pacote booktabs (\toprule, \midrule, \bottomrule)
        - Usa pacote float ([H] = posição exata)
        - Fonte reduzida (\small) para tabelas grandes
        
    Exemplo:
        header = create_latex_table_header(
            "Speedup por Threads", 
            "tab:speedup", 
            ["App", "1T", "2T", "4T", "8T"]
        )
        # Gera: \begin{table}[H]...{lcccc}...
    """
    # Primeira coluna à esquerda, restante centralizada
    col_spec = 'l' + 'c' * (len(columns) - 1)
    
    header = [
        "\\begin{table}[H]",  # [H] = posição exata (requer pacote float)
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",  # Fonte reduzida para tabelas grandes
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",  # Linha superior (booktabs)
        " & ".join([f"\\textbf{{{col}}}" for col in columns]) + " \\\\"  # Cabeçalhos em negrito
    ]
    return header

def create_latex_table_footer():
    """
    Cria rodapé de tabela LaTeX.
    
    Returns:
        list: Linhas de código LaTeX do rodapé
        
    Formato:
        - \bottomrule: Linha inferior (booktabs)
        - Fechamento de ambientes tabular e table
    """
    return [
        "\\bottomrule",  # Linha inferior (booktabs)
        "\\end{tabular}",
        "\\end{table}"
    ]

# ============================================================================
# UTILITÁRIOS GERAIS
# ============================================================================

def ensure_dir(path):
    """
    Garante que diretório existe, criando se necessário.
    
    Args:
        path (str): Caminho do diretório
        
    Comportamento:
        - Cria diretório e todos os pais necessários
        - Não gera erro se diretório já existir
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_latest_benchmark_file(directory='benchmark_results'):
    """
    Obtém arquivo JSON de benchmark mais recente.
    
    Args:
        directory (str): Diretório de resultados
        
    Returns:
        Path: Caminho do arquivo mais recente ou None
        
    Critério de seleção:
        Arquivo modificado mais recentemente (mtime)
        
    Padrão de arquivo esperado:
        benchmark_results_YYYYMMDD_HHMMSS.json
    """
    dir_path = Path(directory)
    json_files = list(dir_path.glob('benchmark_results_*.json'))
    
    if not json_files:
        return None
    
    # Retorna arquivo com maior timestamp de modificação
    return max(json_files, key=lambda p: p.stat().st_mtime)

def summarize_results(metrics):
    """
    Gera sumário textual dos resultados de benchmark.
    
    Args:
        metrics (dict): Estrutura completa de métricas
        
    Returns:
        str: Sumário formatado em texto
        
    Formato de saída:
        ================================================================================
        BENCHMARK RESULTS SUMMARY
        ================================================================================
        
        C_PI
        --------------------------------------------------------------------------------
          small      | Best:  8 threads | Speedup:   7.85× | Efficiency:  98.1% | Overhead:   0.15
          medium     | Best: 16 threads | Speedup:  15.20× | Efficiency:  95.0% | Overhead:   0.53
          ...
        
    Para cada aplicação e tamanho, mostra:
        - Melhor configuração de threads (maior speedup)
        - Speedup alcançado
        - Eficiência percentual
        - Overhead calculado
    """
    summary = []
    summary.append("=" * 80)
    summary.append("BENCHMARK RESULTS SUMMARY")
    summary.append("=" * 80)
    
    for app in sorted(metrics.keys()):
        summary.append(f"\n{app.upper()}")
        summary.append("-" * 80)
        
        for size in ['small', 'medium', 'large', 'huge', 'extreme']:
            if size not in metrics[app]:
                continue
                
            best_threads, best_metrics = get_best_configuration(metrics, app, size)
            if best_metrics:
                summary.append(f"  {size:10} | Best: {best_threads:2} threads | "
                             f"Speedup: {best_metrics['speedup']:6.2f}× | "
                             f"Efficiency: {best_metrics['efficiency']*100:5.1f}% | "
                             f"Overhead: {best_metrics['overhead']:6.2f}")
    
    summary.append("\n" + "=" * 80)
    return '\n'.join(summary)
