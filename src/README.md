# OpenMP Benchmark Suite - Consolidated

Sistema consolidado de benchmarking para aplicações OpenMP com análise integrada e visualização.

## 📁 Estrutura do Projeto

```
src/
├── benchmark_runner.py       # Motor principal de execução de benchmarks
├── analyze_and_visualize.py  # Análise e geração de gráficos consolidados
├── utils.py                  # Funções utilitárias compartilhadas
├── benchmark_results/        # Diretório de saída para resultados
└── applications/             # Aplicações OpenMP de teste
```

## 🚀 Início Rápido

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Benchmarks
```bash
# Teste rápido (tamanhos pequenos, configurações limitadas)
python3 benchmark_runner.py --suite quick-test

# Suite completa de benchmarks
python3 benchmark_runner.py --suite full-test

# Configuração personalizada
python3 benchmark_runner.py --apps c_mandel c_md --sizes large extreme --threads 1 2 4 8 12 16 24
```

### 3. Gerar Análises e Visualizações
```bash
# Gerar todos os gráficos e tabelas
python3 analyze_and_visualize.py

# Gerar apenas gráficos
python3 analyze_and_visualize.py --graphs-only

# Gerar apenas tabelas
python3 analyze_and_visualize.py --tables-only

# Analisar aplicações específicas
python3 analyze_and_visualize.py --apps c_mandel c_md c_lu
```

## 📜 Scripts Disponíveis

### benchmark_runner.py
Motor principal de execução de benchmarks. Executa aplicações OpenMP com várias configurações e coleta métricas de desempenho.

**Funcionalidades:**
- Execução automatizada com múltiplas contagens de threads
- Múltiplos tamanhos de problema (small, medium, large, huge, extreme)
- Validação estatística (5 repetições por configuração)
- Rastreamento de progresso e resultados intermediários
- Saída em formatos JSON e CSV

**Opções Principais:**
- `--suite {quick-test,full-test}` - Suites de benchmark predefinidas
- `--apps APP [APP ...]` - Aplicações específicas para testar
- `--sizes {small,medium,large,huge,extreme}` - Tamanhos de problema
- `--threads N [N ...]` - Contagens de threads para testar
- `--repetitions N` - Número de repetições por configuração (padrão: 5)

### analyze_and_visualize.py
Script consolidado de análise e visualização. Gera todos os gráficos e tabelas LaTeX a partir dos resultados de benchmark.

**Saídas Geradas:**

*Gráficos (salvos em ../tcc/Graficos/):*
- `{app}_speedup_efficiency.png` - Gráficos de speedup e eficiência
- `{app}_overhead_serial.png` - Análise de overhead e fração serial
- `{app}_polynomial_comparison.png` - Avaliação de qualidade do ajuste polinomial
- `comparison_{size}.png` - Comparação entre múltiplas aplicações

*Tabelas LaTeX (salvas em benchmark_results/):*
- `tabela_overhead_extreme_24.tex` - Tabela de análise de overhead
- `tabela_resultados_summary.tex` - Resumo dos melhores resultados

**Opções Principais:**
- `--json PATH` - Caminho para arquivo JSON de benchmark
- `--graphs-only` - Pular geração de tabelas
- `--tables-only` - Pular geração de gráficos
- `--apps APP [APP ...]` - Filtrar aplicações específicas

### utils.py
Funções utilitárias compartilhadas para processamento de dados, cálculos de métricas e formatação LaTeX.

**Funções Principais:**
- `calculate_speedup(T1, Tp)` - Cálculo de speedup
- `calculate_efficiency(speedup, p)` - Cálculo de eficiência
- `calculate_overhead(T1, Tp, p)` - Cálculo de overhead
- `calculate_serial_fraction(speedup, p)` - Fração serial Karp-Flatt
- `polynomial_fit(x_data, y_data, degree)` - Regressão polinomial com R²
- `get_best_configuration(metrics, app, size)` - Encontrar contagem ótima de threads

## 📊 Métricas Calculadas

- **Speedup**: S(p) = T₁ / Tₚ
- **Eficiência**: E(p) = S(p) / p × 100%
- **Overhead**: φ = (p × Tₚ - T₁) / T₁
- **Fração Serial**: ε = (1/S - 1/p) / (1 - 1/p) (Karp-Flatt)
- **Score de Escalabilidade**: Es = E × (1 - φ/max(φ))

## 🎯 Aplicações Testadas

### Aplicações Principais
- **c_pi** - Cálculo de Pi (integração numérica)
- **c_mandel** - Gerador do conjunto de Mandelbrot
- **c_qsort** - Quicksort paralelo
- **c_fft** - Transformada Rápida de Fourier
- **c_fft6** - Variante FFT de 6 pontos
- **c_md** - Simulação de Dinâmica Molecular
- **c_lu** - Decomposição LU
- **c_jacobi01-03** - Solucionadores iterativos Jacobi (3 variantes)

### Variações de Granularidade
Cada aplicação principal (exceto FFT6 e Jacobi) possui três variantes de granularidade:
- `{app}` - Padrão (granularidade balanceada)
- `{app}_fine` - Granularidade fina (scheduling dinâmico, chunks pequenos)
- `{app}_coarse` - Granularidade grossa (scheduling estático, chunks grandes)

## 📂 Arquivos de Saída

### Resultados de Benchmark
- `benchmark_results_YYYYMMDD_HHMMSS.json` - Resultados completos (JSON)
- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Resultados tabulares (CSV)
- `benchmark_summary_YYYYMMDD_HHMMSS.txt` - Resumo legível
- `progress_YYYYMMDD_HHMMSS.json` - Rastreador de progresso em tempo real

### Gráficos (PNG 300 DPI)
- Gráficos individuais de speedup/eficiência por aplicação
- Análise de overhead e fração serial
- Comparações de ajuste polinomial
- Gráficos comparativos entre múltiplas aplicações

### Tabelas LaTeX
- Tabelas de comparação de overhead
- Resumos dos melhores resultados
- Rankings de score de escalabilidade

## 💡 Exemplo de Workflow

```bash
# 1. Executar suite completa de benchmarks
python3 benchmark_runner.py --suite full-test

# 2. Gerar todas as visualizações
python3 analyze_and_visualize.py

# 3. Gerar análise para aplicações específicas
python3 analyze_and_visualize.py --apps c_mandel c_md c_lu

# 4. Visualizar resultados
ls -lh benchmark_results/
ls -lh ../tcc/Graficos/
```

## 📦 Requisitos

- Python 3.8+
- numpy
- matplotlib
- json, csv (biblioteca padrão)

Instalar via:
```bash
pip install -r requirements.txt
```

## 🎓 Dimensões de Problema

| Aplicação | small | medium | large | huge | extreme |
|-----------|-------|--------|-------|------|---------|
| c_pi | 200k iter. | 300k iter. | 500k iter. | 1M iter. | 2M iter. |
| c_mandel | 200k pts | 300k pts | 500k pts | 1M pts | 2M pts |
| c_qsort | 150 KB | 1.5 MB | 15 MB | 75 MB | 300 MB |
| c_fft | 4 KB | 16 KB | 64 KB | 256 KB | 1 MB |
| c_fft6 | 16k pts | 33k pts | 66k pts | 262k pts | 1M pts |
| c_md | 1024 part./5 passos | 2048 part./10 passos | 4096 part./20 passos | 6144 part./30 passos | 10240 part./50 passos |
| c_lu | 96×96 | 384×384 | 1152×1152 | 2304×2304 | 5760×5760 |
| c_jacobi | 256×256/50 iter. | 1024×1024/200 iter. | 3072×3072/600 iter. | 6144×6144/1200 iter. | 15360×15360/3000 iter. |

## 🔧 Mudanças Recentes

### v2.0 - Consolidação (2025-11-30)
- ✅ Consolidados 17 scripts em apenas 3
- ✅ Script unificado de análise e visualização
- ✅ Biblioteca de utilitários compartilhados
- ✅ Documentação atualizada e simplificada
- ✅ Removidos geradores de tabelas obsoletos
- ✅ Mantido benchmark_runner.py intacto (motor principal)

### Scripts Consolidados
**Removidos** (funcionalidade integrada):
- generate_analysis.py
- generate_all_graphs.py
- generate_additional_graphs.py
- generate_polynomial_fit_graphs.py
- generate_graphs.py
- generate_plots.py
- generate_latex_tables.py
- generate_full_results_table.py
- generate_overhead_table.py
- generate_scalability_scores.py
- analyze_dimension_impact.py
- analyze_results.py
- update_tex_results.py
- fix_underscores.py
- restore_underscores.py
- validate_graphs.py
- list_apps.py

**Mantidos** (essenciais):
- benchmark_runner.py (motor de execução)
- analyze_and_visualize.py (análise consolidada)
- utils.py (funções compartilhadas)

## 📝 Licença

Ver arquivo LICENSE
