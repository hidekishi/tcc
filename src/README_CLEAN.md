# OmpSCR v2.0 - Sistema de Benchmark Integrado

Sistema automatizado para execução e análise de benchmarks OpenMP com análise de performance integrada.

## 🚀 Arquivos Essenciais

### Scripts Principais
- **`benchmark_runner.py`** - Sistema principal integrado (execução + análise automática)
- **`monitor_progress.py`** - Monitoramento de progresso em tempo real

### Build e Configuração
- **`Makefile`** / **`GNUmakefile`** - Sistema de build dos benchmarks
- **`requirements.txt`** - Dependências Python necessárias

### Documentação
- **`IMPLEMENTACAO_COMPLETA.md`** - Documentação detalhada das funcionalidades
- **`BENCHMARK_README.md`** - Guia específico dos benchmarks
- **`USAGE_GUIDE.md`** - Guia de uso completo

## ⚡ Uso Rápido

### Teste Básico com Análise Automática
```bash
python3 benchmark_runner.py --quick-test --auto-analyze
```

### Teste Completo
```bash
python3 benchmark_runner.py --full-test --auto-analyze
```

### Teste de Escalabilidade Extrema
```bash
python3 benchmark_runner.py --extreme-test --auto-analyze
```

### Monitoramento (em terminal separado)
```bash
python3 monitor_progress.py
```

## 📊 Funcionalidades Integradas

- ✅ **Execução automatizada** de 17 benchmarks OpenMP
- ✅ **Análise automática pós-execução** com gráficos e relatórios
- ✅ **9 níveis de tamanho** de problema (tiny → gigantic)
- ✅ **Monitoramento em tempo real** do progresso
- ✅ **Relatórios detalhados** de speedup e eficiência paralela
- ✅ **Interface unificada** para execução + análise

## 🔧 Configuração e Instalação

```bash
# 1. Compilar benchmarks
make clean && make

# 2. Instalar dependências Python
pip3 install -r requirements.txt

# 3. Verificar instalação
python3 benchmark_runner.py --list
```

## 📁 Estrutura Limpa

```
src/
├── benchmark_runner.py     # Sistema principal integrado
├── monitor_progress.py     # Monitor de progresso  
├── Makefile               # Sistema de build
├── requirements.txt       # Dependências Python
├── IMPLEMENTACAO_COMPLETA.md  # Documentação detalhada
├── applications/          # Código fonte dos benchmarks
├── bin/                  # Binários compilados
├── benchmark_results/    # Resultados salvos
└── doc/                  # Documentação técnica
```

## 📈 Tamanhos de Problema Disponíveis

| Tamanho   | Grid     | Iterações | Array      | FFT    | Uso de Memória |
|-----------|----------|-----------|------------|--------|----------------|
| tiny      | 25x25    | 10        | 1K         | 512    | ~0.01 MB       |
| small     | 100x100  | 50        | 10K        | 1K     | ~0.1 MB        |
| medium    | 300x300  | 150       | 50K        | 2K     | ~0.4 MB        |
| large     | 750x750  | 300       | 200K       | 4K     | ~1.6 MB        |
| huge      | 1.5Kx1.5K| 500       | 800K       | 8K     | ~6.4 MB        |
| extreme   | 3Kx3K    | 750       | 2M         | 16K    | ~16 MB         |
| massive   | 5Kx5K    | 1000      | 5M         | 32K    | ~40 MB         |
| colossal  | 8Kx8K    | 1500      | 10M        | 64K    | ~80 MB         |
| gigantic  | 12Kx12K  | 2000      | 20M        | 128K   | ~160 MB        |

## 🎯 Benchmarks Disponíveis (17 total)

- **c_pi** - Cálculo de π por integração numérica
- **c_mandel** - Gerador do conjunto de Mandelbrot
- **c_qsort** - Quicksort paralelo
- **c_fft / c_fft6** - Transformada rápida de Fourier
- **c_md** - Simulação de dinâmica molecular
- **c_lu** - Decomposição LU de matrizes
- **c_jacobi01/02/03** - Solvers iterativos de Jacobi
- **c_loopA_sol1/2/3** - Estratégias de paralelização corretas
- **c_loopB_pipeline** - Solução pipeline
- **c_loopA_bad, c_loopB_bad1/2** - Implementações com race conditions

Para documentação completa, consulte os arquivos markdown de documentação.
