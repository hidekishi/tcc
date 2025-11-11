# Guia de Monitoramento de CPU - OmpSCR Benchmark Runner

## Visão Geral

O sistema de benchmark agora inclui funcionalidade de monitoramento de CPU que permite visualizar quais núcleos do processador estão sendo utilizados durante a execução dos benchmarks.

## Como Usar

### Ativação do Monitoramento

Adicione a flag `--show-cpu-usage` ao comando de benchmark:

```bash
python3 benchmark_runner.py --benchmarks c_pi,c_mandel --threads 1,4,8,16 --show-cpu-usage
```

### Informações Exibidas

#### 1. Topologia do Sistema
- Número total de processadores lógicos
- Número de núcleos físicos
- Status do Hyperthreading
- Configuração NUMA
- Configuração OpenMP para afinidade

#### 2. Mapeamento de Núcleos por Execução
Para cada execução, o sistema mostra:
- **CPU Mapping**: Lista de núcleos que serão utilizados
- **Cores used**: Lista de núcleos efetivamente utilizados (extraído do OpenMP)

### Exemplo de Saída

```
🖥️  CPU TOPOLOGY INFORMATION
==================================================
📊 Total logical processors: 24
🔧 Physical cores: 24
🧵 Logical cores (with HT): 24
⚡ Hyperthreading: Disabled

🔧 OpenMP Configuration:
   OMP_PROC_BIND: close (use adjacent cores)
   OMP_PLACES: cores (one thread per core)
   OMP_DISPLAY_AFFINITY: enabled
🏗️  NUMA nodes: 1
==================================================

[1/3]   Running c_pi (small) with 4 threads (iteration 1)...
    💻 CPU Mapping: Cores [0, 1, 2, 3] of 24 available
    ✓ Completed in 0.002s - Cores used: [0, 1, 2, 3] (some shared)
```

## Configuração OpenMP

O sistema configura automaticamente:
- `OMP_PROC_BIND=close`: Usa núcleos adjacentes
- `OMP_PLACES=cores`: Uma thread por núcleo
- `OMP_DISPLAY_AFFINITY=TRUE`: Exibe informações de afinidade
- `OMP_AFFINITY_FORMAT`: Formato personalizado para mostrar thread e núcleo

## Dependências

Para informações detalhadas de topologia, instale:
```bash
pip install psutil
```

Para informações NUMA (opcional):
```bash
sudo apt-get install numactl  # Ubuntu/Debian
```

## Casos de Uso

### Debugging de Performance
- Verificar se threads estão distribuídas corretamente
- Identificar gargalos de afinidade
- Analisar padrões de uso de núcleos

### Análise de Escalabilidade
- Comparar distribuição com 1, 4, 8, 16, 24 threads
- Verificar eficiência do uso de núcleos
- Identificar saturação do sistema

### Otimização
- Ajustar políticas de afinidade OpenMP
- Configurar NUMA binding
- Otimizar para topologia específica

## Benchmarks Disponíveis

Execute sem parâmetros para ver lista completa:
```bash
python3 benchmark_runner.py --help
```

Principais benchmarks:
- `c_pi`: Cálculo de Pi
- `c_mandel`: Conjunto de Mandelbrot  
- `c_fft`: Transform. Fourier Rápida
- `c_qsort`: QuickSort paralelo
- `c_jacobi01/02/03`: Jacobi com diferentes implementações

## Tamanhos de Problema Otimizados

O sistema agora usa 5 níveis otimizados:
- `small`: 2MB
- `medium`: 16MB  
- `large`: 64MB
- `huge`: 256MB
- `extreme`: 1GB

## Limitações

- Máximo de 24 threads (limite do hardware atual)
- Informações detalhadas requerem psutil
- NUMA info requer numactl

## Exemplo Completo

```bash
# Teste básico com monitoramento
python3 benchmark_runner.py --benchmarks c_pi --threads 1,4,8 --show-cpu-usage

# Teste extenso com múltiplos benchmarks
python3 benchmark_runner.py --benchmarks c_pi,c_mandel,c_fft --problem-sizes small,medium --threads 1,2,4,8,16 --iterations 3 --show-cpu-usage

# Análise de escalabilidade máxima
python3 benchmark_runner.py --benchmarks c_mandel --problem-sizes huge --threads 1,2,4,8,12,16,20,24 --iterations 5 --show-cpu-usage
```
