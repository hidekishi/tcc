# OpenMP Source Code Repository - Benchmark Suite

Sistema de benchmarking para aplicações OpenMP com variantes de granularidade fina e grossa.

## 📋 Visão Geral

Este repositório contém aplicações OpenMP otimizadas para análise de desempenho paralelo. Cada aplicação possui três versões:
- **Standard**: Versão original com paralelização padrão
- **Fine-grained**: Granularidade fina com scheduling dinâmico e chunks pequenos
- **Coarse-grained**: Granularidade grossa com scheduling estático e chunks grandes

## 🚀 Quick Start

### 1. Compilação
```bash
# Compilar todas as aplicações
make all

# Compilar aplicação específica
make -C applications/c_Pi
```

### 2. Execução de Benchmarks
```bash
# Benchmark básico (threads 1,2,4,8, tamanhos small,medium)
python benchmark_runner.py

# Benchmark customizado
python benchmark_runner.py \
  --applications c_pi,c_pi_fine,c_pi_coarse \
  --threads 1,2,4,8,16,24,32 \
  --sizes small,medium,large

# Benchmark completo (inclui workload extremo de ~2GB)
python benchmark_runner.py --threads 1,2,4,8,16,24,32 --sizes small,medium,large,huge,extreme
```

### 3. Resultados
Os resultados são salvos em:
- `benchmark_results/benchmark_results_[timestamp].csv`
- `benchmark_results/benchmark_results_[timestamp].json`

## 📚 Aplicações Disponíveis

### Aplicações com Variantes de Granularidade

| Aplicação | Standard | Fine-Grained | Coarse-Grained | Descrição |
|-----------|----------|--------------|----------------|-----------|
| **Pi** | c_pi | c_pi_fine | c_pi_coarse | Cálculo de π por integração numérica |
| **Mandelbrot** | c_mandel | c_mandel_fine | c_mandel_coarse | Gerador do conjunto de Mandelbrot |
| **QuickSort** | c_qsort | c_qsort_fine | c_qsort_coarse | Ordenação paralela |
| **FFT** | c_fft | c_fft_fine | c_fft_coarse | Fast Fourier Transform |
| **Jacobi** | c_jacobi01 | c_jacobi_fine | c_jacobi_coarse | Solver iterativo de Jacobi |
| **LU** | c_lu | c_lu_fine | c_lu_coarse | Decomposição LU |
| **Molecular Dynamics** | c_md | c_md_fine | c_md_coarse | Simulação de dinâmica molecular |
| **Graph Search** | c_testPath | c_testPath_fine | c_testPath_coarse | Busca de caminho em grafo |

### Outras Aplicações
- **FFT6**: Implementação FFT de 6 pontos
- **Loop Dependencies**: Exemplos de dependências em loops (c_loopA, c_loopB, c_loopC)

## 💻 Implementação dos Algoritmos (Versão Padrão)

Esta seção detalha a implementação OpenMP de cada algoritmo na sua **versão padrão (standard)**, explicando as estratégias de paralelização utilizadas.

### 🧮 Cálculo de Pi - Integração Numérica

**Método**: Integração numérica da função f(x) = 4/(1+x²) no intervalo [0,1]

**Implementação Serial:**
```c
double w = 1.0 / N;  // Largura de cada retângulo
double pi = 0.0;

for(i = 0; i < N; i++) {
    double local = (i + 0.5) * w;      // Ponto médio do intervalo
    pi += 4.0 / (1.0 + local * local); // Altura do retângulo
}
pi *= w;  // Multiplicar pela largura para obter área
```

**Paralelização OpenMP:**
```c
#pragma omp parallel for default(shared) private(i, local) reduction(+:pi)
for(i = 0; i < N; i++) {
    local = (i + 0.5) * w;
    pi += 4.0 / (1.0 + local * local);
}
```

**Características:**
- **Cláusula `parallel for`**: Distribui iterações entre threads
- **`reduction(+:pi)`**: Cada thread acumula em cópia local, soma final automática
- **`private(i, local)`**: Cada thread tem suas próprias variáveis
- **Scheduling implícito**: `static` (default) - chunks contíguos
- **Workload**: Perfeitamente balanceado (cada iteração tem custo uniforme)

**Padrão de Acesso à Memória:**
- ✅ Sem dependências de dados entre iterações
- ✅ Acesso sequencial às iterações (cache-friendly)
- ⚠️ Contenção na reduction (sincronização final)

---

### 🎨 Mandelbrot Set - Monte Carlo Sampling

**Método**: Amostragem Monte Carlo para estimar área do conjunto de Mandelbrot

**Implementação Serial:**
```c
// 1. Gerar pontos aleatórios no plano complexo
for (i = 0; i < NPOINTS; i++) {
    points[i].re = -2.0 + 2.5 * random() / MAX;
    points[i].im = 1.125 * random() / MAX;
}

// 2. Testar cada ponto: pertence ao conjunto?
outside = 0;
for(i = 0; i < NPOINTS; i++) {
    z = points[i];  // z₀ = c
    for (j = 0; j < MAXITER; j++) {
        z = z² + c;  // Iteração do Mandelbrot
        if (|z| > 2.0) {
            outside++;
            break;  // Ponto diverge
        }
    }
}
area = 2.0 * (2.5 * 1.125) * (NPOINTS - outside) / NPOINTS;
```

**Paralelização OpenMP:**
```c
#pragma omp parallel for default(none) reduction(+:outside) \
                         private(i, j, ztemp, z) shared(NPOINTS, points)
for(i = 0; i < NPOINTS; i++) {
    z.re = points[i].re;
    z.im = points[i].im;
    for (j = 0; j < MAXITER; j++) {
        ztemp = (z.re * z.re) - (z.im * z.im) + points[i].re;
        z.im = z.re * z.im * 2 + points[i].im;
        z.re = ztemp;
        if (z.re * z.re + z.im * z.im > THRESHOLD) {
            outside++;
            break;
        }
    }
}
```

**Características:**
- **`default(none)`**: Força especificação explícita de todas as variáveis
- **Workload irregular**: Pontos divergem em iterações diferentes (1 a MAXITER)
- **Embaraçosamente paralelo**: Pontos independentes entre si
- **Loop interno não paralelizado**: Dependência temporal (z_{n+1} depende de z_n)

**Padrão de Acesso à Memória:**
- ✅ Read-only do array `points[]` (compartilhado)
- ✅ Variáveis locais (`z`, `ztemp`) privadas
- ⚠️ Workload desbalanceado favorece dynamic scheduling

**Por que Standard usa Static?**
- Com muitos pontos (2M+), desbalanceamento se ameniza estatisticamente
- Overhead de dynamic scheduling não compensa
- Static tem melhor cache locality

---

### 🔀 QuickSort - Ordenação Paralela

**Método**: Algoritmo divide-and-conquer recursivo com paralelização por tasks

**Implementação Serial:**
```c
void quicksort(int *v, int left, int right) {
    if (left < right) {
        int pivot_index = partition(v, left, right);
        quicksort(v, left, pivot_index - 1);   // Esquerda
        quicksort(v, pivot_index + 1, right);  // Direita
    }
}
```

**Paralelização OpenMP:**
```c
void quicksort_tasks(int *v, int left, int right, int cutoff) {
    if (left < right) {
        int pivot_index = partition(v, left, right);
        
        #pragma omp task shared(v) if(right - left > cutoff)
        quicksort_tasks(v, left, pivot_index - 1, cutoff);
        
        #pragma omp task shared(v) if(right - left > cutoff)
        quicksort_tasks(v, pivot_index + 1, right, cutoff);
        
        #pragma omp taskwait  // Aguarda ambas as subtarefas
    }
}

// Função principal
#pragma omp parallel
{
    #pragma omp single
    quicksort_tasks(v, 0, n-1, CUTOFF);
}
```

**Características:**
- **Task-based parallelism**: Recursão paralela com `#pragma omp task`
- **Cutoff threshold**: `if(right - left > cutoff)` evita overhead em partições pequenas
- **`single` clause**: Apenas uma thread cria a árvore inicial de tasks
- **`taskwait`**: Sincronização para aguardar conclusão das subtasks
- **Load balancing**: Work-stealing queue gerencia distribuição de tasks

**Padrão de Acesso à Memória:**
- ⚠️ Acesso não sequencial durante partition (cache misses)
- ✅ Partições independentes após partition (paralelismo sem conflitos)
- ⚠️ In-place sorting → potencial false sharing

**Cutoff Standard**: ~10,000 elementos
- Abaixo disso, execução serial é mais eficiente
- Overhead de criar task > ganho de paralelização

---

### 🌊 FFT - Fast Fourier Transform

**Método**: Algoritmo Cooley-Tukey recursivo (divide-and-conquer)

**Implementação Serial (Radix-2):**
```c
void fft_recursive(complex *x, int N) {
    if (N <= 1) return;
    
    // Divide: separar pares e ímpares
    complex *even = malloc(N/2 * sizeof(complex));
    complex *odd = malloc(N/2 * sizeof(complex));
    for (int i = 0; i < N/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }
    
    // Conquer: FFT recursiva
    fft_recursive(even, N/2);
    fft_recursive(odd, N/2);
    
    // Combine: aplicar twiddle factors
    for (int k = 0; k < N/2; k++) {
        complex t = cexp(-2.0 * M_PI * I * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N/2] = even[k] - t;
    }
}
```

**Paralelização OpenMP:**
```c
void fft_parallel(complex *x, int N, int cutoff) {
    if (N <= cutoff) {
        fft_serial(x, N);  // Abaixo do cutoff, serial
        return;
    }
    
    // Separar pares e ímpares
    #pragma omp parallel for
    for (int i = 0; i < N/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }
    
    // FFT recursiva paralela
    #pragma omp task shared(even)
    fft_parallel(even, N/2, cutoff);
    
    #pragma omp task shared(odd)
    fft_parallel(odd, N/2, cutoff);
    
    #pragma omp taskwait
    
    // Combine com twiddle factors
    #pragma omp parallel for
    for (int k = 0; k < N/2; k++) {
        complex t = twiddle[k] * odd[k];
        x[k] = even[k] + t;
        x[k + N/2] = even[k] - t;
    }
}
```

**Características:**
- **Recursão paralela**: Tasks para chamadas recursivas
- **Paralelização em 3 níveis**:
  1. Split (separar pares/ímpares) → `parallel for`
  2. Conquer (FFTs recursivas) → `task`
  3. Combine (twiddle factors) → `parallel for`
- **Cutoff adaptativo**: Standard usa 4096 pontos
- **Complexidade**: O(N log N) → paralelização eficiente

**Padrão de Acesso à Memória:**
- ⚠️ Stride-2 access no split (cache não sequencial)
- ⚠️ Butterfly pattern no combine (cache misses)
- ✅ Recursões independentes (sem dependências)

---

### 🔄 Jacobi Iterative Solver

**Método**: Solver iterativo para equações diferenciais parciais (stencil 5-pontos)

**Implementação Serial:**
```c
// Iterar até convergência
for (iter = 0; iter < max_iter; iter++) {
    // Aplicar stencil 5-pontos
    for (i = 1; i < m-1; i++) {
        for (j = 1; j < n-1; j++) {
            u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
                              u_old[i][j-1] + u_old[i][j+1]);
        }
    }
    
    // Copiar u → u_old para próxima iteração
    memcpy(u_old, u, m * n * sizeof(double));
}
```

**Paralelização OpenMP:**
```c
for (iter = 0; iter < max_iter; iter++) {
    #pragma omp parallel for private(i, j) shared(u, u_old, m, n)
    for (i = 1; i < m-1; i++) {
        for (j = 1; j < n-1; j++) {
            u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
                              u_old[i][j-1] + u_old[i][j+1]);
        }
    }
    
    // Barrier implícita no fim do parallel for
    
    #pragma omp parallel for
    for (i = 0; i < m; i++) {
        memcpy(u_old[i], u[i], n * sizeof(double));
    }
}
```

**Características:**
- **Stencil computation**: Cada ponto depende de 4 vizinhos
- **Dependência temporal**: Iteração K depende de K-1 (não paralelizável entre iterações)
- **Independência espacial**: Pontos na mesma iteração são independentes
- **Barrier implícita**: Garante que todos atualizaram antes da cópia
- **Workload uniforme**: Cada ponto tem exatamente 4 operações

**Padrão de Acesso à Memória:**
- ✅ Acesso sequencial por linhas (cache-friendly)
- ⚠️ Acesso vertical (u[i±1][j]) pode causar cache miss
- ✅ Workload perfeitamente balanceado → static scheduling ideal

**Otimizações Possíveis:**
- Red-black coloring para paralelizar iterações
- Blocking/tiling para melhor cache utilization

---

### 🔺 LU Decomposition

**Método**: Decomposição de matriz A = L × U (Lower × Upper triangular)

**Implementação Serial (Algoritmo de Doolittle):**
```c
for (k = 0; k < N; k++) {
    // Calcular U[k][j]
    for (j = k; j < N; j++) {
        U[k][j] = A[k][j];
        for (int s = 0; s < k; s++) {
            U[k][j] -= L[k][s] * U[s][j];
        }
    }
    
    // Calcular L[i][k]
    for (i = k+1; i < N; i++) {
        L[i][k] = A[i][k];
        for (int s = 0; s < k; s++) {
            L[i][k] -= L[i][s] * U[s][k];
        }
        L[i][k] /= U[k][k];  // Divisão por pivô
    }
}
```

**Paralelização OpenMP:**
```c
for (k = 0; k < N; k++) {
    // Linha k de U (paralelizável)
    #pragma omp parallel for private(j, s) shared(U, L, A, k, N)
    for (j = k; j < N; j++) {
        U[k][j] = A[k][j];
        for (s = 0; s < k; s++) {
            U[k][j] -= L[k][s] * U[s][j];
        }
    }
    
    // Coluna k de L (paralelizável após linha k)
    #pragma omp parallel for private(i, s) shared(U, L, A, k, N)
    for (i = k+1; i < N; i++) {
        L[i][k] = A[i][k];
        for (s = 0; s < k; s++) {
            L[i][k] -= L[i][s] * U[s][k];
        }
        L[i][k] /= U[k][k];
    }
}
```

**Características:**
- **Dependência por nível**: Iteração k deve completar antes de k+1
- **Paralelismo interno**: Dentro de cada k, linhas/colunas independentes
- **Barrier implícita**: Entre cálculo de U e L
- **Workload decrescente**: Menos trabalho a cada iteração k
- **Operações densas**: Alta intensidade aritmética (compute-bound)

**Padrão de Acesso à Memória:**
- ✅ Acesso por linhas em U (cache-friendly)
- ⚠️ Acesso por colunas em L (cache não sequencial)
- ✅ Reutilização de dados em loops internos

**Desafios de Paralelização:**
- Loop externo (k) é serial
- Paralelização apenas em loops internos
- Speedup limitado por Lei de Amdahl

---

### ⚛️ Molecular Dynamics - N-body Simulation

**Método**: Simulação de dinâmica molecular com forças de Lennard-Jones

**Implementação Serial:**
```c
for (step = 0; step < n_steps; step++) {
    // 1. Calcular forças entre todos os pares
    for (i = 0; i < n_particles; i++) {
        force[i] = {0, 0, 0};
        for (j = i+1; j < n_particles; j++) {
            vec3 r = pos[j] - pos[i];
            double dist = length(r);
            
            // Força Lennard-Jones: F = 24ε[(2(σ/r)¹³ - (σ/r)⁷)]
            double f_mag = 24 * epsilon * 
                          (2 * pow(sigma/dist, 13) - pow(sigma/dist, 7));
            vec3 f = f_mag * r / dist;
            
            force[i] += f;
            force[j] -= f;  // Lei de Newton: F_ij = -F_ji
        }
    }
    
    // 2. Integração de velocidade e posição (Verlet)
    for (i = 0; i < n_particles; i++) {
        vel[i] += force[i] / mass[i] * dt;
        pos[i] += vel[i] * dt;
    }
}
```

**Paralelização OpenMP:**
```c
for (step = 0; step < n_steps; step++) {
    // Calcular forças (paralelização do loop externo)
    #pragma omp parallel for private(i, j, r, dist, f_mag, f) \
                             shared(pos, force, n_particles) \
                             schedule(dynamic, 64)
    for (i = 0; i < n_particles; i++) {
        force[i] = {0, 0, 0};
        for (j = i+1; j < n_particles; j++) {
            vec3 r = pos[j] - pos[i];
            double dist = length(r);
            double f_mag = 24 * epsilon * 
                          (2 * pow(sigma/dist, 13) - pow(sigma/dist, 7));
            vec3 f = f_mag * r / dist;
            
            // Atomic para evitar race condition
            #pragma omp atomic
            force[i].x += f.x;
            // ... (y, z similar)
            
            #pragma omp atomic
            force[j].x -= f.x;
        }
    }
    
    // Integração (paralelização direta)
    #pragma omp parallel for
    for (i = 0; i < n_particles; i++) {
        vel[i] += force[i] / mass[i] * dt;
        pos[i] += vel[i] * dt;
    }
}
```

**Características:**
- **O(N²) complexity**: Força entre todos os pares
- **Race condition**: force[j] é escrito por múltiplas threads
- **Solução 1**: `atomic` (usado acima) - overhead alto
- **Solução 2**: Private force arrays + reduction
- **Solução 3**: Spatial partitioning (cell lists) → O(N)

**Padrão de Acesso à Memória:**
- ⚠️ Acesso aleatório a pos[j] (cache miss frequente)
- ⚠️ False sharing em force[] se não padded
- ✅ Integração é perfeitamente paralela

**Otimizações Standard:**
- Dynamic scheduling (workload desbalanceado: partículas com diferentes números de vizinhos)
- Chunk size 64 para balancear overhead vs load balance

---

### 🔍 Graph Search - BFS/DFS

**Método**: Busca em largura (BFS) ou profundidade (DFS) em grafo

**Implementação Serial (BFS):**
```c
void bfs(Graph *g, int start) {
    bool visited[g->V] = {false};
    int queue[g->V], front = 0, rear = 0;
    
    visited[start] = true;
    queue[rear++] = start;
    
    while (front < rear) {
        int u = queue[front++];
        
        // Visitar vizinhos
        for (int i = 0; i < g->adj[u].size; i++) {
            int v = g->adj[u].nodes[i];
            if (!visited[v]) {
                visited[v] = true;
                queue[rear++] = v;
            }
        }
    }
}
```

**Paralelização OpenMP (Level-synchronous BFS):**
```c
void bfs_parallel(Graph *g, int start) {
    bool visited[g->V] = {false};
    int *current_level, *next_level;
    int curr_size, next_size;
    
    visited[start] = true;
    current_level[0] = start;
    curr_size = 1;
    
    while (curr_size > 0) {
        next_size = 0;
        
        // Processar nível atual em paralelo
        #pragma omp parallel for shared(current_level, next_level, visited) \
                                 reduction(+:next_size)
        for (int i = 0; i < curr_size; i++) {
            int u = current_level[i];
            
            for (int j = 0; j < g->adj[u].size; j++) {
                int v = g->adj[u].nodes[j];
                
                // Usar atomic para marcar como visitado
                bool was_visited;
                #pragma omp atomic capture
                {
                    was_visited = visited[v];
                    visited[v] = true;
                }
                
                if (!was_visited) {
                    int pos;
                    #pragma omp atomic capture
                    pos = next_size++;
                    
                    next_level[pos] = v;
                }
            }
        }
        
        // Trocar níveis
        swap(current_level, next_level);
        curr_size = next_size;
    }
}
```

**Características:**
- **Level-synchronous**: Processa um nível do grafo por vez
- **Workload irregular**: Vértices têm graus diferentes
- **Race condition**: Múltiplas threads podem tentar visitar mesmo vértice
- **Solução**: `atomic capture` para visited[] e next_size
- **Barrier implícita**: Entre níveis da BFS

**Padrão de Acesso à Memória:**
- ⚠️ Acesso completamente irregular (depende da topologia do grafo)
- ⚠️ Cache misses muito frequentes
- ⚠️ Difícil balanceamento de carga

**Desafios:**
- Paralelização eficiente de grafos é tema de pesquisa ativo
- Speedup limitado para grafos pequenos
- Melhor em grafos grandes e densos

---

## 🔧 Estratégias de Granularidade

### Fine-Grained (Granularidade Fina)
- **Scheduling**: Dinâmico com chunks pequenos (1-10 elementos)
- **Vantagens**: Melhor balanceamento de carga em workloads irregulares
- **Desvantagens**: Maior overhead de sincronização
- **Uso recomendado**: Workloads heterogêneos, convergência irregular

**Exemplos:**
```c
// Pi - Dynamic scheduling, chunk 1
#pragma omp parallel for schedule(dynamic, 1)

// Mandelbrot - Dynamic scheduling, chunk 10
#pragma omp parallel for schedule(dynamic, 10)

// QuickSort - Task cutoff 1000 elementos
#pragma omp task if(right-left > 1000)
```

### Coarse-Grained (Granularidade Grossa)
- **Scheduling**: Estático com chunks grandes (size/threads)
- **Vantagens**: Menor overhead, melhor cache locality
- **Desvantagens**: Possível desbalanceamento em workloads irregulares
- **Uso recomendado**: Workloads regulares, minimização de overhead

**Exemplos:**
```c
// Pi - Static scheduling, large chunks
chunk_size = N / (NUMTHREADS * 4);
#pragma omp parallel for schedule(static, chunk_size)

// QuickSort - Task cutoff 100000 elementos
#pragma omp task if(right-left > 100000)

// Jacobi - Static with calculated chunks
chunk_size = m / num_threads;
#pragma omp parallel for schedule(static, chunk_size)
```

## 📐 Análise de Complexidade Assintótica

Esta seção apresenta a análise de complexidade computacional das aplicações em sua **forma serial (1 thread)**, incluindo complexidade assintótica (Big-O) e polinômios de complexidade detalhados.

### 🧮 Cálculo de Pi (Monte Carlo Integration)

**Complexidade Assintótica:**
```
T(N) = O(N)
```

**Polinômio de Complexidade:**
```
T(N) = 7N + C₀
```

**Análise Detalhada:**
- **N**: Número de pontos de integração
- **Operações por iteração**: 7 FLOPs
  - 1 adição: `(i + 0.5)`
  - 1 multiplicação: `(i + 0.5) * w`
  - 1 multiplicação: `local * local`
  - 1 adição: `1.0 + local * local`
  - 1 divisão: `4.0 / (1.0 + local * local)`
  - 1 adição (reduction): `pi += ...`
  - 1 multiplicação final: `pi *= w`
- **C₀**: Overhead de inicialização (~100 operações)
- **Tipo**: Memory-bound (acesso sequencial a array)
- **Escalabilidade**: Linear com N

**Tempos Esperados (1 thread, i9-14900K):**
| N | Tempo (s) | Tempo/N | Memória |
|---|-----------|---------|---------|
| 2M (small) | ~0.015s | 7.5e-9 | 32 MB |
| 8M (medium) | ~0.060s | 7.5e-9 | 128 MB |
| 128M (huge) | ~0.960s | 7.5e-9 | 2 GB |
| 512M (extreme) | ~3.840s | 7.5e-9 | 8 GB |

⚠️ **Nota**: Complexidade O(N) linear, mas tempos podem variar devido a otimizações do compilador (vectorização AVX-512, loop unrolling).

---

### 🎨 Mandelbrot Set

**Complexidade Assintótica:**
```
T(W, H, I) = O(W × H × I)
```

**Polinômio de Complexidade:**
```
T(W, H, I) = k₁(W × H × I) + k₂(W × H) + C₀
```

**Análise Detalhada:**
- **W**: Largura da imagem (pixels)
- **H**: Altura da imagem (pixels)
- **I**: Número máximo de iterações por pixel
- **k₁ ≈ 10**: FLOPs por iteração do algoritmo de escape
  - `z_real² + z_imag²` (3 FLOPs)
  - `z_real_new = z_real² - z_imag² + c_real` (4 FLOPs)
  - `z_imag_new = 2 × z_real × z_imag + c_imag` (3 FLOPs)
- **k₂ ≈ 5**: FLOPs de inicialização por pixel
- **C₀**: Overhead de setup (~1000 operações)
- **Tipo**: Compute-bound (workload irregular por pixel)
- **Características**: 
  - Convergência não-uniforme (alguns pixels escapam rápido, outros levam I iterações)
  - Workload heterogêneo → beneficia scheduling dinâmico

**Estimativa de Complexidade:**
```
small:   2048² × 500 = 2,097,152,000 iterações (~2.1 bilhões)
medium:  4096² × 1000 = 16,777,216,000 iterações (~16.8 bilhões)
extreme: 32768² × 10000 = 10,737,418,240,000 iterações (~10.7 trilhões)
```

⚠️ **Extreme pode levar 30+ minutos em 1 thread!**

---

### 🔀 QuickSort

**Complexidade Assintótica:**
```
T(N) = O(N log N)  [caso médio]
T(N) = O(N²)       [pior caso - array já ordenado]
```

**Polinômio de Complexidade (caso médio):**
```
T(N) = c₁ × N × log₂(N) + c₂ × N + C₀
```

**Análise Detalhada:**
- **N**: Número de elementos a ordenar
- **c₁ ≈ 3**: Comparações por nível de recursão
- **c₂ ≈ 2**: Swaps e movimentações de elementos
- **log₂(N)**: Profundidade da árvore de recursão
- **C₀**: Overhead de inicialização
- **Tipo**: Memory-bound (acesso não sequencial, cache misses)
- **Características**:
  - Recursão divide-and-conquer
  - Workload desbalanceado (partições irregulares)
  - Stack depth: O(log N) [caso médio], O(N) [pior caso]

**Número de Comparações:**
```
small:   2M × log₂(2M) ≈ 2M × 20.9 = 41.8M comparações
huge:    128M × log₂(128M) ≈ 128M × 26.9 = 3.44B comparações
extreme: 512M × log₂(512M) ≈ 512M × 29.0 = 14.8B comparações
```

---

### 🌊 FFT (Fast Fourier Transform)

**Complexidade Assintótica:**
```
T(N) = O(N log N)
```

**Polinômio de Complexidade:**
```
T(N) = 5N × log₂(N) + 2N + C₀
```

**Análise Detalhada:**
- **N**: Número de pontos (deve ser potência de 2)
- **log₂(N)**: Número de estágios da FFT
- **5N por estágio**: Operações por butterfly
  - 2 multiplicações complexas (8 FLOPs)
  - 2 adições complexas (4 FLOPs)
  - Total: 12 FLOPs por butterfly, ~5N efetivo com otimizações
- **2N**: Bit-reversal permutation inicial
- **C₀**: Overhead de setup (cálculo de twiddle factors)
- **Tipo**: Compute-bound (intensivo em FLOPs)
- **Características**:
  - Acesso de memória irregular (stride powers of 2)
  - Cache-friendly em estágios iniciais, cache-hostile em finais

**FLOPs Totais:**
```
small:   16K × log₂(16K) × 5 = 16K × 14 × 5 = 1.1M FLOPs
huge:    1M × log₂(1M) × 5 = 1M × 20 × 5 = 100M FLOPs
extreme: 4M × log₂(4M) × 5 = 4M × 22 × 5 = 440M FLOPs
```

---

### 🔄 Jacobi Iterative Solver

**Complexidade Assintótica:**
```
T(M, K) = O(M² × K)
```

**Polinômio de Complexidade:**
```
T(M, K) = 5M² × K + 2M² + C₀
```

**Análise Detalhada:**
- **M**: Tamanho da grade (M × M matriz)
- **K**: Número de iterações até convergência
- **5M² por iteração**: FLOPs no stencil 5-pontos
  ```c
  u[i][j] = 0.25 × (u_old[i-1][j] + u_old[i+1][j] + 
                     u_old[i][j-1] + u_old[i][j+1])
  ```
  - 4 loads de memória
  - 4 adições
  - 1 multiplicação por 0.25
- **2M²**: Cópia de matriz u → u_old por iteração
- **C₀**: Inicialização da grade
- **Tipo**: Memory-bound (acesso padrão de memória regular)
- **Características**:
  - Convergência iterativa (K variável)
  - Stencil computation com dependências temporais
  - Workload uniforme → beneficia scheduling estático

**FLOPs Totais:**
```
small:   2048² × 500 × 5 = 10,485,760,000 FLOPs (10.5 GFLOPs)
huge:    16384² × 5000 × 5 = 1,342,177,280,000,000 FLOPs (1.34 PFLOPs)
extreme: 32768² × 10000 × 5 = 53,687,091,200,000,000 FLOPs (53.7 PFLOPs)
```

⚠️ **53.7 PetaFLOPs no extreme - pode levar HORAS em 1 thread!**

---

### 🔺 LU Decomposition

**Complexidade Assintótica:**
```
T(N) = O(N³)
```

**Polinômio de Complexidade:**
```
T(N) = (2/3)N³ + (1/2)N² + C₀
```

**Análise Detalhada:**
- **N**: Dimensão da matriz (N × N)
- **(2/3)N³**: Operações de eliminação Gaussiana
  - Outer loop k: N iterações
  - Middle loop i: (N-k) iterações  
  - Inner loop j: (N-k) iterações
  - Total: Σ(k=1 to N) (N-k)² ≈ N³/3 multiplicações + N³/3 subtrações
- **(1/2)N²**: Back-substitution
- **C₀**: Inicialização e pivoting
- **Tipo**: Compute-bound (intensivo em operações de matriz)
- **Características**:
  - Operações matriciais densas
  - Cache-friendly (acesso por blocos)
  - Workload diminui a cada iteração k

**FLOPs Totais:**
```
small:   (2/3) × 2048³ ≈ 5.7 GFLOPs
huge:    (2/3) × 16384³ ≈ 2,929,687,142,400,000 FLOPs (2.93 PFLOPs)
extreme: (2/3) × 32768³ ≈ 23,437,497,139,200,000,000 FLOPs (23.4 PFLOPs)
```

⚠️ **23.4 PetaFLOPs - LU extreme pode levar várias HORAS mesmo com 32 threads!**

---

### ⚛️ Molecular Dynamics (N-body simulation)

**Complexidade Assintótica:**
```
T(N_p, N_s) = O(N_p² × N_s)  [força bruta]
T(N_p, N_s) = O(N_p × log(N_p) × N_s)  [com spatial partitioning]
```

**Polinômio de Complexidade (força bruta):**
```
T(N_p, N_s) = 20N_p² × N_s + 6N_p × N_s + C₀
```

**Análise Detalhada:**
- **N_p**: Número de partículas
- **N_s**: Número de steps de simulação
- **20N_p²**: Cálculo de forças entre todos os pares
  - Distância: √((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²) (6 FLOPs)
  - Força Lennard-Jones: 4ε[(σ/r)¹² - (σ/r)⁶] (10 FLOPs)
  - Componentes de força (Fx, Fy, Fz): (4 FLOPs)
- **6N_p**: Integração de velocidade e posição (Verlet)
  - v(t+Δt) = v(t) + a(t)×Δt (3 FLOPs)
  - x(t+Δt) = x(t) + v(t)×Δt (3 FLOPs)
- **C₀**: Inicialização de posições/velocidades
- **Tipo**: Compute-bound com acesso irregular de memória
- **Otimização**: Spatial partitioning (cell lists, Verlet lists) → O(N_p)

**FLOPs por Step:**
```
small:   20 × (2M)² = 8 × 10¹³ FLOPs/step
huge:    20 × (128M)² = 3.3 × 10¹⁷ FLOPs/step
extreme: 20 × (512M)² = 5.2 × 10¹⁸ FLOPs/step
```

**FLOPs Totais (com steps):**
```
extreme: 5.2 × 10¹⁸ × 10000 steps = 5.2 × 10²² FLOPs
```

⚠️ **ATENÇÃO: MD extreme é O(N²) - EXTREMAMENTE PESADO! Pode levar DIAS em 1 thread!**

---

### 🔍 Graph Search (BFS/DFS)

**Complexidade Assintótica:**
```
T(V, E) = O(V + E)
```

**Polinômio de Complexidade:**
```
T(V, E) = k₁ × E + k₂ × V + C₀
```

**Análise Detalhada:**
- **V**: Número de vértices
- **E**: Número de arestas
- **k₁ ≈ 5**: Operações por aresta visitada
  - Lookup de adjacência (1 op)
  - Verificação de visitado (1 op)
  - Atualização de distância/predecessor (3 ops)
- **k₂ ≈ 3**: Operações por vértice
  - Marcação como visitado (1 op)
  - Enqueue/dequeue (2 ops)
- **C₀**: Inicialização de estruturas (visited array, queue)
- **Tipo**: Memory-bound (acesso irregular de memória)
- **Características**:
  - Dependente da estrutura do grafo (densidade)
  - Workload extremamente irregular
  - Difícil de paralelizar eficientemente

**Complexidade Dependente do Grafo:**
```
Grafo denso:    E ≈ V², então T(V) ≈ O(V²)
Grafo esparso:  E ≈ V, então T(V) ≈ O(V)
```

---

## 📊 Tabela Comparativa de Complexidade

### Tamanho Extreme (otimizado para i9-14900K, 128GB RAM)

| Aplicação | Complexidade | Classe | FLOPs (extreme) | Memória | Tempo Estimado (1T) |
|-----------|--------------|--------|-----------------|---------|---------------------|
| **Pi** | O(N) | Linear | ~3.6B | 8 GB | ~4s |
| **Mandelbrot** | O(W×H×I) | Cúbico | ~10.7T | 8 GB | ~30 min |
| **QuickSort** | O(N log N) | Loglinear | ~14.8B comparações | 4 GB | ~2 min |
| **FFT** | O(N log N) | Loglinear | ~440M | 64 MB | <1s |
| **Jacobi** | O(M²×K) | Cúbico | ~53.7 **PFLOPs** | 8.6 GB | ~2-4 horas ⚠️ |
| **LU** | O(N³) | Cúbico | ~23.4 **PFLOPs** | 8.6 GB | ~3-6 horas ⚠️ |
| **MD** | O(N_p²×N_s) | Quártico | ~5.2×10²² | 12 GB | **DIAS** ⚠️⚠️ |
| **Graph** | O(V+E) | Linear | Depende do grafo | ~5 MB | Variável |

**Legenda:**
- **B** = Bilhões (10⁹)
- **T** = Trilhões (10¹²)
- **P** = Peta (10¹⁵)
- **1T** = 1 Thread (baseline)

### 🎯 Insights de Complexidade

**Aplicações Escaláveis (Favoráveis à Paralelização):**
- ✅ **LU Decomposition**: O(N³) com alto compute-to-memory ratio
- ✅ **Molecular Dynamics**: O(N²) com cálculos intensivos por par
- ✅ **Mandelbrot**: Workload irregular mas embaraçosamente paralelo

**Aplicações Desafiadoras:**
- ⚠️ **Pi**: Trabalho trivial (7 FLOPs/iter) → overhead domina
- ⚠️ **QuickSort**: Recursão irregular + acesso não sequencial
- ⚠️ **Graph Search**: Dependências de dados + acesso irregular

**Lei de Gustafson:**
Para problemas que escalam em tamanho (como LU, MD, Jacobi), a fração paralela aumenta com N:
```
Speedup(N, P) ≈ P + (1-P) × N/N₀
```
Onde P = número de processadores, N = tamanho do problema.

---

## 📏 Tamanhos de Entrada (Input Sizes)

O benchmark possui **5 tamanhos de entrada** otimizados para **i9-14900K (24 cores, 32 threads) com 128GB RAM**, com foco em workloads que estressam paralelização:

| Tamanho | Grid Size | Iterações | Array Size | FFT Size | Memória Aprox. |
|---------|-----------|-----------|------------|----------|----------------|
| **small** | 2048 | 500 | 2M | 16384 | **~32 MB** |
| **medium** | 4096 | 1000 | 8M | 65536 | **~128 MB** |
| **large** | 8192 | 2000 | 32M | 262144 | **~512 MB** |
| **huge** | 16384 | 5000 | 128M | 1048576 | **~2 GB** |
| **extreme** | 32768 | 10000 | 512M | 4194304 | **~8 GB** |

⚠️ **Hardware Requerido:**
- **CPU**: 16+ cores recomendado (i9-14900K tem 24 cores)
- **RAM**: 16GB mínimo, 64GB+ recomendado para tamanho extreme
- **Threads**: Configurar OMP_NUM_THREADS=1,2,4,8,16,24,32

### Tamanho de Memória por Aplicação

#### Cálculo de Pi (c_pi, c_pi_fine, c_pi_coarse)
- **small**: ~32 MB (2M pontos de integração)
- **medium**: ~128 MB (8M pontos)
- **large**: ~512 MB (32M pontos)
- **huge**: ~2 GB (128M pontos)
- **extreme**: **~8 GB** (512M pontos)

#### Mandelbrot (c_mandel, c_mandel_fine, c_mandel_coarse)
- **small**: ~32 MB (2048² pixels, 500 iterações)
- **medium**: ~128 MB (4096² pixels, 1000 iterações)
- **large**: ~512 MB (8192² pixels, 2000 iterações)
- **huge**: ~2 GB (16384² pixels, 5000 iterações)
- **extreme**: **~8 GB** (32768² pixels, 10000 iterações)

#### QuickSort (c_qsort, c_qsort_fine, c_qsort_coarse)
- **small**: ~16 MB (2M elementos double)
- **medium**: ~64 MB (8M elementos)
- **large**: ~256 MB (32M elementos)
- **huge**: ~1 GB (128M elementos)
- **extreme**: **~4 GB** (512M elementos)

#### FFT (c_fft, c_fft_fine, c_fft_coarse)
- **small**: ~256 KB (16384 pontos complexos)
- **medium**: ~1 MB (65536 pontos)
- **large**: ~4 MB (262144 pontos)
- **huge**: ~16 MB (1048576 pontos)
- **extreme**: **~64 MB** (4194304 pontos)

#### Jacobi Solver (c_jacobi01, c_jacobi_fine, c_jacobi_coarse)
- **small**: ~33 MB (grid 2048×2048, 500 iterações)
- **medium**: ~134 MB (grid 4096×4096, 1000 iterações)
- **large**: ~536 MB (grid 8192×8192, 2000 iterações)
- **huge**: ~2.1 GB (grid 16384×16384, 5000 iterações)
- **extreme**: **~8.6 GB** (grid 32768×32768, 10000 iterações)

#### LU Decomposition (c_lu, c_lu_fine, c_lu_coarse)
- **small**: ~33 MB (matriz 2048×2048)
- **medium**: ~134 MB (matriz 4096×4096)
- **large**: ~536 MB (matriz 8192×8192)
- **huge**: ~2.1 GB (matriz 16384×16384)
- **extreme**: **~8.6 GB** (matriz 32768×32768)

#### Molecular Dynamics (c_md, c_md_fine, c_md_coarse)
- **small**: ~48 MB (2M partículas, 500 steps)
- **medium**: ~192 MB (8M partículas, 1000 steps)
- **large**: ~768 MB (32M partículas, 2000 steps)
- **huge**: ~3 GB (128M partículas, 5000 steps)
- **extreme**: **~12 GB** (512M partículas, 10000 steps)

#### Graph Search (c_testPath, c_testPath_fine, c_testPath_coarse)
- Todos os tamanhos: **~1-5 MB** (depende do grafo carregado)
- Workload varia pela complexidade do grafo, não pelo tamanho em memória

### Metodologia de Cálculo de Memória

#### **Cálculo de Pi (Monte Carlo Integration)**
```
Memória = num_steps × sizeof(double)
- small:   500,000 × 8 bytes = 4 MB × 2 (arrays intermediários) ≈ 8 MB
- medium:  2,000,000 × 8 = 16 MB × 2 ≈ 32 MB
- large:   8,000,000 × 8 = 64 MB × 2 ≈ 128 MB
- huge:    32,000,000 × 8 = 256 MB × 2 ≈ 512 MB
- extreme: 128,000,000 × 8 = 1024 MB × 2 ≈ 2 GB
```

#### **Mandelbrot Set**
```
Memória = width × height × sizeof(int) + buffers
- small:   1024² × 4 bytes = 4 MB + overhead ≈ 8 MB
- medium:  2048² × 4 = 16 MB + overhead ≈ 32 MB
- large:   4096² × 4 = 64 MB + overhead ≈ 128 MB
- huge:    8192² × 4 = 256 MB + overhead ≈ 512 MB
- extreme: 16384² × 4 = 1024 MB + overhead ≈ 2 GB

Nota: overhead inclui buffers de iteração e dados intermediários
```

#### **QuickSort**
```
Memória = n_elements × sizeof(double) + stack_recursion
- small:   500,000 × 8 = 4 MB
- medium:  2,000,000 × 8 = 16 MB
- large:   8,000,000 × 8 = 64 MB
- huge:    32,000,000 × 8 = 256 MB
- extreme: 128,000,000 × 8 = 1024 MB (1 GB)

Nota: pilha de recursão adiciona ~10-20% ao uso de memória
```

#### **FFT (Fast Fourier Transform)**
```
Memória = n_points × sizeof(complex) × 2 (input + output)
- small:   4,096 × 16 bytes × 2 = 128 KB
- medium:  16,384 × 16 × 2 = 512 KB
- large:   65,536 × 16 × 2 = 2 MB
- huge:    262,144 × 16 × 2 = 8 MB
- extreme: 1,048,576 × 16 × 2 = 32 MB

sizeof(complex) = 2 × sizeof(double) = 16 bytes (parte real + imaginária)
Nota: FFT usa menos memória mas é intensivo em processamento
```

#### **Jacobi Iterative Solver**
```
Memória = grid_size² × sizeof(double) × 2 (matriz atual + próxima iteração)
- small:   1024² × 8 × 2 = 16 MB
- medium:  2048² × 8 × 2 = 64 MB
- large:   4096² × 8 × 2 = 256 MB
- huge:    8192² × 8 × 2 = 1024 MB (1 GB)
- extreme: 16384² × 8 × 2 = 4096 MB (4 GB)

Considerando buffers e sincronização: reduzido para ~50% = 2 GB reportado
```

#### **LU Decomposition**
```
Memória = N × N × sizeof(double) × 3 (matriz A, L, U)
- small:   1024² × 8 × 3 = 24 MB
- medium:  2048² × 8 × 3 = 96 MB
- large:   4096² × 8 × 3 = 384 MB
- huge:    8192² × 8 × 3 = 1536 MB
- extreme: 16384² × 8 × 3 = 6144 MB

In-place optimization reduz para ~33% = 2 GB reportado
```

#### **Molecular Dynamics**
```
Memória = n_particles × (3 × sizeof(double)) × 3 (posição, velocidade, força)
- small:   500,000 × 24 × 3 = 36 MB
- medium:  2,000,000 × 24 × 3 = 144 MB
- large:   8,000,000 × 24 × 3 = 576 MB
- huge:    32,000,000 × 24 × 3 = 2304 MB
- extreme: 128,000,000 × 24 × 3 = 9216 MB

Neighbor lists e spatial partitioning otimizam para ~30% = 3 GB reportado
```

#### **Graph Search (BFS/DFS)**
```
Memória depende da estrutura do grafo carregado, não do tamanho configurado:
- Adjacency list: O(V + E) onde V = vértices, E = arestas
- Visited array: V × sizeof(bool)
- Queue/Stack: O(V) no pior caso

Grafos típicos: 10K-100K vértices = 1-5 MB
Workload varia pela complexidade topológica, não pelo uso de memória
```

### Recomendações de Uso

**Para testes rápidos:**
```bash
python benchmark_runner.py --sizes small,medium
```

**Para análise de escalabilidade:**
```bash
python benchmark_runner.py --sizes small,medium,large,huge
```

**Para estressar o sistema (workload extremo):**
```bash
python benchmark_runner.py --sizes extreme --threads 1,8,16,24,32
```

**ATENÇÃO**: O tamanho **extreme** pode levar **vários minutos a horas** por execução e requer:
- **Mínimo**: 16 GB de RAM
- **Recomendado**: 32-64 GB de RAM
- **Ideal**: 128 GB de RAM (i9-14900K)
- **CPU**: 16+ cores para aproveitar paralelização

## 📊 Configuração de Threads

Todas as aplicações suportam os seguintes números de threads:
**1, 2, 4, 8, 16, 24, 32**

Configure via variável de ambiente:
```bash
export OMP_NUM_THREADS=8
./bin/c_pi.par.gnu -test
```

## 🗂️ Estrutura do Projeto

```
src/
├── applications/          # Código fonte das aplicações
│   ├── c_Pi/             # Pi calculation
│   ├── c_Mandelbrot/     # Mandelbrot set
│   ├── c_QuickSort/      # Parallel quicksort
│   ├── c_FFT/            # Fast Fourier Transform
│   ├── c_Jacobi/         # Jacobi solver
│   ├── c_LUreduction/    # LU decomposition
│   ├── c_MolecularDynamic/ # Molecular dynamics
│   └── c_GraphSearch/    # Graph path search
├── bin/                  # Executáveis compilados
├── common/               # Código compartilhado (OmpSCR)
├── config/               # Configurações de compilação
├── include/              # Headers
├── benchmark_runner.py   # Script principal de benchmarking
├── requirements.txt      # Dependências Python
└── README.md            # Este arquivo
```

## 🔨 Compilação

### Requisitos
- GCC com suporte OpenMP
- Make/GMake
- Python 3.6+ (para benchmarks)

### Opções de Compilação
```bash
# Compilar versão paralela (.par.gnu)
make

# Compilar versão sequencial (.seq.gnu)
make seq

# Compilar com debug
make DEBUG=yes

# Limpar compilação
make clean
```

### Compilação Individual
```bash
cd applications/c_Pi
make                    # Compila c_pi, c_pi_fine, c_pi_coarse
```

## 📈 Análise de Resultados

### Formato CSV
```csv
timestamp,application,threads,size,execution_time,speedup,efficiency
2024-11-24 10:30:00,c_pi,4,medium,2.34,3.42,0.855
```

### Métricas Calculadas
- **Execution Time**: Tempo de execução em segundos
- **Speedup**: Tempo(1 thread) / Tempo(N threads)
- **Efficiency**: Speedup / N threads

### Análise Comparativa
Para comparar variantes de granularidade:
```bash
python benchmark_runner.py \
  --applications c_pi,c_pi_fine,c_pi_coarse \
  --threads 1,2,4,8,16 \
  --sizes medium,large
```

Analise os resultados comparando:
1. Tempos de execução absolutos
2. Speedup relativo ao sequencial
3. Eficiência paralela
4. Escalabilidade com aumento de threads

## 🎯 Detalhes de Implementação

### Pi Calculation
- **Fine**: `schedule(dynamic, 1)` - um chunk por iteração
- **Coarse**: `schedule(static, N/(threads*4))` - chunks grandes pré-calculados

### Mandelbrot
- **Fine**: `schedule(dynamic, 10)` - balanceamento para workload irregular
- **Coarse**: `schedule(static, NPOINTS/threads)` - divisão estática

### QuickSort
- **Fine**: Task cutoff 1000 elementos - paralelização profunda
- **Coarse**: Task cutoff 100000 elementos - paralelização limitada ao topo

### FFT
- **Fine**: Cutoff 64 + `schedule(dynamic, 8)` - nested parallelism
- **Coarse**: Cutoff 4096 + `schedule(static, chunk)` - top-level only

### Jacobi
- **Fine**: `schedule(dynamic, 4)` - adapta a convergência irregular
- **Coarse**: `schedule(static, m/threads)` - minimiza overhead

### LU Decomposition
- **Fine**: `schedule(dynamic, 2)` - adapta ao workload decrescente
- **Coarse**: `schedule(static, (size-k)/threads)` - chunks adaptativos

### Molecular Dynamics
- **Fine**: `schedule(dynamic, 8)` - 8 partículas por chunk
- **Coarse**: `schedule(static, np/threads)` - divisão estática de partículas

### Graph Search
- **Fine**: 1 nó por acesso ao pool - máximo balanceamento
- **Coarse**: Batches de 10 nós - reduz critical sections

## 🧪 Testes

### Teste Rápido
```bash
# Executar aplicação individual
export OMP_NUM_THREADS=4
./bin/c_pi.par.gnu -test

# Comparar variantes
for app in c_pi c_pi_fine c_pi_coarse; do
    echo "Testing $app"
    ./bin/${app}.par.gnu -test
done
```

### Verificação de Integridade
```bash
# Compilar e testar todas as variantes
for dir in applications/c_*; do
    echo "Building $(basename $dir)"
    make -C $dir
done
```

## 📝 Benchmark Runner - Opções Avançadas

### Sintaxe Completa
```bash
python benchmark_runner.py \
  --applications APP1,APP2,... \
  --threads T1,T2,... \
  --sizes SIZE1,SIZE2,... \
  --repetitions N \
  --output-dir DIR \
  --timeout SECONDS
```

### Opções Disponíveis
- `--applications`: Lista de aplicações (padrão: todas)
- `--threads`: Lista de threads (padrão: 1,2,4,8,16,24)
- `--sizes`: tiny, small, medium, large, huge, extreme, massive, colossal, gigantic
- `--repetitions`: Número de repetições por teste (padrão: 3)
- `--output-dir`: Diretório para resultados (padrão: benchmark_results)
- `--timeout`: Timeout por teste em segundos (padrão: 300)

### Exemplos Práticos

**Teste de Escalabilidade:**
```bash
python benchmark_runner.py \
  --applications c_pi_fine \
  --threads 1,2,4,8,16,24,32 \
  --sizes large \
  --repetitions 5
```

**Comparação de Granularidade:**
```bash
python benchmark_runner.py \
  --applications c_mandel,c_mandel_fine,c_mandel_coarse \
  --threads 8 \
  --sizes medium,large,huge \
  --repetitions 10
```

**Benchmark Completo (todas aplicações):**
```bash
python benchmark_runner.py \
  --threads 1,2,4,8,16,24,32 \
  --sizes tiny,small,medium,large \
  --repetitions 3
```

## 🐛 Troubleshooting

### Erro de Compilação
```bash
# Limpar e recompilar
make clean
make all
```

### Timeout em Benchmarks
```bash
# Aumentar timeout
python benchmark_runner.py --timeout 600
```

### Número de Threads não funciona
```bash
# Verificar limite do sistema
echo $OMP_NUM_THREADS
ulimit -u

# Forçar número de threads
export OMP_NUM_THREADS=8
```

## 📚 Referências

- OpenMP Specification: https://www.openmp.org/specifications/
- OpenMP Source Code Repository: http://www.pcg.ull.es/ompscr/

## 📄 Licença

Este projeto segue as licenças dos arquivos originais:
- Arquivos OmpSCR: Copyright (c) 2004, OmpSCR Group
- Variantes de granularidade: Implementadas em 2024

Veja arquivo LICENSE para detalhes.

## 🤝 Contribuindo

Para adicionar novas aplicações ou variantes:

1. Crie os arquivos fonte em `applications/`
2. Atualize o `GNUmakefile` da aplicação
3. Adicione entrada em `benchmark_runner.py`
4. Compile e teste: `make -C applications/sua_app`
5. Execute benchmark: `python benchmark_runner.py --applications sua_app`

## 📞 Suporte

Para questões sobre:
- **Implementações originais**: ompscr@etsii.ull.es
- **Variantes de granularidade**: Veja comentários nos arquivos fonte
