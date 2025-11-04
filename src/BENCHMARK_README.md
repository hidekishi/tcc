# OpenMP Benchmark Suite Automation

This repository contains automated tools for running the OmpSCR v2.0 benchmark suite on Linux Mint with different thread configurations and analyzing the results.

## Files Overview

- `benchmark_runner.py` - Main script that handles dependencies, compilation, and benchmark execution
- `analyze_results.py` - Script for analyzing results and generating visualizations
- `setup_and_run.sh` - Quick setup and run script for Linux systems
- `requirements.txt` - Python package requirements for analysis tools

## Quick Start

### Method 1: Automated Setup (Recommended)

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

This script will:
1. Install all system dependencies
2. Set up the benchmark suite
3. Optionally run the benchmarks immediately

### Method 2: Manual Setup

1. **Install system dependencies:**
```bash
sudo apt update
sudo apt install -y build-essential gcc g++ gfortran libomp-dev make git time
```

2. **Run the benchmark suite:**
```bash
python3 benchmark_runner.py
```

## Benchmark Configuration

### Default Configuration
- **Thread counts**: 1, 2, 4, 8, 16, 32
- **Iterations**: 10 per configuration
- **Applications**: All available C benchmarks
- **Output**: `./benchmark_results/` directory

### Custom Configurations

#### Custom Thread Counts
```bash
python3 benchmark_runner.py --threads 1,2,4,8
```

#### Fewer Iterations (for testing)
```bash
python3 benchmark_runner.py --iterations 3
```

#### Specific Applications Only
```bash
python3 benchmark_runner.py --apps c_Pi,c_Mandelbrot,c_Jacobi
```

#### Custom Output Directory
```bash
python3 benchmark_runner.py --output /path/to/results
```

### Available Applications

The benchmark suite includes these OpenMP applications:
- `c_FFT` - Fast Fourier Transform
- `c_FFT6` - 6-point FFT 
- `c_Jacobi` - Jacobi iterative solver
- `c_LoopsWithDependencies` - Loop dependency examples
- `c_LUreduction` - LU decomposition
- `c_Mandelbrot` - Mandelbrot set generator
- `c_MolecularDynamic` - Molecular dynamics simulation
- `c_Pi` - Pi calculation using numerical integration
- `c_QuickSort` - Parallel quicksort

## Results Analysis

### Automatic Analysis
After running benchmarks, analyze the results:

```bash
# Install analysis dependencies (optional, for visualizations)
pip3 install pandas matplotlib seaborn

# Analyze results
python3 analyze_results.py benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.csv
```

### Analysis Outputs
The analysis generates:
- **Performance plots** - Execution time vs thread count
- **Speedup analysis** - Speedup and parallel efficiency graphs
- **Memory usage plots** - Memory consumption analysis
- **Performance heatmap** - Overview of all results
- **Detailed report** - Text summary with recommendations

### Sample Analysis Commands
```bash
# Analyze specific results file
python3 analyze_results.py results.csv --output ./my_analysis

# Analyze JSON results
python3 analyze_results.py results.json
```

## Understanding Results

### Output Files
Each benchmark run creates several files:

1. **CSV Results** (`benchmark_results_YYYYMMDD_HHMMSS.csv`)
   - Raw timing data for all runs
   - Columns: application, threads, iteration, wall_time, cpu_time, etc.

2. **JSON Results** (`benchmark_results_YYYYMMDD_HHMMSS.json`)
   - Same data in JSON format for programmatic analysis

3. **Summary Report** (`benchmark_summary_YYYYMMDD_HHMMSS.txt`)
   - High-level overview of benchmark results
   - Success rates and average performance

4. **Log File** (`benchmark_run_YYYYMMDD_HHMMSS.log`)
   - Detailed execution log with any errors

### Key Metrics

- **Wall Time**: Total elapsed time for execution
- **CPU Time**: Total CPU time used (user + system)
- **Speedup**: Performance improvement over single thread
- **Efficiency**: Speedup divided by number of threads
- **Memory Usage**: Peak memory consumption

### Performance Analysis

The analysis tools help identify:
- **Optimal thread counts** for each application
- **Scaling behavior** as thread count increases  
- **Efficiency degradation** due to overhead
- **Memory usage patterns**

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```
   Error: Build process failed
   ```
   - Check that GCC and OpenMP are properly installed
   - Verify `gcc -fopenmp` works
   - Check log files in `./log/` directory

2. **Permission Errors**
   ```
   Error: Permission denied
   ```
   - Ensure scripts are executable: `chmod +x *.sh *.py`
   - Check directory write permissions

3. **Missing Dependencies**
   ```
   Error: gcc: command not found
   ```
   - Run the setup script: `./setup_and_run.sh`
   - Or install manually: `sudo apt install build-essential`

4. **OpenMP Issues**
   ```
   Error: libgomp not found
   ```
   - Install OpenMP library: `sudo apt install libomp-dev`
   - For GCC: `sudo apt install libgomp1`

### Performance Issues

- **Long execution times**: Reduce iterations or thread counts for testing
- **Memory errors**: Check available RAM, reduce problem sizes
- **Inconsistent results**: Ensure system is not under load during benchmarks

### Getting Help

1. **Check logs**: Look in `benchmark_results/*.log` for detailed error messages
2. **Verbose output**: The benchmark runner provides detailed logging
3. **Test individual apps**: Use `--apps` flag to test specific applications
4. **Minimal test**: Run with `--iterations 1 --threads 1,2` for quick testing

## Advanced Usage

### Batch Processing
```bash
# Run multiple configurations
for threads in "1,2" "1,4" "1,8"; do
    python3 benchmark_runner.py --threads $threads --output results_$threads
done
```

### Custom Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('benchmark_results.csv')

# Custom analysis
successful_runs = df[df['exit_code'] == 0]
avg_times = successful_runs.groupby(['application', 'threads'])['wall_time'].mean()

# Create custom plots
plt.figure(figsize=(10, 6))
for app in successful_runs['application'].unique():
    app_data = avg_times[app]
    plt.plot(app_data.index, app_data.values, label=app, marker='o')

plt.xlabel('Thread Count')
plt.ylabel('Average Wall Time (s)')
plt.legend()
plt.savefig('custom_analysis.png')
```

### Integration with Other Tools

The results can be exported for use with:
- **R** - Load CSV files with `read.csv()`
- **MATLAB** - Use `readtable()` function
- **Excel** - Direct CSV import
- **Jupyter notebooks** - Use pandas for analysis

## System Requirements

### Minimum Requirements
- **OS**: Linux Mint 19+ or Ubuntu 18.04+
- **RAM**: 2GB (4GB+ recommended)
- **CPU**: Multi-core processor
- **Storage**: 500MB for build artifacts
- **Python**: 3.6+

### Recommended Setup
- **RAM**: 8GB+ for larger benchmarks
- **CPU**: 8+ cores to see scaling effects
- **Storage**: 1GB+ for multiple result sets
- **Python**: 3.8+ with pip

## Contributing

This benchmark automation suite can be extended:

1. **New applications**: Add to the `applications` list in `benchmark_runner.py`
2. **Additional metrics**: Extend result collection in `run_single_benchmark()`
3. **New analysis**: Add visualization functions to `analyze_results.py`
4. **Platform support**: Adapt dependency installation for other Linux distributions

## License

This automation suite is provided under the same license as the original OmpSCR benchmark suite. The original benchmark code is included under its respective licenses.

## References

- [Original OmpSCR Benchmark Suite](https://sourceforge.net/projects/ompscr/files/OmpSCR/OmpSCR%20Full%20Distribution%20v2.0/)
- [OpenMP Specification](https://www.openmp.org/specifications/)
- [GCC OpenMP Documentation](https://gcc.gnu.org/onlinedocs/libgomp/)