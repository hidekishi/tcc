#!/bin/bash

# OpenMP Benchmark Suite Setup Script for Linux Mint
# This script prepares the system and runs the benchmark suite

set -e  # Exit on any error

echo "======================================================"
echo "OpenMP Benchmark Suite Setup and Runner"
echo "======================================================"
echo

# Check if running on a Debian/Ubuntu-based system
if ! command -v apt &> /dev/null; then
    echo "Error: This script is designed for Debian/Ubuntu-based systems (like Linux Mint)"
    echo "Please install dependencies manually or use the Python script directly."
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3..."
    sudo apt update
    sudo apt install -y python3 python3-pip
fi

# Make the Python script executable
chmod +x benchmark_runner.py

echo "Setup complete!"
echo
echo "You can now run the benchmark suite in several ways:"
echo
echo "1. Run with default settings (recommended):"
echo "   python3 benchmark_runner.py"
echo
echo "2. Run with custom thread configurations:"
echo "   python3 benchmark_runner.py --threads 1,2,4,8"
echo
echo "3. Run with fewer iterations (for testing):"
echo "   python3 benchmark_runner.py --iterations 3"
echo
echo "4. Run specific applications only:"
echo "   python3 benchmark_runner.py --apps c_Pi,c_Mandelbrot"
echo
echo "5. View all options:"
echo "   python3 benchmark_runner.py --help"
echo
echo "Results will be saved in the 'benchmark_results' directory."
echo

# Ask if user wants to run now
read -p "Would you like to run the benchmark suite now with default settings? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting benchmark suite..."
    python3 benchmark_runner.py
else
    echo "You can run the benchmark suite later using the commands shown above."
fi