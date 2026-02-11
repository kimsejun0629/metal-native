#!/bin/bash
# MetalNative benchmark runner
# Usage: ./scripts/benchmark.sh [--quick|--full]

set -e  # Exit on error

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

BENCHMARK_MODE="standard"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            BENCHMARK_MODE="quick"
            shift
            ;;
        --full)
            BENCHMARK_MODE="full"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick|--full]"
            exit 1
            ;;
    esac
done

echo "==================================="
echo "MetalNative Benchmark Suite"
echo "==================================="
echo "Mode: ${BENCHMARK_MODE}"
echo ""

# Check build
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found: ${BUILD_DIR}"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

# Check if package is installed
if ! python -c "import metal_native" 2>/dev/null; then
    echo "Error: metal_native Python package not installed"
    echo "Install with: pip install -e ."
    exit 1
fi

# Run benchmarks based on mode
cd "${PROJECT_ROOT}"

if [ "${BENCHMARK_MODE}" == "quick" ]; then
    echo "Running quick benchmark suite..."
    python benchmarks/run_all.py \
        --iterations 5 \
        --warmup 2 \
        --matmul-sizes 128 256 512 \
        --attention-seq-lens 128 256 \
        --skip-transformer \
        --output benchmark_results_quick.md
elif [ "${BENCHMARK_MODE}" == "full" ]; then
    echo "Running full benchmark suite..."
    python benchmarks/run_all.py \
        --iterations 20 \
        --warmup 5 \
        --matmul-sizes 128 256 512 1024 2048 4096 \
        --attention-seq-lens 128 256 512 1024 2048 \
        --conv-preset all \
        --transformer-models all \
        --output benchmark_results_full.md
else
    echo "Running standard benchmark suite..."
    python benchmarks/run_all.py \
        --iterations 10 \
        --warmup 3 \
        --output benchmark_results.md
fi

BENCHMARK_STATUS=$?

echo ""
echo "==================================="
echo "Benchmark Summary"
echo "==================================="

if [ ${BENCHMARK_STATUS} -eq 0 ]; then
    echo "Benchmarks completed successfully ✓"
    echo ""
    if [ "${BENCHMARK_MODE}" == "quick" ]; then
        echo "Results saved to: benchmark_results_quick.md"
    elif [ "${BENCHMARK_MODE}" == "full" ]; then
        echo "Results saved to: benchmark_results_full.md"
    else
        echo "Results saved to: benchmark_results.md"
    fi
    exit 0
else
    echo "Benchmarks failed ✗"
    exit 1
fi
