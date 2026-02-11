#!/bin/bash
# MetalNative test runner
# Usage: ./scripts/test.sh [--cpp-only|--python-only]

set -e  # Exit on error

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

RUN_CPP=1
RUN_PYTHON=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only)
            RUN_PYTHON=0
            shift
            ;;
        --python-only)
            RUN_CPP=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cpp-only|--python-only]"
            exit 1
            ;;
    esac
done

CPP_STATUS=0
PYTHON_STATUS=0

echo "==================================="
echo "MetalNative Test Runner"
echo "==================================="
echo ""

# Run C++ tests
if [ ${RUN_CPP} -eq 1 ]; then
    echo "==================================="
    echo "Running C++ tests..."
    echo "==================================="

    if [ ! -d "${BUILD_DIR}" ]; then
        echo "Error: Build directory not found: ${BUILD_DIR}"
        echo "Run ./scripts/build.sh first"
        exit 1
    fi

    cd "${BUILD_DIR}"
    if ! ctest --output-on-failure --parallel $(sysctl -n hw.ncpu); then
        CPP_STATUS=1
        echo "C++ tests failed!"
    else
        echo "C++ tests passed!"
    fi
    cd "${PROJECT_ROOT}"
    echo ""
fi

# Run Python tests
if [ ${RUN_PYTHON} -eq 1 ]; then
    echo "==================================="
    echo "Running Python tests..."
    echo "==================================="

    # Check if pytest is available
    if ! command -v pytest >/dev/null 2>&1; then
        echo "Warning: pytest not found. Install with: pip install pytest"
        PYTHON_STATUS=1
    else
        # Check if package is installed
        if ! python -c "import metal_native" 2>/dev/null; then
            echo "Warning: metal_native Python package not installed"
            echo "Install with: pip install -e ."
            PYTHON_STATUS=1
        else
            if ! pytest "${PROJECT_ROOT}/tests/python/" -v; then
                PYTHON_STATUS=1
                echo "Python tests failed!"
            else
                echo "Python tests passed!"
            fi
        fi
    fi
    echo ""
fi

# Summary
echo "==================================="
echo "Test Summary"
echo "==================================="

if [ ${RUN_CPP} -eq 1 ]; then
    if [ ${CPP_STATUS} -eq 0 ]; then
        echo "C++ tests:    PASSED ✓"
    else
        echo "C++ tests:    FAILED ✗"
    fi
fi

if [ ${RUN_PYTHON} -eq 1 ]; then
    if [ ${PYTHON_STATUS} -eq 0 ]; then
        echo "Python tests: PASSED ✓"
    else
        echo "Python tests: FAILED ✗"
    fi
fi

echo "==================================="

# Exit with combined status
COMBINED_STATUS=$((CPP_STATUS + PYTHON_STATUS))
if [ ${COMBINED_STATUS} -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi
