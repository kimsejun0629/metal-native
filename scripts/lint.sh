#!/bin/bash
# MetalNative linting script
# Usage: ./scripts/lint.sh [--fix]

set -e  # Exit on error

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FIX_MODE=0

# Parse arguments
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=1
fi

CLANG_FORMAT_STATUS=0
RUFF_STATUS=0

echo "==================================="
echo "MetalNative Linting"
echo "==================================="
echo ""

# Check for clang-format
if ! command -v clang-format >/dev/null 2>&1; then
    echo "Warning: clang-format not found. Install with: brew install clang-format"
    CLANG_FORMAT_STATUS=1
else
    echo "==================================="
    echo "Running clang-format on C++/ObjC++ files..."
    echo "==================================="

    # Find all C++/ObjC++ files
    CPP_FILES=$(find "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/tests" \
        -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.mm" \) 2>/dev/null || true)

    if [ -z "${CPP_FILES}" ]; then
        echo "No C++ files found"
    else
        if [ ${FIX_MODE} -eq 1 ]; then
            echo "Fixing formatting..."
            echo "${CPP_FILES}" | xargs clang-format -i
            echo "Formatting applied!"
        else
            echo "Checking formatting (dry run)..."
            if ! echo "${CPP_FILES}" | xargs clang-format --dry-run --Werror; then
                CLANG_FORMAT_STATUS=1
                echo ""
                echo "Formatting violations found!"
                echo "Run './scripts/lint.sh --fix' to auto-fix"
            else
                echo "All C++ files properly formatted!"
            fi
        fi
    fi
    echo ""
fi

# Check for ruff
if ! command -v ruff >/dev/null 2>&1; then
    echo "Warning: ruff not found. Install with: pip install ruff"
    RUFF_STATUS=1
else
    echo "==================================="
    echo "Running ruff on Python files..."
    echo "==================================="

    # Find Python directories
    PYTHON_DIRS="${PROJECT_ROOT}/python ${PROJECT_ROOT}/tests/python ${PROJECT_ROOT}/benchmarks"

    if [ ${FIX_MODE} -eq 1 ]; then
        echo "Fixing Python code..."
        ruff check --fix ${PYTHON_DIRS}
        ruff format ${PYTHON_DIRS}
        echo "Python code fixed!"
    else
        echo "Checking Python code..."
        if ! ruff check ${PYTHON_DIRS}; then
            RUFF_STATUS=1
            echo ""
            echo "Ruff violations found!"
            echo "Run './scripts/lint.sh --fix' to auto-fix"
        else
            echo "All Python files pass ruff check!"
        fi
    fi
    echo ""
fi

# Summary
echo "==================================="
echo "Lint Summary"
echo "==================================="

if [ ${CLANG_FORMAT_STATUS} -eq 0 ]; then
    echo "clang-format: PASSED ✓"
else
    echo "clang-format: FAILED ✗"
fi

if [ ${RUFF_STATUS} -eq 0 ]; then
    echo "ruff:         PASSED ✓"
else
    echo "ruff:         FAILED ✗"
fi

echo "==================================="

# Exit with combined status
COMBINED_STATUS=$((CLANG_FORMAT_STATUS + RUFF_STATUS))
if [ ${COMBINED_STATUS} -eq 0 ]; then
    echo "All lint checks passed!"
    exit 0
else
    echo "Some lint checks failed!"
    exit 1
fi
