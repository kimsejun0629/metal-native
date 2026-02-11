#!/bin/bash
# Build script for MetalNative project

set -e

# Configuration
BUILD_TYPE="${1:-Release}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/${BUILD_TYPE}"
INSTALL_DIR="${PROJECT_ROOT}/install"

# Validate build type
if [[ ! "${BUILD_TYPE}" =~ ^(Debug|Release|RelWithDebInfo|MinSizeRel)$ ]]; then
    echo "Error: Invalid build type '${BUILD_TYPE}'"
    echo "Usage: $0 [Debug|Release|RelWithDebInfo|MinSizeRel]"
    exit 1
fi

echo "======================================"
echo "MetalNative Build Script"
echo "======================================"
echo "Build Type: ${BUILD_TYPE}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Build Directory: ${BUILD_DIR}"
echo "Install Directory: ${INSTALL_DIR}"
echo "======================================"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo ""
echo "Configuring CMake..."
cmake \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DBUILD_PYTHON=ON \
    "${PROJECT_ROOT}"

# Build
echo ""
echo "Building..."
cmake --build . --config "${BUILD_TYPE}" --parallel $(sysctl -n hw.ncpu)

# Copy compile_commands.json to project root for IDE support
if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    cp "${BUILD_DIR}/compile_commands.json" "${PROJECT_ROOT}/"
    echo ""
    echo "Copied compile_commands.json to project root"
fi

echo ""
echo "======================================"
echo "Build completed successfully!"
echo "======================================"
echo ""
echo "To install, run:"
echo "  cmake --install ${BUILD_DIR}"
echo ""
echo "To run tests, run:"
echo "  cd ${BUILD_DIR} && ctest"
echo ""
