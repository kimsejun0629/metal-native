"""Setup script for metal_native Python package.

This setup.py uses CMake to build the C++ extension and pybind11 bindings.
It leverages the existing CMakeLists.txt infrastructure.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Read version from _version.py
version_file = Path(__file__).parent / "metal_native" / "_version.py"
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)
__version__ = version_dict["__version__"]


class CMakeExtension(Extension):
    """Extension that is built using CMake."""

    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command that uses CMake."""

    def build_extension(self, ext: CMakeExtension):
        """Build the extension using CMake.

        Args:
            ext: Extension to build
        """
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_BENCHMARKS=OFF",
        ]

        # Build type
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # Platform-specific configuration
        if sys.platform == "darwin":
            # Ensure we're building for the current architecture
            cmake_args.extend([
                f"-DCMAKE_BUILD_TYPE={cfg}",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0",  # macOS Sonoma minimum
            ])
            # Use all available cores
            build_args.extend(["--", "-j"])
        else:
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={cfg}")
            build_args.extend(["--", "-j"])

        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake configure
        print(f"Configuring CMake in {build_temp}")
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
        )

        # Run CMake build
        print(f"Building with CMake")
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )


# Read long description from README
readme_file = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()


setup(
    name="metal-native",
    version=__version__,
    author="MetalNative Contributors",
    author_email="",
    description="High-performance deep learning on Apple Silicon GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kimsejun/metal-native",
    project_urls={
        "Bug Tracker": "https://github.com/kimsejun/metal-native/issues",
        "Documentation": "https://metal-native.readthedocs.io",
        "Source Code": "https://github.com/kimsejun/metal-native",
    },
    packages=find_packages(),
    ext_modules=[
        CMakeExtension("metal_native._C", sourcedir=str(Path(__file__).parent.parent))
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "torch": ["torch>=2.2"],
        "accelerate": ["accelerate>=0.28"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.2",
            "sphinx-autodoc-typehints>=1.22",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="metal mps apple-silicon gpu deep-learning pytorch numpy",
    platforms=["macOS >= 13.0"],
)
