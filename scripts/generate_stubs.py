#!/usr/bin/env python3
"""Generate .pyi type stub files from pybind11 bindings.

This script validates that the metal_native._C module can be imported
and provides basic type information. Since pybind11 bindings don't
expose full introspection, we primarily validate the module structure.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def validate_module_import() -> bool:
    """Validate that metal_native._C can be imported.

    Returns:
        True if import successful, False otherwise
    """
    try:
        import metal_native._C as _C
        print("✓ Successfully imported metal_native._C")
        return True
    except ImportError as e:
        print(f"✗ Failed to import metal_native._C: {e}", file=sys.stderr)
        return False


def check_stub_exists(package_path: Path) -> Optional[Path]:
    """Check if _C.pyi stub file exists.

    Args:
        package_path: Path to metal_native package

    Returns:
        Path to stub file if it exists, None otherwise
    """
    stub_path = package_path / "_C.pyi"
    if stub_path.exists():
        print(f"✓ Found existing stub file: {stub_path}")
        return stub_path
    else:
        print(f"ℹ No stub file found at: {stub_path}")
        return None


def validate_stub_content(stub_path: Path) -> bool:
    """Validate that stub file has basic required content.

    Args:
        stub_path: Path to stub file

    Returns:
        True if validation passes, False otherwise
    """
    try:
        content = stub_path.read_text()

        # Check for basic markers of a valid stub
        required_patterns = [
            "def ",  # At least one function
            "class ",  # At least one class
        ]

        missing = []
        for pattern in required_patterns:
            if pattern not in content:
                missing.append(pattern)

        if missing:
            print(f"⚠ Stub file may be incomplete. Missing patterns: {missing}")
            return False

        print(f"✓ Stub file appears valid ({len(content)} bytes)")
        return True

    except Exception as e:
        print(f"✗ Error reading stub file: {e}", file=sys.stderr)
        return False


def get_module_info():
    """Print information about the _C module."""
    try:
        import metal_native._C as _C

        print("\nModule Information:")
        print(f"  Module: {_C.__name__}")

        # List available attributes
        attrs = [name for name in dir(_C) if not name.startswith('_')]
        print(f"  Public attributes: {len(attrs)}")

        if attrs:
            print("\n  Available exports:")
            for attr in sorted(attrs):
                obj = getattr(_C, attr)
                obj_type = type(obj).__name__
                print(f"    - {attr}: {obj_type}")

        return True

    except Exception as e:
        print(f"✗ Error getting module info: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate/validate .pyi type stubs for metal_native._C',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--package-dir', type=str,
                       help='Path to metal_native package directory')
    parser.add_argument('--info', action='store_true',
                       help='Print module information')

    args = parser.parse_args()

    print("===================================")
    print("MetalNative Stub Generator/Validator")
    print("==================================="))
    print()

    # Validate module can be imported
    if not validate_module_import():
        print("\nError: Cannot import metal_native._C")
        print("Make sure the package is built and installed:")
        print("  1. Build: ./scripts/build.sh")
        print("  2. Install: pip install -e .")
        sys.exit(1)

    # Show module info if requested
    if args.info:
        if not get_module_info():
            sys.exit(1)

    # Find package directory
    if args.package_dir:
        package_path = Path(args.package_dir)
    else:
        try:
            import metal_native
            package_path = Path(metal_native.__file__).parent
        except Exception as e:
            print(f"Error: Cannot locate metal_native package: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nPackage location: {package_path}")

    # Check for existing stub
    stub_path = check_stub_exists(package_path)

    if stub_path:
        # Validate existing stub
        if validate_stub_content(stub_path):
            print("\n✓ Stub validation passed")
            sys.exit(0)
        else:
            print("\n⚠ Stub validation failed")
            print("Consider regenerating stubs with pybind11-stubgen:")
            print("  pip install pybind11-stubgen")
            print("  pybind11-stubgen metal_native._C")
            sys.exit(1)
    else:
        print("\nℹ No stub file found. To generate stubs:")
        print("  pip install pybind11-stubgen")
        print("  pybind11-stubgen metal_native._C -o python/")
        print("  # Then move generated stubs to python/metal_native/")
        sys.exit(0)


if __name__ == "__main__":
    main()
