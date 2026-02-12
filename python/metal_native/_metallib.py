"""Runtime discovery of Metal shader library."""

import os
from pathlib import Path
from typing import Optional


def find_metallib() -> Optional[str]:
    """Find metal_native.metallib.

    Search order:
    1. METAL_NATIVE_METALLIB environment variable
    2. Package-internal shaders/ directory (wheel install)
    3. build/ directory (development mode)

    Returns:
        Absolute path to metallib file, or None if not found.
    """
    env_path = os.environ.get("METAL_NATIVE_METALLIB")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    package_dir = Path(__file__).parent
    wheel_path = package_dir / "shaders" / "metal_native.metallib"
    if wheel_path.is_file():
        return str(wheel_path.absolute())

    # Development mode: python/metal_native/ -> ../../build/
    repo_root = package_dir.parent.parent
    for build_dir in ["build", "build/Release", "build/Debug"]:
        dev_path = repo_root / build_dir / "shaders" / "metal_native.metallib"
        if dev_path.is_file():
            return str(dev_path.absolute())

    return None


def get_metallib_path() -> str:
    """Find metal_native.metallib or raise an error.

    Returns:
        Absolute path to metallib file.

    Raises:
        RuntimeError: If metallib cannot be found.
    """
    path = find_metallib()
    if path is None:
        raise RuntimeError(
            "Cannot find metal_native.metallib. "
            "Try: pip install --force-reinstall metal-native"
        )
    return path
