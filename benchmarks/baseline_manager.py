"""Manage benchmark baselines with versioning."""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class BaselineManager:
    """Manages saving, loading, and versioning of performance baselines."""

    def __init__(self, baseline_dir: str):
        """Initialize baseline manager.

        Args:
            baseline_dir: Directory to store baseline files
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, results: List[Dict], label: str = None, metadata: Dict = None) -> str:
        """Save benchmark results as a baseline.

        Args:
            results: List of benchmark result dictionaries
            label: Optional label for the baseline (defaults to timestamp)
            metadata: Optional metadata dictionary

        Returns:
            The label/filename of the saved baseline
        """
        if label is None:
            label = datetime.now().strftime("%Y%m%d_%H%M%S")

        baseline_data = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "git_hash": self._get_git_hash(),
            "git_branch": self._get_git_branch(),
            "metadata": metadata or {},
            "results": results
        }

        baseline_file = self.baseline_dir / f"{label}.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        # Create a symlink to "latest"
        latest_link = self.baseline_dir / "latest.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(baseline_file.name)

        return label

    def load_baseline(self, label: str = "latest") -> Dict:
        """Load a baseline for comparison.

        Args:
            label: Label of the baseline to load (default: "latest")

        Returns:
            Dictionary containing baseline data

        Raises:
            FileNotFoundError: If the baseline doesn't exist
        """
        baseline_file = self.baseline_dir / f"{label}.json"

        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline '{label}' not found at {baseline_file}")

        with open(baseline_file, 'r') as f:
            return json.load(f)

    def list_baselines(self) -> List[Dict]:
        """List available baselines.

        Returns:
            List of dictionaries with baseline info (label, timestamp, git_hash)
        """
        baselines = []
        for baseline_file in sorted(self.baseline_dir.glob("*.json")):
            if baseline_file.name == "latest.json":
                continue

            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    baselines.append({
                        "label": data.get("label", baseline_file.stem),
                        "timestamp": data.get("timestamp", "unknown"),
                        "git_hash": data.get("git_hash", "unknown"),
                        "git_branch": data.get("git_branch", "unknown"),
                        "num_benchmarks": len(data.get("results", []))
                    })
            except (json.JSONDecodeError, KeyError):
                # Skip malformed files
                continue

        return baselines

    def delete_baseline(self, label: str) -> None:
        """Delete a baseline.

        Args:
            label: Label of the baseline to delete

        Raises:
            FileNotFoundError: If the baseline doesn't exist
        """
        baseline_file = self.baseline_dir / f"{label}.json"

        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline '{label}' not found")

        baseline_file.unlink()

    def get_baseline_by_git_hash(self, git_hash: str) -> Optional[Dict]:
        """Find a baseline by git commit hash.

        Args:
            git_hash: Git commit hash to search for

        Returns:
            Baseline data if found, None otherwise
        """
        for baseline_file in self.baseline_dir.glob("*.json"):
            if baseline_file.name == "latest.json":
                continue

            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    if data.get("git_hash", "").startswith(git_hash):
                        return data
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    @staticmethod
    def _get_git_hash() -> str:
        """Get current git commit hash.

        Returns:
            Git commit hash or 'unknown'
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _get_git_branch() -> str:
        """Get current git branch name.

        Returns:
            Git branch name or 'unknown'
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
