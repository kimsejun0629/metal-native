"""Generate HTML and Markdown reports for regression test results."""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class ReportGenerator:
    """Generates formatted reports from regression test results."""

    def __init__(self, report_dir: str = "benchmarks/reports"):
        """Initialize report generator.

        Args:
            report_dir: Directory to save reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        comparisons: List[Dict],
        current_meta: Dict,
        baseline_meta: Dict,
        output_file: Optional[str] = None
    ) -> str:
        """Generate a markdown report from regression results.

        Args:
            comparisons: List of comparison dictionaries
            current_meta: Metadata for current run
            baseline_meta: Metadata for baseline
            output_file: Optional output file path

        Returns:
            Markdown report as string
        """
        lines = [
            "# Performance Regression Test Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Current Git Hash:** `{current_meta.get('git_hash', 'unknown')}`",
            f"- **Current Branch:** `{current_meta.get('git_branch', 'unknown')}`",
            f"- **Baseline:** {baseline_meta.get('label', 'unknown')} (Git: `{baseline_meta.get('git_hash', 'unknown')[:8]}`)",
            f"- **Total Benchmarks:** {len(comparisons)}",
            "",
        ]

        # Count statuses
        regressions = sum(1 for c in comparisons if c["status"] == "REGRESSION")
        improvements = sum(1 for c in comparisons if c["status"] == "IMPROVEMENT")
        stable = sum(1 for c in comparisons if c["status"] == "STABLE")

        lines.extend([
            "### Status Breakdown",
            "",
            f"- ✅ **Stable:** {stable}",
            f"- 🚀 **Improvements:** {improvements}",
            f"- ⚠️ **Regressions:** {regressions}",
            "",
        ])

        if regressions > 0:
            lines.extend([
                "## ⚠️ Regressions Detected",
                "",
                "| Benchmark | Current (ms) | Baseline (ms) | Change | Status |",
                "|-----------|--------------|---------------|--------|--------|",
            ])

            for comp in comparisons:
                if comp["status"] == "REGRESSION":
                    lines.append(self._format_comparison_row(comp))

            lines.append("")

        if improvements > 0:
            lines.extend([
                "## 🚀 Improvements",
                "",
                "| Benchmark | Current (ms) | Baseline (ms) | Change | Status |",
                "|-----------|--------------|---------------|--------|--------|",
            ])

            for comp in comparisons:
                if comp["status"] == "IMPROVEMENT":
                    lines.append(self._format_comparison_row(comp))

            lines.append("")

        # All results table
        lines.extend([
            "## All Results",
            "",
            "| Benchmark | Current (ms) | Baseline (ms) | Change | Status |",
            "|-----------|--------------|---------------|--------|--------|",
        ])

        for comp in comparisons:
            lines.append(self._format_comparison_row(comp))

        lines.append("")

        # Category breakdown
        lines.extend(self._generate_category_breakdown(comparisons))

        report = "\n".join(lines)

        if output_file:
            output_path = self.report_dir / output_file
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def generate_text_report(
        self,
        comparisons: List[Dict],
        current_meta: Dict,
        baseline_meta: Dict
    ) -> str:
        """Generate a plain text report.

        Args:
            comparisons: List of comparison dictionaries
            current_meta: Metadata for current run
            baseline_meta: Metadata for baseline

        Returns:
            Text report as string
        """
        lines = [
            "=" * 80,
            "PERFORMANCE REGRESSION TEST REPORT",
            "=" * 80,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Current Git Hash: {current_meta.get('git_hash', 'unknown')[:12]}",
            f"Baseline: {baseline_meta.get('label', 'unknown')} (Git: {baseline_meta.get('git_hash', 'unknown')[:8]})",
            "",
            "=" * 80,
            "SUMMARY",
            "=" * 80,
            "",
        ]

        # Count statuses
        regressions = sum(1 for c in comparisons if c["status"] == "REGRESSION")
        improvements = sum(1 for c in comparisons if c["status"] == "IMPROVEMENT")
        stable = sum(1 for c in comparisons if c["status"] == "STABLE")

        lines.extend([
            f"Total Benchmarks: {len(comparisons)}",
            f"  - Stable:       {stable}",
            f"  - Improvements: {improvements}",
            f"  - Regressions:  {regressions}",
            "",
        ])

        if regressions > 0:
            lines.extend([
                "=" * 80,
                "REGRESSIONS DETECTED",
                "=" * 80,
                "",
                f"{'Benchmark':<35} {'Current':>12} {'Baseline':>12} {'Change':>10}",
                "-" * 80,
            ])

            for comp in comparisons:
                if comp["status"] == "REGRESSION":
                    lines.append(
                        f"{comp['name']:<35} "
                        f"{comp['current_ms']:>12.3f} "
                        f"{comp['baseline_ms']:>12.3f} "
                        f"{comp['change_pct']:>+9.2f}%"
                    )

            lines.append("")

        if improvements > 0:
            lines.extend([
                "=" * 80,
                "IMPROVEMENTS",
                "=" * 80,
                "",
                f"{'Benchmark':<35} {'Current':>12} {'Baseline':>12} {'Change':>10}",
                "-" * 80,
            ])

            for comp in comparisons:
                if comp["status"] == "IMPROVEMENT":
                    lines.append(
                        f"{comp['name']:<35} "
                        f"{comp['current_ms']:>12.3f} "
                        f"{comp['baseline_ms']:>12.3f} "
                        f"{comp['change_pct']:>+9.2f}%"
                    )

            lines.append("")

        lines.extend([
            "=" * 80,
            "ALL RESULTS",
            "=" * 80,
            "",
            f"{'Benchmark':<35} {'Current':>12} {'Baseline':>12} {'Change':>10} {'Status':>12}",
            "-" * 80,
        ])

        for comp in comparisons:
            lines.append(
                f"{comp['name']:<35} "
                f"{comp['current_ms']:>12.3f} "
                f"{comp['baseline_ms']:>12.3f} "
                f"{comp['change_pct']:>+9.2f}% "
                f"{comp['status']:>12}"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_json_report(
        self,
        comparisons: List[Dict],
        current_meta: Dict,
        baseline_meta: Dict,
        output_file: Optional[str] = None
    ) -> Dict:
        """Generate a JSON report.

        Args:
            comparisons: List of comparison dictionaries
            current_meta: Metadata for current run
            baseline_meta: Metadata for baseline
            output_file: Optional output file path

        Returns:
            Report as dictionary
        """
        import json

        regressions = [c for c in comparisons if c["status"] == "REGRESSION"]
        improvements = [c for c in comparisons if c["status"] == "IMPROVEMENT"]
        stable = [c for c in comparisons if c["status"] == "STABLE"]

        report = {
            "generated_at": datetime.now().isoformat(),
            "current": current_meta,
            "baseline": baseline_meta,
            "summary": {
                "total": len(comparisons),
                "stable": len(stable),
                "improvements": len(improvements),
                "regressions": len(regressions)
            },
            "regressions": regressions,
            "improvements": improvements,
            "all_results": comparisons
        }

        if output_file:
            output_path = self.report_dir / output_file
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    @staticmethod
    def _format_comparison_row(comp: Dict) -> str:
        """Format a single comparison as a markdown table row.

        Args:
            comp: Comparison dictionary

        Returns:
            Markdown table row
        """
        status_emoji = {
            "REGRESSION": "⚠️",
            "IMPROVEMENT": "🚀",
            "STABLE": "✅"
        }

        emoji = status_emoji.get(comp["status"], "")

        return (
            f"| {comp['name']} "
            f"| {comp['current_ms']:.3f} "
            f"| {comp['baseline_ms']:.3f} "
            f"| {comp['change_pct']:+.2f}% "
            f"| {emoji} {comp['status']} |"
        )

    @staticmethod
    def _generate_category_breakdown(comparisons: List[Dict]) -> List[str]:
        """Generate category-level breakdown.

        Args:
            comparisons: List of comparison dictionaries

        Returns:
            List of markdown lines
        """
        # Extract categories from benchmark names
        categories = {}
        for comp in comparisons:
            # Simple heuristic: extract category from name prefix
            parts = comp['name'].split('_')
            category = parts[0] if parts else 'unknown'

            if category not in categories:
                categories[category] = []
            categories[category].append(comp)

        lines = [
            "## Category Breakdown",
            "",
        ]

        for category, comps in sorted(categories.items()):
            regressions = sum(1 for c in comps if c["status"] == "REGRESSION")
            improvements = sum(1 for c in comps if c["status"] == "IMPROVEMENT")

            lines.append(f"### {category.capitalize()}")
            lines.append("")
            lines.append(f"- Total: {len(comps)}")
            lines.append(f"- Regressions: {regressions}")
            lines.append(f"- Improvements: {improvements}")
            lines.append("")

        return lines
