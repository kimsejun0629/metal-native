# Regression Testing Framework

A comprehensive regression testing framework for tracking MetalNative kernel performance across code changes and preventing performance regressions.

## Quick Start

### 1. Save a Baseline

Save current performance as a baseline (typically done on main branch):

```bash
cd /Users/kimsejun/Documents/projects/pytorch_mps/metal_native
python3 benchmarks/regression_runner.py --save-baseline --baseline-label v1.0
```

### 2. Compare Against Baseline

After making code changes, compare against the baseline:

```bash
python3 benchmarks/regression_runner.py --compare --baseline v1.0
```

### 3. Run in CI with Fail on Regression

Use in continuous integration to block PRs with performance regressions:

```bash
python3 benchmarks/regression_runner.py \
  --compare \
  --baseline latest \
  --fail-on-regression \
  --report markdown \
  --report-file regression_report.md
```

## Usage

### Basic Commands

```bash
# Run all benchmarks and save as baseline
python3 benchmarks/regression_runner.py --save-baseline

# Compare with latest baseline
python3 benchmarks/regression_runner.py --compare

# Run specific category only
python3 benchmarks/regression_runner.py --category matmul --save-baseline

# Compare with specific baseline
python3 benchmarks/regression_runner.py --compare --baseline v1.0

# Generate markdown report
python3 benchmarks/regression_runner.py --compare --report markdown

# Save report to file
python3 benchmarks/regression_runner.py --compare --report markdown --report-file results.md
```

### Command-Line Options

- `--config PATH`: Path to regression config file (default: `benchmarks/regression_config.json`)
- `--save-baseline`: Save results as a new baseline
- `--baseline-label LABEL`: Label for saved baseline (default: timestamp)
- `--compare`: Compare with existing baseline
- `--baseline LABEL`: Baseline to compare against (default: "latest")
- `--category CATEGORY`: Run only benchmarks in this category (matmul, attention, softmax, normalization, activation)
- `--report FORMAT`: Report format: text, markdown, or json (default: text)
- `--report-file FILE`: Save report to file
- `--fail-on-regression`: Exit with non-zero code if regressions detected (for CI)

## Configuration

Edit `benchmarks/regression_config.json` to configure benchmarks:

```json
{
    "benchmarks": [
        {
            "name": "matmul_fp16_decode",
            "category": "matmul",
            "op": "matmul",
            "params": {"M": 1, "N": 4096, "K": 4096, "dtype": "float16"},
            "warmup_iters": 50,
            "bench_iters": 200,
            "regression_threshold": 0.05
        }
    ],
    "baseline_dir": "benchmarks/baselines",
    "report_dir": "benchmarks/reports",
    "regression_threshold_default": 0.05,
    "improvement_threshold": 0.03,
    "min_samples": 100,
    "outlier_removal": "iqr"
}
```

### Benchmark Categories

- **matmul**: Matrix multiplication kernels
- **attention**: Flash attention kernels
- **softmax**: Softmax operations
- **normalization**: RMSNorm and other normalization layers
- **activation**: SwiGLU and other activation functions

### Adding New Benchmarks

Add entries to the `benchmarks` array in `regression_config.json`:

```json
{
    "name": "my_new_benchmark",
    "category": "matmul",
    "op": "matmul",
    "params": {"M": 512, "N": 512, "K": 512, "dtype": "float32"},
    "warmup_iters": 50,
    "bench_iters": 200,
    "regression_threshold": 0.05
}
```

## Baseline Management

### List Available Baselines

```python
from baseline_manager import BaselineManager

bm = BaselineManager("benchmarks/baselines")
baselines = bm.list_baselines()

for b in baselines:
    print(f"{b['label']}: {b['git_hash'][:8]} ({b['timestamp']})")
```

### Load a Baseline

```python
baseline_data = bm.load_baseline("v1.0")
print(f"Baseline from git {baseline_data['git_hash'][:8]}")
```

### Find Baseline by Git Hash

```python
baseline = bm.get_baseline_by_git_hash("9dbf497d")
```

## Report Formats

### Text Report

Plain text format suitable for terminal output:

```
================================================================================
PERFORMANCE REGRESSION TEST REPORT
================================================================================
...
```

### Markdown Report

Formatted markdown with tables and emoji status indicators:

```markdown
# Performance Regression Test Report

## Summary
- ✅ **Stable:** 4
- 🚀 **Improvements:** 0
- ⚠️ **Regressions:** 0
...
```

### JSON Report

Machine-readable JSON format:

```json
{
  "summary": {
    "total": 9,
    "stable": 9,
    "improvements": 0,
    "regressions": 0
  },
  "regressions": [],
  "improvements": []
}
```

## CI Integration

### GitHub Actions Example

```yaml
name: Performance Regression Check

on: [pull_request]

jobs:
  regression-test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build MetalNative
        run: |
          cd metal_native
          mkdir build && cd build
          cmake .. && make

      - name: Run Regression Tests
        run: |
          python3 benchmarks/regression_runner.py \
            --compare \
            --baseline main \
            --fail-on-regression \
            --report markdown \
            --report-file regression_report.md

      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: regression-report
          path: benchmarks/reports/regression_report.md
```

## Thresholds

- **Regression threshold**: Default 5% (configurable per benchmark)
  - If current performance is >5% slower than baseline, marked as REGRESSION
- **Improvement threshold**: Default 3%
  - If current performance is >3% faster than baseline, marked as IMPROVEMENT
- **Stable**: Performance within threshold bounds

## Statistical Methods

### Outlier Removal

Uses IQR (Interquartile Range) method by default:
- Calculate Q1 (25th percentile) and Q3 (75th percentile)
- IQR = Q3 - Q1
- Remove samples outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- If too many samples filtered (>50%), use original samples

### Metrics Reported

- **Median**: Primary performance metric (robust to outliers)
- **Mean**: Average performance
- **Std Dev**: Performance variance
- **Min/Max**: Range of observed performance
- **P95/P99**: 95th and 99th percentile latencies

## File Structure

```
benchmarks/
├── regression_config.json       # Benchmark configuration
├── regression_runner.py         # Main test runner
├── baseline_manager.py          # Baseline storage/retrieval
├── report_generator.py          # Report formatting
├── baselines/                   # Saved baselines
│   ├── latest.json -> v1.0.json
│   ├── v1.0.json
│   └── 20260214_211900.json
└── reports/                     # Generated reports
    └── regression_report.md
```

## Example Workflow

1. **On main branch**: Save baseline after merging features
   ```bash
   git checkout main
   python3 benchmarks/regression_runner.py --save-baseline --baseline-label v1.1
   ```

2. **On feature branch**: Compare against main baseline
   ```bash
   git checkout feature/optimize-matmul
   python3 benchmarks/regression_runner.py --compare --baseline v1.1
   ```

3. **In CI**: Automatically check all PRs
   - Baseline saved on main after merge
   - PR checks compare against main baseline
   - Fail CI if regressions detected

## Troubleshooting

### metal_native not available

If you see "Warning: metal_native not available", the framework will use placeholder timing. This is normal during development. The framework structure is ready for when Python bindings are complete.

### No baseline found

Create a baseline first:
```bash
python3 benchmarks/regression_runner.py --save-baseline
```

### Flaky benchmarks

If benchmarks show high variance:
- Increase `bench_iters` in config
- Check system load (close other applications)
- Ensure GPU is not being used by other processes
- Consider increasing `regression_threshold` for noisy benchmarks
