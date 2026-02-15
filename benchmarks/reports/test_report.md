# Performance Regression Test Report

**Generated:** 2026-02-14 21:20:03

## Summary

- **Current Git Hash:** `9dbf497d72d11ec228f5de7b26a13e000284cd54`
- **Current Branch:** `main`
- **Baseline:** all_ops (Git: `9dbf497d`)
- **Total Benchmarks:** 4

### Status Breakdown

- ✅ **Stable:** 4
- 🚀 **Improvements:** 0
- ⚠️ **Regressions:** 0

## All Results

| Benchmark | Current (ms) | Baseline (ms) | Change | Status |
|-----------|--------------|---------------|--------|--------|
| matmul_fp16_decode | 0.017 | 0.017 | -0.40% | ✅ STABLE |
| matmul_fp16_small | 2.137 | 2.137 | -0.01% | ✅ STABLE |
| matmul_fp16_medium | 8.542 | 8.605 | -0.73% | ✅ STABLE |
| matmul_fp16_large | 34.684 | 34.326 | +1.04% | ✅ STABLE |

## Category Breakdown

### Matmul

- Total: 4
- Regressions: 0
- Improvements: 0
