<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# profiling
## Purpose
GPU profiling and performance analysis. Provides Metal System Trace integration for programmatic GPU capture and hardware performance counter reading.
## Key Files
| File | Description |
|------|-------------|
| trace.h | TraceManager: programmatic GPU capture (MTLCaptureManager) and os_signpost markers |
| perf_counters.h | PerfCounterReader: GPU hardware counter sampling (utilization, bandwidth, occupancy) |
## For AI Agents
### Working In This Directory
- **TraceManager** enables programmatic GPU frame capture for viewing in Xcode Instruments
- **TraceScope** provides RAII signpost management for operation tracking
- **PerfCounterReader** reads MTLCounterSampleBuffer for GPU metrics (not all counters available on all chips)
### Common Patterns
```cpp
// GPU capture
TraceManager& trace = TraceManager::instance();
trace.set_capture_destination("/tmp/trace.gputrace");
trace.begin_capture();
// ... run operations ...
trace.end_capture();
// View in Xcode: Instruments -> Metal System Trace

// Signpost markers
{
    TraceScope scope("matmul_forward");
    // ... GPU work ...
}  // signpost ends automatically

// Performance counters
PerfCounterReader reader;
if (reader.is_supported()) {
    PerfCounterSnapshot snap = reader.sample();
    if (snap.is_valid) {
        printf("GPU util: %.1f%%\n", snap.gpu_utilization);
        printf("Bandwidth: %.2f GB/s\n", snap.memory_bandwidth);
    }
}
```
## Dependencies
### Internal
- core/ (MNDevice)
### External
- Metal framework (MTLCaptureManager, MTLCounterSampleBuffer)
- os_signpost (Darwin logging framework)
<!-- MANUAL: -->
