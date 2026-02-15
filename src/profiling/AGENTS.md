<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# profiling
## Purpose
Implementation files for GPU profiling and performance analysis. Implements Metal System Trace integration and performance counter reading.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for profiling module |
| trace.mm | TraceManager implementation (MTLCaptureManager API, os_signpost integration) |
| perf_counters.mm | PerfCounterReader implementation (MTLCounterSampleBuffer queries) |
## For AI Agents
### Working In This Directory
- **trace.mm**: Uses MTLCaptureManager for programmatic GPU capture, os_log for signpost markers
- **perf_counters.mm**: Queries available MTLCounterSets and samples GPU hardware counters
### Common Patterns
```objc
// GPU capture
MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
MTLCaptureDescriptor* descriptor = [[MTLCaptureDescriptor alloc] init];
descriptor.captureObject = device;
descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
descriptor.outputURL = [NSURL fileURLWithPath:path];
[captureManager startCaptureWithDescriptor:descriptor error:&error];
// ... GPU work ...
[captureManager stopCapture];

// Signpost markers
os_log_t log = os_log_create("com.metal_native", "operations");
os_signpost_interval_begin(log, signpost_id, "operation_name");
// ... GPU work ...
os_signpost_interval_end(log, signpost_id, "operation_name");

// Performance counters
id<MTLCounterSet> counterSet = [device.counterSets objectAtIndex:0];
id<MTLCounterSampleBuffer> sampleBuffer =
    [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
// Sample and read counter values
```
## Dependencies
### Internal
- ../include/metal_native/profiling/ (public headers)
- core/ (MNDevice)
### External
- Metal framework (MTLCaptureManager, MTLCounterSampleBuffer)
- os_signpost (Darwin unified logging)
<!-- MANUAL: -->
