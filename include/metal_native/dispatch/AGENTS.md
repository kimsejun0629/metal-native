<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# dispatch
## Purpose
GPU command submission and synchronization. Provides triple-buffered command pipeline, backpressure control, CPU-GPU sync primitives, and worker thread for GIL-free encoding.
## Key Files
| File | Description |
|------|-------------|
| command_pipeline.h | CommandPipeline: triple-buffered MTLCommandBuffer management with lazy commit mode |
| backpressure.h | BackpressureController: limits in-flight command buffers to prevent OOM |
| sync.h | EventManager: CPU-GPU synchronization via MTLSharedEvent |
| worker_thread.h | WorkerThread: dedicated thread for Metal encoding (avoids Python GIL contention) |
## For AI Agents
### Working In This Directory
- **CommandPipeline** enables overlapped CPU encoding and GPU execution (default: 3 buffers)
- **BackpressureController** blocks `acquire()` when max in-flight limit is reached
- **EventManager** provides monotonic counter-based sync with timeout support
- **WorkerThread** allows Python to submit work without holding GIL during Metal encoding
### Common Patterns
```cpp
// Triple-buffered dispatch
CommandPipeline& pipeline = device.command_pipeline();
id<MTLCommandBuffer> cb = pipeline.current_buffer();
// ... encode work ...
pipeline.commit_and_continue();  // rotates to next buffer

// CPU-GPU sync
EventManager event_mgr(device);
uint64_t val = event_mgr.next_value();
event_mgr.encode_signal(cb, val);
cb commit];
event_mgr.cpu_wait(val);  // block until GPU reaches this point

// Worker thread (Python binding usage)
WorkerThread worker;
worker.start();
worker.submit([&]() { /* Metal encoding */ });
worker.drain();
```
## Dependencies
### Internal
- core/ (MNDevice)
### External
- Metal framework (MTLCommandBuffer, MTLSharedEvent)
- Foundation framework (dispatch_source for memory pressure)
<!-- MANUAL: -->
