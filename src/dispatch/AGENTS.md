<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# dispatch
## Purpose
Implementation files for GPU command submission and synchronization. Implements triple-buffered command pipeline, backpressure control, CPU-GPU sync, and worker thread.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for dispatch module |
| command_pipeline.mm | CommandPipeline implementation (MTLCommandBuffer ring buffer, lazy commit) |
| backpressure.cpp | BackpressureController implementation (condition variable-based throttle) |
| sync.mm | EventManager implementation (MTLSharedEvent, listener dispatch queue) |
| worker_thread.cpp | WorkerThread implementation (SPSC queue, std::thread, mutex/cv) |
## For AI Agents
### Working In This Directory
- **command_pipeline.mm**: Maintains ring buffer of MTLCommandBuffer slots, rotates on commit_and_continue()
- **backpressure.cpp**: Uses std::condition_variable (no std::counting_semaphore in C++17)
- **sync.mm**: Creates MTLSharedEvent and dispatch queue for async CPU-GPU synchronization
- **worker_thread.cpp**: Pure C++ worker thread with SPSC work queue (no Metal dependencies)
### Common Patterns
- CommandPipeline integrates BackpressureController to block when all slots are in-flight
- EventManager uses MTLCommandBuffer's `encodeSignalEvent:value:` and `encodeWaitForEvent:value:`
- WorkerThread allows Python to submit Metal encoding work without holding GIL
## Dependencies
### Internal
- ../include/metal_native/dispatch/ (public headers)
- core/ (MNDevice)
### External
- Metal framework (command_pipeline.mm, sync.mm)
- libdispatch (sync.mm for dispatch queue)
- C++ standard library (thread, mutex, condition_variable)
<!-- MANUAL: -->
