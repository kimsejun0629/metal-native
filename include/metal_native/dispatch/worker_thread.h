#pragma once

/// @file worker_thread.h
/// @brief Dedicated thread for Metal command encoding.
///
/// WorkerThread provides a single-producer single-consumer (SPSC) work queue
/// backed by a dedicated std::thread.  The Python (or other host) thread
/// submits encoding work via submit(); the worker thread dequeues and
/// executes each item sequentially, ensuring all Metal encoding happens on
/// a single, predictable thread.
///
/// This design avoids GIL contention: the Python thread only enqueues work
/// and can release the GIL while the worker thread does the actual Metal
/// encoding.
///
/// Thread-safe: submit() may be called from any thread; all internal state
/// is protected by a mutex + condition variable.

#include <cstddef>
#include <functional>
#include <memory>

namespace metal_native {

class WorkerThread {
public:
    WorkerThread();
    ~WorkerThread();

    // Non-copyable, non-movable.
    WorkerThread(const WorkerThread&) = delete;
    WorkerThread& operator=(const WorkerThread&) = delete;
    WorkerThread(WorkerThread&&) = delete;
    WorkerThread& operator=(WorkerThread&&) = delete;

    // -- Lifecycle -----------------------------------------------------------

    /// Start the worker thread.  No-op if already running.
    void start();

    /// Stop the worker thread.  Drains the queue first, then joins.
    /// No-op if not running.
    void stop();

    /// True if the worker thread is currently running.
    bool is_running() const;

    // -- Work submission -----------------------------------------------------

    /// Submit a unit of work to be executed on the worker thread.
    ///
    /// @param work  A callable to execute.  Must not be null.
    /// @throws MNException if the worker thread is not running.
    void submit(std::function<void()> work);

    /// Block the calling thread until all previously submitted work
    /// has been executed.
    void drain();

    // -- Queries -------------------------------------------------------------

    /// Number of work items waiting to be executed.
    size_t pending_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal_native
