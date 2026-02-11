/// @file worker_thread.cpp
/// @brief Implementation of WorkerThread.

#include "metal_native/dispatch/worker_thread.h"
#include "metal_native/core/error.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace metal_native {

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct WorkerThread::Impl {
    std::thread             thread;
    std::atomic<bool>       running{false};
    std::atomic<bool>       drain_waiters{false};
    bool                    stop_requested = false;

    mutable std::mutex      mu;
    std::condition_variable work_cv;   ///< Signalled when work is enqueued or stop requested.
    std::condition_variable drain_cv;  ///< Signalled when queue becomes empty.

    std::deque<std::function<void()>> queue;

    /// The worker loop executed on the dedicated thread.
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mu);
                work_cv.wait(lock, [this] {
                    return !queue.empty() || stop_requested;
                });

                if (stop_requested && queue.empty()) {
                    break;
                }

                task = std::move(queue.front());
                queue.pop_front();
            }

            // Execute outside the lock.
            if (task) {
                task();
            }

            // Notify drain waiters after each task completes (only if someone is waiting).
            if (drain_waiters.load(std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lock(mu);
                if (queue.empty()) {
                    drain_cv.notify_all();
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

WorkerThread::WorkerThread()
    : impl_(std::make_unique<Impl>()) {}

WorkerThread::~WorkerThread() {
    stop();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void WorkerThread::start() {
    if (impl_->running.load()) {
        return; // Already running.
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->stop_requested = false;
    }

    impl_->thread = std::thread([this] { impl_->run(); });
    impl_->running.store(true);
}

void WorkerThread::stop() {
    if (!impl_->running.load()) {
        return; // Not running.
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->stop_requested = true;
    }
    impl_->work_cv.notify_one();

    if (impl_->thread.joinable()) {
        impl_->thread.join();
    }
    impl_->running.store(false);
}

bool WorkerThread::is_running() const {
    return impl_->running.load();
}

// ---------------------------------------------------------------------------
// Work submission
// ---------------------------------------------------------------------------

void WorkerThread::submit(std::function<void()> work) {
    MN_CHECK(impl_->running.load(),
             MetalNativeError::InternalError,
             "WorkerThread::submit: worker thread is not running");
    MN_CHECK(work != nullptr,
             MetalNativeError::InvalidArgument,
             "WorkerThread::submit: work function must not be null");

    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->queue.push_back(std::move(work));
    }
    impl_->work_cv.notify_one();
}

void WorkerThread::drain() {
    impl_->drain_waiters.store(true, std::memory_order_relaxed);
    std::unique_lock<std::mutex> lock(impl_->mu);
    impl_->drain_cv.wait(lock, [this] { return impl_->queue.empty(); });
    impl_->drain_waiters.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

size_t WorkerThread::pending_count() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->queue.size();
}

} // namespace metal_native
