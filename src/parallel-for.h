// Shares the work of a CPU custom call across XLA's intra-op thread pool.
//
// XLA spreads its own loops over this pool, but it runs a custom call on a
// single thread. A handler that binds `.Ctx<xla::ffi::ThreadPool>()` is given
// the same pool. Using it means our kernels use as many threads as XLA was
// set up with, and never compete with XLA's own loops for extra cores.

#ifndef ANVL_PARALLEL_FOR_H_
#define ANVL_PARALLEL_FOR_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace anvl {

namespace internal {

// The state shared by everyone working on one ParallelFor() call.
//
// Tasks hold it through a shared_ptr because the pool may start a task late,
// after all the work is done and ParallelFor() has returned. That task still
// reads `next`, finds nothing left and returns, so the state must outlive the
// call.
struct ParallelForState {
  std::atomic<size_t> next{0};
  size_t n_chunks = 0;
  size_t chunk = 0;
  size_t n = 0;
  std::mutex mutex;
  std::condition_variable all_done;
  size_t done = 0;    // guarded by `mutex`
  std::string error;  // guarded by `mutex`; empty if none
};

// Claims chunks until none are left, and runs `body` on each one.
//
// `body` lives on the caller's stack, so it disappears once ParallelFor()
// returns. That is safe because we only touch `body` after claiming a chunk,
// and ParallelFor() waits for every claimed chunk to finish.
template <class Body>
void RunChunks(const std::shared_ptr<ParallelForState>& state,
               const Body* body) {
  for (;;) {
    const size_t i = state->next.fetch_add(1, std::memory_order_relaxed);
    if (i >= state->n_chunks) return;
    const size_t begin = i * state->chunk;
    const size_t end = std::min(state->n, begin + state->chunk);
    // A C++ exception must never escape into XLA's threads, so we catch it
    // here and report it as an error instead.
    std::string error;
    try {
      (*body)(begin, end);
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "unknown C++ exception";
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!error.empty() && state->error.empty()) state->error = error;
    if (++state->done == state->n_chunks) state->all_done.notify_all();
  }
}

}  // namespace internal

// Runs `body(begin, end)` over ranges that together cover [0, n), using the
// calling thread and XLA's intra-op pool. It returns once every range is done.
// It is meant for compute-bound kernels, so every thread in the pool may help.
//
// We cut the range into chunks of at least `min_chunk` elements, aiming for
// about `chunks_per_thread` chunks per thread. Threads claim chunks as they go,
// rather than each getting a fixed share up front. That matters because for
// some platforms the cores aren't equally fast (e.g. efficiency cores), some
// may be busy with other work, and some elements cost more than others. With
// fixed shares we would always end up waiting for the slowest thread.
//
// The calling thread claims chunks too, instead of just waiting. So the call
// still finishes even if the pool never gets round to the tasks we gave it.
template <class Body>
xla::ffi::Error ParallelFor(xla::ffi::ThreadPool& pool, size_t n,
                            size_t min_chunk, const Body& body,
                            size_t chunks_per_thread = 4) {
  // We make every chunk start at a multiple of 16, the most lanes any CPU
  // target has (16 floats with AVX-512). Then only the last chunk ends in a
  // partial vector.
  constexpr size_t kAlign = 16;
  if (n == 0) return xla::ffi::Error::Success();
  const size_t threads =
      static_cast<size_t>(std::max<int64_t>(pool.num_threads(), 0));
  const size_t workers = threads + 1;  // the calling thread works too
  size_t chunk =
      (n + chunks_per_thread * workers - 1) / (chunks_per_thread * workers);
  chunk = std::max(chunk, min_chunk);
  chunk = (chunk + kAlign - 1) / kAlign * kAlign;

  auto state = std::make_shared<internal::ParallelForState>();
  state->n = n;
  state->chunk = chunk;
  state->n_chunks = (n + chunk - 1) / chunk;
  const Body* body_ptr = &body;
  const size_t tasks = std::min(threads, state->n_chunks - 1);
  for (size_t t = 0; t < tasks; ++t) {
    pool.Schedule([state, body_ptr] { internal::RunChunks(state, body_ptr); });
  }
  internal::RunChunks(state, body_ptr);

  std::unique_lock<std::mutex> lock(state->mutex);
  state->all_done.wait(lock, [&] { return state->done == state->n_chunks; });
  if (!state->error.empty()) return xla::ffi::Error::Internal(state->error);
  return xla::ffi::Error::Success();
}

}  // namespace anvl

#endif  // ANVL_PARALLEL_FOR_H_
