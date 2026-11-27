// XLA FFI handlers for the Poisson distribution. When we lower the matching
// primitives for the CPU, XLA runs these as custom calls (see
// R/rules-stablehlo.R).
//
// A handler runs on one of XLA's threads, so it must never touch the R API.
// To report a problem, it returns an xla::ffi::Error, and pjrt turns that into
// an R error.

#include <cstddef>

#include "ffi-handlers.h"
#include "parallel-for.h"
#include "poisson-kernels.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace anvl {
namespace {

// The smallest chunk of elements worth handing to another thread. An element
// can need up to about 170 terms, each a division and a few multiplications,
// so even a small chunk is plenty of work to pay for scheduling a task.
constexpr size_t kSeriesMinChunk = 256;

template <typename T>
ffi::Error RunSeries(ffi::ThreadPool& pool, ffi::AnyBuffer k,
                     ffi::AnyBuffer lambda, ffi::Result<ffi::AnyBuffer>& out,
                     size_t n) {
  anvl_hwy::PoissonSeriesArgs<T> args;
  args.k = k.typed_data<T>();
  args.lambda = lambda.typed_data<T>();
  args.out = out->typed_data<T>();
  args.scalar_k = k.element_count() == 1;
  args.scalar_lambda = lambda.element_count() == 1;
  return ParallelFor(pool, n, kSeriesMinChunk,
                     [&args](size_t begin, size_t end) {
                       if constexpr (sizeof(T) == 4) {
                         anvl_hwy::poisson_series_f32(args, begin, end);
                       } else {
                         anvl_hwy::poisson_series_f64(args, begin, end);
                       }
                     });
}

// Sums the Poisson tail series for each element of `k` and `lambda` (see
// Series() in poisson-kernels.cpp).
//
// Each operand holds either a single element or as many elements as the
// result. The lowering gives every operand and the result the same layout, so
// element i of one buffer lines up with element i of the others. All three
// share one floating-point data type.
ffi::Error PoissonSeries(ffi::ThreadPool pool, ffi::AnyBuffer k,
                         ffi::AnyBuffer lambda,
                         ffi::Result<ffi::AnyBuffer> out) {
  const ffi::DataType dt = out->element_type();
  if (k.element_type() != dt || lambda.element_type() != dt) {
    return ffi::Error::InvalidArgument(
        "anvl_poisson_series: k, lambda and the result must share one data "
        "type");
  }
  const size_t n = out->element_count();
  for (const size_t m : {k.element_count(), lambda.element_count()}) {
    if (m != 1 && m != n) {
      return ffi::Error::InvalidArgument(
          "anvl_poisson_series: k and lambda must have one element or as many "
          "as the result");
    }
  }
  if (n == 0) return ffi::Error::Success();

  switch (dt) {
    case ffi::DataType::F32:
      return RunSeries<float>(pool, k, lambda, out, n);
    case ffi::DataType::F64:
      return RunSeries<double>(pool, k, lambda, out, n);
    default:
      return ffi::Error::InvalidArgument(
          "anvl_poisson_series: only f32 and f64 are supported");
  }
}

XLA_FFI_DEFINE_HANDLER(kPoissonSeriesHandler, PoissonSeries,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::ThreadPool>()
                           .Arg<ffi::AnyBuffer>()  // k
                           .Arg<ffi::AnyBuffer>()  // lambda
                           .Ret<ffi::AnyBuffer>()  // series
);

}  // namespace

XLA_FFI_Handler* poisson_series_handler() { return kPoissonSeriesHandler; }

}  // namespace anvl
