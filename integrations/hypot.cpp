// The `hypot` custom-call handler from the "Custom Calls" article, kept in
// sync with it: an `AnyBuffer` handler that dispatches on the operand dtype,
// so the same target serves f32 and f64.
//
// Compiled on the fly by integrations/run.R via Rcpp::sourceCpp().

// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(pjrt)]]
#include <Rcpp.h>
#include <cmath>
#include "xla/ffi/api/ffi.h"

using namespace xla::ffi;

template <typename T>
static Error hypot_impl(AnyBuffer x, AnyBuffer y, Result<AnyBuffer> out) {
  size_t n = out->element_count();
  if (x.element_count() != n || y.element_count() != n) {
    return Error::InvalidArgument("hypot: operands must have the same size");
  }
  const T *xp = x.typed_data<T>();
  const T *yp = y.typed_data<T>();
  T *op = out->typed_data<T>();
  for (size_t i = 0; i < n; ++i) {
    op[i] = std::hypot(xp[i], yp[i]);
  }
  return Error::Success();
}

static Error do_hypot(AnyBuffer x, AnyBuffer y, Result<AnyBuffer> out) {
  DataType dt = x.element_type();
  if (y.element_type() != dt || out->element_type() != dt) {
    return Error::InvalidArgument("hypot: operands and result must have the "
                                  "same dtype");
  }
  switch (dt) {
    case DataType::F32:
      return hypot_impl<float>(x, y, out);
    case DataType::F64:
      return hypot_impl<double>(x, y, out);
    default:
      return Error::InvalidArgument("hypot: only f32 and f64 are supported");
  }
}

XLA_FFI_DEFINE_HANDLER(hypot_handler, do_hypot,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>());

// [[Rcpp::export]]
SEXP hypot_handler_ptr() {
  return R_MakeExternalPtr((void *)hypot_handler, R_NilValue, R_NilValue);
}
