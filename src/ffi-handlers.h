// The XLA FFI handlers for anvl's CPU kernels. src/init.cpp hands them to R,
// and `register_cpu_kernels()` in R/cpu-kernels.R registers them with pjrt.

#ifndef ANVL_FFI_HANDLERS_H_
#define ANVL_FFI_HANDLERS_H_

#include "xla/ffi/api/c_api.h"

namespace anvl {

XLA_FFI_Handler* poisson_series_handler();

}  // namespace anvl

#endif  // ANVL_FFI_HANDLERS_H_
