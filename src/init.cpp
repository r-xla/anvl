// Registers anvl's native routines with R.

#define R_NO_REMAP
#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rinternals.h>

#include "ffi-handlers.h"
#include "highway/runtime.h"

namespace {

// Returns the CPU custom-call handlers as a list of external pointers. Each
// is named after the custom-call target it should be registered under.
SEXP cpu_kernel_handlers() {
  struct Entry {
    const char* target;
    XLA_FFI_Handler* handler;
  };
  const Entry entries[] = {
      {"anvl_poisson_series", anvl::poisson_series_handler()},
  };
  const R_xlen_t n = sizeof(entries) / sizeof(entries[0]);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, n));
  for (R_xlen_t i = 0; i < n; ++i) {
    SET_VECTOR_ELT(
        out, i,
        R_MakeExternalPtr(reinterpret_cast<void*>(entries[i].handler),
                          R_NilValue, R_NilValue));
    SET_STRING_ELT(names, i, Rf_mkChar(entries[i].target));
  }
  Rf_setAttrib(out, R_NamesSymbol, names);
  UNPROTECT(2);
  return out;
}

const R_CallMethodDef call_methods[] = {
    {"cpu_kernel_handlers", reinterpret_cast<DL_FUNC>(&cpu_kernel_handlers), 0},
    {nullptr, nullptr, 0}};

}  // namespace

extern "C" attribute_visible void R_init_anvl(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_methods, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
  // Highway detects the CPU once, the first time it's needed. We do it here
  // so that it doesn't happen inside a kernel on one of XLA's threads.
  highway::warm_up();
}
