#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>

#include "trace.h"

static const R_CallMethodDef call_methods[] = {
    {"boxes_dtype", (DL_FUNC)&anvl_boxes_dtype, 2},
    {"operand_nodes", (DL_FUNC)&anvl_operand_nodes, 2},
    {"record_call", (DL_FUNC)&anvl_record_call, 5},
    {NULL, NULL, 0}};

void R_init_anvl(DllInfo *dll) {
  anvl_init_trace_symbols();
  R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
