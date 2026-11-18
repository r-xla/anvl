#ifndef ANVL_TRACE_H
#define ANVL_TRACE_H

#include <Rinternals.h>

void anvl_init_trace_symbols(void);
SEXP anvl_boxes_dtype(SEXP xs, SEXP same_shape_);
SEXP anvl_operand_nodes(SEXP args, SEXP desc);
SEXP anvl_record_call(SEXP primitive, SEXP gnodes, SEXP params, SEXP out_avals,
                      SEXP desc);

#endif
