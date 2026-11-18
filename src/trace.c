// The hot path of tracing: every traced operation reads its operands' boxes
// and records one call. These read and build the same objects as their R
// counterparts in R/graph.R and R/promotion.R, for a fraction of the cost.

#include <R.h>
#include <Rinternals.h>
#include <Rversion.h>
#include <string.h>

#include "trace.h"

static SEXP sym_aval = NULL;
static SEXP sym_label = NULL;

void anvl_init_trace_symbols(void) {
  sym_aval = Rf_install("aval");
  sym_label = Rf_install("label");
}

// The element `name` of the list `x`, or NULL.
static SEXP list_elt(SEXP x, const char *name) {
  SEXP names = Rf_getAttrib(x, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP) {
    return R_NilValue;
  }
  R_xlen_t n = XLENGTH(x);
  for (R_xlen_t i = 0; i < n; i++) {
    if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
      return VECTOR_ELT(x, i);
    }
  }
  return R_NilValue;
}

// The aval of the graph node `gnode` (a GraphValue or a GraphLiteral), or NULL.
static SEXP node_aval(SEXP gnode) {
  if (TYPEOF(gnode) != ENVSXP || !R_existsVarInFrame(gnode, sym_aval)) {
    return R_NilValue;
  }
#if R_VERSION >= R_Version(4, 5, 0)
  return R_getVar(sym_aval, gnode, FALSE);
#else
  return findVarInFrame(gnode, sym_aval);
#endif
}

// The aval of `x` if it is a box of a traced array that is not an R value, and
// with `desc` not NULL, of the trace `desc`; NULL otherwise.
static SEXP typed_box_aval(SEXP x, SEXP desc) {
  if (TYPEOF(x) != VECSXP || !Rf_inherits(x, "GraphBox")) {
    return R_NilValue;
  }
  if (desc != R_NilValue && list_elt(x, "desc") != desc) {
    return R_NilValue;
  }
  SEXP aval = node_aval(list_elt(x, "gnode"));
  if (TYPEOF(aval) != VECSXP || Rf_inherits(aval, "RData")) {
    return R_NilValue;
  }
  return aval;
}

// Whether the data types `x` and `y` (each a string with class "DataType")
// are the same.
static int same_dtype(SEXP x, SEXP y) {
  if (TYPEOF(x) != STRSXP || TYPEOF(y) != STRSXP || XLENGTH(x) != 1 ||
      XLENGTH(y) != 1) {
    return 0;
  }
  SEXP cx = STRING_ELT(x, 0);
  SEXP cy = STRING_ELT(y, 0);
  return cx == cy || strcmp(CHAR(cx), CHAR(cy)) == 0;
}

// Whether the shapes `x` and `y` (integer vectors) are the same.
static int same_shape(SEXP x, SEXP y) {
  if (TYPEOF(x) != INTSXP || TYPEOF(y) != INTSXP || XLENGTH(x) != XLENGTH(y)) {
    return 0;
  }
  return memcmp(INTEGER(x), INTEGER(y), XLENGTH(x) * sizeof(int)) == 0;
}

// The data type all of the list `xs` share as boxes of traced arrays that are
// not R values, which with `same_shape_` TRUE also share one shape; NULL if
// they do not, or `xs` is empty.
SEXP anvl_boxes_dtype(SEXP xs, SEXP same_shape_) {
  R_xlen_t n = XLENGTH(xs);
  if (n == 0) {
    return R_NilValue;
  }
  int check_shape = Rf_asLogical(same_shape_) == TRUE;
  SEXP dtype0 = R_NilValue;
  SEXP shape0 = R_NilValue;
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP aval = typed_box_aval(VECTOR_ELT(xs, i), R_NilValue);
    if (aval == R_NilValue) {
      return R_NilValue;
    }
    SEXP dtype = list_elt(aval, "dtype");
    if (i == 0) {
      if (TYPEOF(dtype) != STRSXP) {
        return R_NilValue;
      }
      dtype0 = dtype;
      shape0 = list_elt(aval, "shape");
      continue;
    }
    if (!same_dtype(dtype, dtype0)) {
      return R_NilValue;
    }
    if (check_shape && !same_shape(list_elt(aval, "shape"), shape0)) {
      return R_NilValue;
    }
  }
  return dtype0;
}

// The graph nodes of `args` and their avals, named as `args`, as
// list(gnodes, avals), if every one is a box of the trace `desc` that is not an
// R value; NULL otherwise.
SEXP anvl_operand_nodes(SEXP args, SEXP desc) {
  R_xlen_t n = XLENGTH(args);
  SEXP gnodes = PROTECT(Rf_allocVector(VECSXP, n));
  SEXP avals = PROTECT(Rf_allocVector(VECSXP, n));
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP x = VECTOR_ELT(args, i);
    SEXP aval = typed_box_aval(x, desc);
    if (aval == R_NilValue) {
      UNPROTECT(2);
      return R_NilValue;
    }
    SET_VECTOR_ELT(gnodes, i, list_elt(x, "gnode"));
    SET_VECTOR_ELT(avals, i, aval);
  }
  Rf_setAttrib(avals, R_NamesSymbol, Rf_getAttrib(args, R_NamesSymbol));
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, gnodes);
  SET_VECTOR_ELT(out, 1, avals);
  UNPROTECT(3);
  return out;
}

// A list of the values `values`, named `names` and of class `cls`.
static SEXP classed_list(int n, SEXP *values, const char **names,
                         const char *cls) {
  SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
  SEXP nms = PROTECT(Rf_allocVector(STRSXP, n));
  for (int i = 0; i < n; i++) {
    SET_VECTOR_ELT(out, i, values[i]);
    SET_STRING_ELT(nms, i, Rf_mkChar(names[i]));
  }
  Rf_setAttrib(out, R_NamesSymbol, nms);
  Rf_setAttrib(out, R_ClassSymbol, Rf_mkString(cls));
  UNPROTECT(2);
  return out;
}

// The aval an output of a call gets for the aval `aval` its inference rule
// returned: a ConcreteArray (a rule may hand an input back) as the
// AbstractArray of its data type and shape, as `abstract_aval()` does.
static SEXP output_aval(SEXP aval) {
  if (!Rf_inherits(aval, "ConcreteArray")) {
    return aval;
  }
  SEXP values[2] = {list_elt(aval, "dtype"), list_elt(aval, "shape")};
  const char *names[2] = {"dtype", "shape"};
  return classed_list(2, values, names, "AbstractArray");
}

// A GraphValue of the aval `aval`, as `GraphValue()` builds one.
static SEXP new_graph_value(SEXP aval) {
  SEXP env = PROTECT(R_NewEnv(R_EmptyEnv, FALSE, 0));
  Rf_defineVar(sym_aval, aval, env);
  Rf_defineVar(sym_label, R_NilValue, env);
  Rf_setAttrib(env, R_ClassSymbol, Rf_mkString("GraphValue"));
  UNPROTECT(1);
  return env;
}

// The call of the primitive `primitive` on the graph nodes `gnodes` with the
// parameters `params`, whose inference rule returned the avals `out_avals`, in
// the trace `desc`: list(statement, gvals, boxes), the GraphStatement, a fresh
// GraphValue per output and a GraphBox of each in `desc`, which the caller
// records and registers.
SEXP anvl_record_call(SEXP primitive, SEXP gnodes, SEXP params, SEXP out_avals,
                      SEXP desc) {
  R_xlen_t n = XLENGTH(out_avals);
  SEXP gvals = PROTECT(Rf_allocVector(VECSXP, n));
  SEXP boxes = PROTECT(Rf_allocVector(VECSXP, n));
  const char *box_names[2] = {"gnode", "desc"};
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP aval = PROTECT(output_aval(VECTOR_ELT(out_avals, i)));
    SEXP gval = PROTECT(new_graph_value(aval));
    SET_VECTOR_ELT(gvals, i, gval);
    SEXP box_values[2] = {gval, desc};
    SET_VECTOR_ELT(boxes, i,
                   classed_list(2, box_values, box_names, "GraphBox"));
    UNPROTECT(2);
  }
  SEXP stmt_values[4] = {primitive, gnodes, params, gvals};
  const char *stmt_names[4] = {"primitive", "inputs", "params", "outputs"};
  SEXP statement =
      PROTECT(classed_list(4, stmt_values, stmt_names, "GraphStatement"));
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
  SET_VECTOR_ELT(out, 0, statement);
  SET_VECTOR_ELT(out, 1, gvals);
  SET_VECTOR_ELT(out, 2, boxes);
  UNPROTECT(4);
  return out;
}

// Whether the avals `x` and `y` are arrays of one data type and shape, neither
// of them an R value (`eq_type()`), for the inference rules of element-wise
// operations to accept them at once.
SEXP anvl_same_aval_type(SEXP x, SEXP y) {
  if (TYPEOF(x) != VECSXP || TYPEOF(y) != VECSXP ||
      !Rf_inherits(x, "AbstractArray") || !Rf_inherits(y, "AbstractArray") ||
      Rf_inherits(x, "RData") || Rf_inherits(y, "RData")) {
    return Rf_ScalarLogical(FALSE);
  }
  int same = same_dtype(list_elt(x, "dtype"), list_elt(y, "dtype")) &&
             same_shape(list_elt(x, "shape"), list_elt(y, "shape"));
  return Rf_ScalarLogical(same);
}
