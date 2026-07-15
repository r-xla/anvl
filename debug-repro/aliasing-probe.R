library(pjrt)
Rcpp::cppFunction("
NumericVector prot_as_float(SEXP x, int nread){
  SEXP p = R_ExternalPtrProtected(x);
  if (p == R_NilValue) return NumericVector(0);
  float* f = reinterpret_cast<float*>(RAW(p));
  NumericVector out(nread);
  for (int i = 0; i < nread; ++i) out[i] = f[i];
  return out;
}")

n <- 64L
t <- sprintf("tensor<%dx%dxf32>", n, n)
src <- paste0(
  "
func.func @main(
  %x: ", t, ",
  %p: ", t, " {tf.aliasing_output = 0 : i32}
) -> ", t, " {
  %0 = \"stablehlo.dot_general\"(%x, %x) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (", t, ", ", t, ") -> ", t, "
  \"func.return\"(%0) : (", t, ") -> ()
}
"
)

prog <- pjrt_program(src = src)
exec <- pjrt_compile(prog)

x <- pjrt_buffer(diag(2, n), dtype = "f32") # x %*% x = diag(4)
await(x)
p <- pjrt_empty(dtype = "f32", shape = c(n, n))
y <- pjrt_execute(exec, x, p)
await(y)

vals <- as_array(y)
cat("output correct:", identical(vals[1, 1], 4), vals[1, 1], "\n")
raw_head <- prot_as_float(y, 8L)
cat("output prot RAWSXP head:", paste(raw_head, collapse = " "), "\n")
cat("expected if aliased: 4 0 0 0 0 0 0 0\n")
