# Eager mode and `jit()` have to mean the same thing. They do not share a code
# path: eagerly, every `nv_*` function is its own jitted call and an R value is
# an argument leaf of it, while under `jit()` the same value is written into the
# body of a larger trace. Tracing is an optimization, not a different language,
# so anything a call produces -- its dtype as much as its digits -- has to agree.
#
# The comparison is on every digit: a value that was rounded through `f32` on the
# way to `f64` differs from the exact one only past the seventh, which is exactly
# the class of difference this is here to catch.

# One R storage type of each kind. `sqrt(2)` rather than a round number: a double
# that survives a detour through f32 tells us nothing.
r_values <- list(double = sqrt(2), integer = 3L, logical = TRUE)

typed_arrays <- function() {
  list(
    f64 = nv_array(c(1, 2), dtype = "f64"),
    f32 = nv_array(c(1, 2), dtype = "f32"),
    i8 = nv_array(c(1L, 2L), dtype = "i8"),
    i64 = nv_array(c(1L, 2L), dtype = "i64"),
    bool = nv_array(c(TRUE, FALSE))
  )
}

# Everything an observer can see about a call's result: the dtype, the shape and
# every digit of the data.
observable <- function(value) {
  if (is_anvl_array(value)) {
    data <- as_array(value)
    return(list(
      dtype = as.character(dtype(value)),
      shape = shape(value),
      data = if (is.double(data)) sprintf("%.17g", data) else format(data)
    ))
  }
  if (is.list(value)) {
    return(lapply(value, observable))
  }
  list(value = format(value))
}

# The result of `f`, or the error it raised -- an error is an observable outcome
# too, and one that must not depend on the mode either.
outcome <- function(f, args) {
  tryCatch(observable(do.call(f, args)), error = function(e) list(error = conditionMessage(e)))
}

expect_eager_jit_equal <- function(f, args, label) {
  eager <- outcome(f, args)
  traced <- outcome(jit(f), args)
  testthat::expect_equal(eager, traced, info = label)
}

# Run `calls` over every (typed array, R value) pair, checking both modes agree.
expect_eager_jit_equal_grid <- function(calls) {
  for (name in names(calls)) {
    for (dt in names(typed_arrays())) {
      for (rt in names(r_values)) {
        expect_eager_jit_equal(
          calls[[name]],
          list(typed_arrays()[[dt]], r_values[[rt]]),
          sprintf("%s [%s, %s]", name, dt, rt)
        )
      }
    }
  }
}
