# Extending the API

In this vignette we will cover some general guidelines that ensure your
anvl functions come without surprises. This is primarily intended for
extending the API – either in your own package or contributing to {anvl}
itself – but is also helpful when writing your own scripts.

The general guidelines are:

1.  The function must be pure.
2.  Consistent input and output types:
    1.  The dynamic (arrayish) inputs should accept `AnvlArray`s as well
        as R vectors of length 1 and `array`s.
    2.  The function should only output `AnvlArray`s.
3.  The function should work with arbitrary devices.
4.  The function should (unless there are specific reasons) work in
    eager and jit mode.
5.  Use static arguments when you require data-dependent input checks.
6.  Tag the function with `#' @jit` so it is jit-wrapped at package
    build time (see *Jit-wrapping API Functions* below).

## Pure Functions

This is extensively covered in the *JIT Deep Dive*, so we won’t repeat
it here. While the subsequent sections mostly address issues that are
relevant in eager mode, purity is the primary requirement to enable
usage of [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)
with your function.

## Consistent Input and Output Types

Functions in anvl have dynamic (arrayish) and static (standard R values)
inputs. However, it can also be convenient to pass R objects as dynamic
inputs and let anvl convert them. To enable this, there are the
[`as_anvl_array()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
and
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
converters. You should call them at the top of your function. Not only
will these functions convert the inputs, they will also check them for
compatibility, specifically w.r.t. their device and backend. If they
don’t live on the same device, an error will be thrown.

Note that this is only really necessary for using your function in eager
mode (i.e. without
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)). This is
because when a function is wrapped in
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md), {anvl}
itself can perform these checks automatically.

The advantage of this input standardization is best illustrated with an
example.

Consider the naive implementation of reshaping, which will fail when
called on an R vector:

``` r

library(anvl)
# x: dynamic, shape: static
nv_reshape_naive <- function(x, shape) {
  if (!identical(shape(x), shape)) {
    prim_reshape(x, shape)
  } else {
    x
  }
}
nv_reshape_naive(1L, c(2, 2))
#> Error in `UseMethod()`:
#> ! no applicable method for 'shape' applied to an object of class "c('integer', 'numeric')"
```

This is because the attribute-getters such as
[`shape()`](https://r-xla.github.io/anvl/dev/reference/shape.md),
[`dtype()`](https://r-xla.github.io/anvl/dev/reference/dtype.md), etc.
are only implemented for `AnvlArray`s, not for R vectors, so
canonicalizing inputs at the top ensures the function works correctly.

Also, consider this function that converts an input to a specific dtype
(or keeps it as-is if `dtype` is `NULL`). The problem is that in the
no-op case, we return a static R object instead of (as intended) an
`AnvlArray`.

``` r

# x: dynamic, dtype: static
nv_convert_naive <- function(x, dtype) {
  if (is.null(dtype)) {
    return(x)
  }
  prim_convert(x, dtype)
}
nv_convert_naive(1L, "i16")
#> AnvlArray
#>  1
#> [ CPUi16{} ]
nv_convert_naive(1L, NULL)
#> [1] 1
```

By canonicalizing inputs, such pitfalls can be avoided.

Finally, note that primitives such as
[`prim_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_convert.md)
already canonicalize their inputs, so if you are only wrapping
primitives (or other `nv_<op>` functions that already canonicalize), you
might not have to do this yourself.

When a function takes multiple arrayish inputs, normalize them in a
single `as_anvl_arrays(...)` call covering all of them, so R
literals/arrays adopt the device of their AnvlArray siblings instead of
landing on the default device.

## Arbitrary Devices

In order to ensure that your function works with inputs from arbitrary
devices, you need to be careful when creating new constants within your
function. Let’s say you are creating your function and working on GPU:

``` r

nv_add_one_naive <- function(x) {
  x <- as_anvl_array(x)
  x + nv_fill(1L, shape(x), device = "cuda")
}
```

As long as you are adding ones on a CUDA GPU, this function will work
fine! However, if you suddenly use it on the CPU, it will fail, because
we can’t add a CPU array to a CUDA array. Constants should always be
initialized on the same device as the inputs. If there are multiple
inputs and you called
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
on them at the top, you know that there is only a single device.

One way to achieve this is to simply pass the input’s device to
[`nv_fill()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md):

``` r

nv_add_one1 <- function(x) {
  x <- as_anvl_array(x)
  x + nv_fill(1L, shape(x), device = device(x))
}
```

Another option is to rely on `nv_<op>_like` functions. These take in
another `AnvlArray` as their first input and use its properties as the
defaults for their arguments. In this case, the created array will
assume the data type, shape and device from the input array.

``` r

nv_add_one2 <- function(x) {
  x + nv_fill_like(x, 1L)
}
```

Note that when you only want to use a function with
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md), you can
just omit specifying the device at all, as
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) is smart
enough to place it on the correct device.

## Static Arguments to Enable Input Checks

One restriction of the XLA compiler is that it does not really allow for
runtime checks. Let’s say you want to sample from a Bernoulli
distribution with probability `p`. If you make `p` a dynamic input, you
can’t check that it is within `[0, 1]`, so you need to make it a static
input. Don’t convert it to an `AnvlArray` before checking its value.
Later in the function, it will actually be converted, but from XLA’s
point of view, it will just be a constant within the compiled program
and not a dynamic input.

``` r

nv_rbernoulli <- function(initial_state, p) {
  initial_state <- as_anvl_array(initial_state)
  stopifnot((p >= 0) && (p <= 1))

  # returns: (state, sample)
  out <- nv_runif(1L, initial_state)
  out_state <- out[[1L]]
  x <- nv_convert(out[[2L]] <= p, "i32")
  list(out_state, x)
}
nv_rbernoulli(nv_rng_state(1), 0.2)[[2L]]
#> AnvlArray
#>  0
#> [ CPUi32{1} ]
```

## Jit-wrapping API Functions

Most user-facing API functions in anvl are wrapped in
`jit(f, backend = "auto", ...)` so that calling them traces and compiles
a single program instead of executing each operation eagerly. The
wrapping is driven by the `@jit` roclet (see
[`?jit_roclet`](https://r-xla.github.io/anvl/dev/reference/jit_roclet.md)).

In `R/api*.R`, tag any function that performs more than one primitive
operation with `#' @jit`:

``` r

#' @export
#' @jit
nv_log2 <- function(x) {
  x <- as_anvl_array(x)
  nv_log(x) / log(2)
}
```

If the function has static arguments (anything that is not an arrayish
input – dims, shape, dtype, control flags, functions used as templates,
…), list them with `static = c(...)` using either positional indices or
argument names:

``` r

#' @export
#' @jit static = c(2L, 3L)
nv_mean <- function(x, dims = NULL, drop = TRUE) {
  ...
}

#' @export
#' @jit static = c("dimension")
nv_concatenate <- function(..., dimension = NULL) {
  ...
}
```

Use names rather than positions whenever an argument lives after `...`,
since `...` has no fixed position.

**When to skip `#' @jit`.** Don’t tag a function whose body is
essentially a single primitive call – direct aliases
(`nv_log <- prim_log`), `make_do_binary(prim_X)` factories, or thin
wrappers that just validate and forward to one primitive. The underlying
primitive is already jit-wrapped, so adding another
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) layer adds
tracing overhead without fusing anything new. Also skip pure I/O
(`nv_save`, `nv_serialize`), backend constructors (`nv_array`,
`nv_scalar`, `nv_matrix`), and device/state objects (`nv_device`,
`nv_rng_state`).

The roclet writes the list of tagged functions to `R/jit-registry.R` on
every `devtools::document()` run, and `R/zzz.R` applies that registry at
package source time so the wrapped functions are byte-compiled with the
rest of the package.
