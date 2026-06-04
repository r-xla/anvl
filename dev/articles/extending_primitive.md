# Adding a Primitive

This guide explains how to implement a new primitive. It will primarily
focus on *how* to do this. See the [internals
vignette](https://r-xla.github.io/anvl/dev/articles/internals.md) for
more information on how primitives work.

In general, there are two main reasons to add a new primitive:

1.  The operation is not expressible using existing primitives.
2.  The operation (or, for example, its derivative) would benefit from a
    dedicated implementation.

In order to be able to add a new primitive, it needs to be expressible
in the [{stablehlo}](https://github.com/r-xla/stablehlo) package. There
are various scenarios:

1.  The operation is expressible in the StableHLO language and:
    1.  All required operations are already implemented in the
        {stablehlo} R package.

        \\\rightarrow\\ This is the simplest case and the one we assume
        in this guide.

    2.  One or more required operations are not already implemented in
        the {stablehlo} R package.

        \\\rightarrow\\ You first need to implement the missing
        operations in the {stablehlo} R package. See [this
        issue](https://github.com/r-xla/stablehlo/issues/6) for which
        operations are missing and the [StableHLO
        specification](https://openxla.org/stablehlo/spec) for which are
        available.
2.  The operation is not available/cannot be efficiently expressed in
    StableHLO, because:
    1.  It requires shape dynamism (output shape depends on data, such
        as using boolean values for subsetting):

        \\\rightarrow\\ This is currently not possible, but we hope we
        can add support for this in the future, e.g. via a second,
        dynamic Fortran backend.

    2.  It cannot be expressed or can only expressed inefficiently:

        \\\rightarrow\\ You can implement a stableHLO custom call, see
        the custom print operation in
        [pjrt](https://github.com/r-xla/pjrt). The {pjrt} package has a
        dedicated [“Adding Custom
        Calls”](https://r-xla.github.io/pjrt/dev/articles/ffi.html)
        tutorial. This mechanism is used, for example, to implement some
        of the linear algebra primitives such as SVD: on CPU these call
        into LAPACK, and on GPU they use NVIDIA’s cuSOLVER library. In
        the future it will also be possible to call into user-provided
        custom CUDA kernels, but there is no infrastructure for this yet
        – currently we only call into predefined kernels shipped with
        the CUDA toolkit.

## Adding a Primitive: Practical Example

Let’s add a new primitive step by step. We’ll implement
`prim_repeat_along` – a primitive that repeats an array multiple times
along a specified dimension.

For example, repeating `c(1, 2, 3)` twice along dimension 1 gives
`c(1, 2, 3, 1, 2, 3)`.

This primitive has a *dynamic* input (an array) and two *static*
parameters (how many times to repeat and which dimension).

### Step 1: Define the Primitive

Primitives are created with
[`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md):
it builds the `AnvlPrimitive` metadata object that holds the rules,
wraps the body with
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md), attaches
the metadata, and registers the result in the internal primitive
registry. The returned callable becomes the primitive and is bound to a
`prim_<name>` R symbol.

``` r

library(anvl)
prim_repeat_along <- new_primitive(
  "repeat_along",
  function(operand, times, dim) {
    # type of operand is checked by graph_desc_add()
    infer_fn <- function(operand, times, dim) {
      if (!checkmate::test_integerish(dim, lower = 1, upper = ndims(operand), len = 1L)) {
        cli::cli_abort("{.arg dim} must be between 1 and {ndims(operand)}, but is {.val dim}")
      }
      if (!checkmate::test_integerish(times, lower = 1, len = 1L)) {
        cli_abort("times must be a positive integer, but is {times}")
      }
      new_shape <- shape(operand)
      new_shape[dim] <- new_shape[dim] * times
      list(AbstractArray(
        dtype = dtype(operand),
        shape = Shape(new_shape),
        ambiguous = operand$ambiguous
      ))
    }

    graph_desc_add(
      self,                       # lexically bound to the AnvlPrimitive
      list(operand = operand),    # Dynamic inputs (arrays)
      params = list(              # Static parameters
        times = times,
        dim = dim
      ),
      infer_fn = infer_fn
    )[[1L]]  # Extract single output from list
  },
  static = c("times", "dim")
)
```

The primitive is now callable directly as
`prim_repeat_along(x, times, dim)`.

Key points:

- Pass the lexically-bound `self` (the \[`AnvlPrimitive`\]) as the first
  argument to
  [`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md).
  [`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md)
  installs `self` into the enclosing environment of the body, so this is
  always available inside a primitive.
- The inference function receives abstract arrays (types, not values)
  for dynamic inputs, and actual values for static parameters. It must
  verify that the arguments to the function are valid, and should throw
  clear error messages — {anvl} programs are otherwise hard to debug.
- [`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md)
  returns a list of outputs; use `[[1L]]` for single-output primitives.
- Propagate the `ambiguous` flag from inputs to outputs, see [type
  promotion](https://r-xla.github.io/anvl/dev/articles/type-promotion.md)
  for what this means.

Every argument that is *not* a dynamic array must be listed in `static`.
`new_primitive` calls
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) with
`backend = "auto"` internally, which defers the choice of backend to
call time.

If the primitive is a wrapper around a stablehlo operation, it is
possible to use the corresponding inference function from the stablehlo
package (such as
[`stablehlo::infer_types_concatenate`](https://r-xla.github.io/stablehlo/reference/hlo_concatenate.html)).
When doing so, you need to:

1.  Convert the abstract arrays to stablehlo `ValueType`s using
    [`at2vt()`](https://r-xla.github.io/anvl/dev/reference/at2vt.md).
2.  Call the stablehlo inference function and obtain a list of
    `ValueType`s.
3.  Convert the `ValueType`s back to abstract arrays using
    [`vt2at()`](https://r-xla.github.io/anvl/dev/reference/vt2at.md).
4.  Set the `ambiguous` flag of the output depending on the inputs
    (`ambiguity` is strictly an {anvl} concept, not a stablehlo
    concept).

#### Special case: primitives with 0 dynamic inputs

Array *constructors* (e.g. `prim_fill`, `prim_iota`) have no dynamic
array inputs – every argument is static. Two things change in this case:

1.  **All formals must be listed in `static`**. Otherwise {anvl} would
    try to interpret a scalar argument like `value` or `shape` as an
    `AnvlArray` input.
2.  **`backend = "auto"` cannot infer a device from inputs**, because
    there are no array inputs. Instead, add a `device` formal to the
    function and pass `device = device_arg("device")` to
    [`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md).
    At call time, the user-supplied device is read from that argument
    and used to determine both the backend (via
    \[[`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md)\]
    dispatch) and the compilation device.

For example, the real definition of `prim_fill` looks like this:

``` r

prim_fill <- new_primitive(
  "fill",
  function(value, shape, dtype, ambiguous = FALSE, device = NULL) {
    # ... graph_desc_add(self, ...) ...
  },
  static = 1:5,
  device = device_arg("device")
)
```

See
\[[`device_arg()`](https://r-xla.github.io/anvl/dev/reference/device_arg.md)\]
for the detailed semantics.

#### Shortcut helpers for common shapes

`prim_repeat_along` has its own parameters and needs a custom body. Many
primitives are simpler – elementwise unary or binary ops, reductions,
and comparisons all share a standard shape. `R/primitives.R` exposes
factories that generate the body for you:

- `make_unary_op(stablehlo_infer)` – elementwise unary (e.g. `prim_abs`,
  `prim_negate`).
- `make_binary_op(stablehlo_infer)` – elementwise binary
  (e.g. `prim_add`, `prim_mul`).
- `make_reduce_op(infer_fn)` – reductions with `dims` / `drop`
  parameters (e.g. `prim_reduce_sum`).
- `make_compare_op(direction)` – comparison ops with a fixed `direction`
  string (e.g. `prim_eq`, `prim_lt`).

Used together with the corresponding stablehlo inference function, the
primitive definition collapses to a one-liner:

``` r

prim_add <- new_primitive("add", make_binary_op(stablehlo::infer_types_add))
prim_negate <- new_primitive("negate", make_unary_op(stablehlo::infer_types_negate))
prim_reduce_sum <- new_primitive("reduce_sum", make_reduce_op(), static = 2:3)
```

Reach for the manual
[`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md)
form when the primitive has extra static parameters, a custom inference
function, or multiple outputs.

### Step 2: Add the StableHLO Rule

The StableHLO rule defines how to lower this primitive to actual
operations, which is used in the
[`stablehlo()`](https://r-xla.github.io/anvl/dev/reference/stablehlo.md)
lowering pass. We implement `repeat_along` using concatenation:

``` r

prim_repeat_along[["stablehlo"]] <- function(operand, times, dim) {
  operands <- rep(list(operand), times)
  list(rlang::exec(stablehlo::hlo_concatenate, !!!operands, dimension = dim - 1L))
}
```

The rule receives:

- Dynamic inputs as
  [`stablehlo::FuncValue`](https://r-xla.github.io/stablehlo/reference/FuncValue.html)s.
- Static parameters as their R values.

It must return a list of
[`stablehlo::FuncValue`](https://r-xla.github.io/stablehlo/reference/FuncValue.html)s,
even if there is only one output.

**Important**: StableHLO uses 0-based indexing, while {anvl} uses R’s
1-based indexing. Always convert dimension indices by subtracting 1.
Also note that in
[`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md)
we are converting the error messages from stablehlo to our 1-based
indexing, so you do not have to worry about that here.

### Step 3: Add the Reverse Rule

If the operation should support automatic differentiation, attach a
reverse rule built with
[`rule_reverse()`](https://r-xla.github.io/anvl/dev/reference/rule_reverse.md).
The idea here is the following, where we assume the input `operand` has
shape `(s_1, ..., s_n)`, which means that the output (and therefore it’s
gradient) has shape
`(s_1, ..., s_{dim-1}, s_dim * times, s_{dim+1}, ..., s_n)`.

1.  Reshape the gradient to
    `(s_1, ..., s_{dim-1}, s_dim, times, s_{dim+1}, ..., s_n)`.
2.  Sum over the `times` dimension and drop the `times` dimension.

``` r

prim_repeat_along[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  if (!required[[1L]]) {
    return(list(NULL))
  }

  grad <- grads[[1L]]
  operand <- inputs[[1L]]
  dim <- params$dim
  times <- params$times

  old_shape <- shape(operand)
  grad_shape <- shape(grad)

  new_shape <- grad_shape
  new_shape[dim] <- old_shape[dim]
  new_shape <- append(new_shape, times, after = dim - 1L)

  grad_reshaped <- prim_reshape(grad, new_shape)
  grad_summed <- prim_reduce_sum(grad_reshaped, dims = dim, drop = TRUE)
  list(grad_summed)
})
```

The wrapped backward receives:

- `inputs`: Input `GraphValue`s from the forward pass
- `outputs`: Output `GraphValue`s from the forward pass
- `grads`: Gradients flowing back from downstream (one per output)
- `params`: Named list of the call’s static parameters (here:
  `params$dim`, `params$times`)
- `required`: Logical vector indicating which input gradients are needed

It returns a list with one gradient per input (or `NULL` if not
required).

Note that the reverse rule will only be called if at least one input
gradient is required, so this can be assumed.

#### Optional: alternative-forward rule

For most primitives the backward-only form above is enough. If the
gradient computation can be made significantly more efficient by running
an alternative forward pass that exposes intermediate values for reuse,
use the alternative-forward form: `rule_reverse(forward = ...)`. The
forward hook receives `(inputs, params)`, emits whatever forward
primitives it likes via the normal `prim_*` callables, and returns a
list with the primal outputs and a `backward` closure. Intermediates
flow from forward to backward via R’s lexical scoping:

``` r
prim_<name>[["reverse"]] <- rule_reverse(forward = function(inputs, params) {
  x <- inputs[[1L]]
  y <- prim_some_op(x)        # forward emit 1
  z <- prim_other_op(x, y)    # forward emit 2; both `x` and `y` captured
  list(
    outputs  = list(z),       # one box per original call output
    backward = function(inputs, outputs, grads, params, required) {
      # `x` is also available via `inputs`; what lexical capture buys us is
      # access to intermediates like `y` that the framework doesn't pass in.
      ...
    }
  )
})
```

The `backward` closure shares the same signature as the backward-only
form. Intermediates that aren’t passed in (like `y` above) flow in via
lexical capture. The forward must return one output box per original
call output, with matching shape/dtype.

#### Optional: a quickr rule

`prim_<name>[["stablehlo"]]` and `prim_<name>[["reverse"]]` are the two
rules you almost always want. A primitive can optionally also carry a
`quickr` rule, which lowers it to plain R code for the quickr backend
(see `R/rules-quickr.R`). The quickr rule is only required if you want
the primitive to run under `local_backend("quickr")`; if you skip it,
the primitive will still work on the xla backend.

### Step 4: Verify the Registration

[`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md)
returns the callable directly and also registers the primitive in the
internal registry used by the graph machinery, so no separate
registration step is needed:

``` r

prim_repeat_along
#> function (operand, times, dim) 
#> {
#>     if (currently_tracing()) {
#>         cl <- match.call()
#>         cl[[1L]] <- f
#>         return(eval.parent(cl))
#>     }
#>     args <- lapply(as.list(match.call())[-1L], eval, envir = parent.frame())
#>     be <- if (!is.null(device_argname) && !is.null(args[[device_argname]])) {
#>         dev_val <- args[[device_argname]]
#>         if (is.character(dev_val)) 
#>             default_backend()
#>         else backend(dev_val)
#>     }
#>     else {
#>         jit_auto_detect_backend(flatten(args[!names(args) %in% 
#>             static]))
#>     }
#>     if (is.null(jit_fns[[be]])) {
#>         jit_fns[[be]] <<- do.call(jit_with_backend, c(list(f = f, 
#>             static = static, cache_size = cache_size, backend = be), 
#>             if (!is.null(device_argname)) {
#>                 list(device = device_arg(device_argname))
#>             } else if (!is.null(device)) {
#>                 list(device = device)
#>             }, dots))
#>     }
#>     do.call(jit_fns[[be]], args)
#> }
#> <environment: 0x5610c93dce70>
#> attr(,"class")
#> [1] "JitPrimitive" "JitFunction" 
#> attr(,"backend")
#> [1] "auto"
#> attr(,"primitive")
#> <AnvlPrimitive:repeat_along>
```

### Step 5: Add an `nv_` API Function

In {anvl}, we also offer convenience wrappers around the primitives. An
example is `prim_add` vs `nv_add`, where the latter calls into the
former after optionally broadcasting (scalar) inputs:

``` r

nv_add(1L, nv_array(2:3))
#> AnvlArray
#>  3
#>  4
#> [ CPUi32{2} ]
prim_add(1L, nv_array(2:3))
#> Error in `prim_add()`:
#> ! `lhs` and `rhs` must have the same tensor type.
#> ✖ Got tensor<i32> and tensor<2xi32>.
```

In our case, no such convenience is needed and the functionality is not
too low-level (for it to be generally useful), so we can just reassign
the `prim_*` function to an `nv_*` function:

``` r

nv_repeat_along <- prim_repeat_along
```

Note that in the `nv_*` wrapper function, you can only access certain
properties of the input arrayish values via:

- [`shape_abstract()`](https://r-xla.github.io/anvl/dev/reference/abstract_properties.md)
- [`ndims_abstract()`](https://r-xla.github.io/anvl/dev/reference/abstract_properties.md)
- [`dtype_abstract()`](https://r-xla.github.io/anvl/dev/reference/abstract_properties.md)
- [`ambiguous_abstract()`](https://r-xla.github.io/anvl/dev/reference/abstract_properties.md)

If you, for example, use
[`shape()`](https://r-xla.github.io/anvl/dev/reference/shape.md) instead
of
[`shape_abstract()`](https://r-xla.github.io/anvl/dev/reference/abstract_properties.md),
your function won’t work with R literals. I.e., `<extract>_abstract()`
first converts the input to an `AbstractArray` (if possible) and then
extracts the property.

### Using Your Primitive

You can now use the primitive, both eagerly and inside a larger
JIT-compiled function:

``` r

x <- nv_array(c(1, 2, 3), shape = c(3, 1))
# Eager call -- works because prim_repeat_along is itself jit-compiled.
prim_repeat_along(x, times = 2L, dim = 2L)
#> AnvlArray
#>  1 1
#>  2 2
#>  3 3
#> [ CPUf32{3,2} ]
# Traced into the outer graph when composed with another jit-compiled function.
jit(function(x) prim_repeat_along(x, times = 2L, dim = 2L))(x)
#> AnvlArray
#>  1 1
#>  2 2
#>  3 3
#> [ CPUf32{3,2} ]
```

And compute gradients through it.

``` r

f <- function(x) {
  repeated <- prim_repeat_along(x, times = 2L, dim = 2L)
  sum(repeated)
}

grad_f <- jit(gradient(f))
grad_f(x)[[1L]]
#> AnvlArray
#>  2
#>  2
#>  2
#> [ CPUf32{3,1} ]
```

## Contributing to the Package

If you want to contribute your primitive to {anvl}, there are some
additional things to be aware of.

### File Organization

- **`R/primitives.R`**: Define the `prim_*` primitive via
  [`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md)
- **`R/rules-stablehlo.R`**: Add the StableHLO lowering rule
- **`R/rules-reverse.R`**: Add the reverse rule (if differentiable)
- **`R/rules-quickr.R`**: Add the quickr lowering rule (optional; only
  if the primitive should run on the quickr backend)
- **`R/api.R`**: Add the `nv_*` wrapper function (or possibly in another
  **api** file).

### Testing

Tests can go in two places:

1.  **`inst/extra-tests/`**: For tests that compare against torch. These
    live in `inst/` to avoid listing torch as a dependency.
2.  **`tests/testthat/`**: For tests without a torch counterpart.

**Important**: the `describe()` / `test_that()` label must contain the
full primitive name, e.g. `describe("prim_repeat_along", { ... })`. The
meta tests in `tests/testthat/test-primitives-meta.R` verify that every
primitive has corresponding stablehlo and reverse tests, and flag any
that are missing.

Since no torch counterpart exists for `prim_repeat_along`, we would add
manual tests in:

- `tests/testthat/test-primitives-stablehlo.R`
- `tests/testthat/test-primitives-reverse.R`

Also, ensure that no linter errors are present, `devtools::check()`
passes, and format the code using `make format`.

## Higher-Order Primitives

Higher-Order Primitives are primitives that parameterized by an R
function or expression. Examples include `prim_if` and `prim_while`.
These are generally much more complex to handle, so we don’t cover them
here in detail (for now). The general idea, however, is that the
primitive `prim_*` function needs to trace the provided function using
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md)
and then forward this graph to the stablehlo lowering rule and reverse
rule.
