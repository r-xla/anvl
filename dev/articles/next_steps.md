# Next Steps

Here, we pick up where the [Get
Started](https://r-xla.github.io/anvl/dev/articles/anvl.md) introduction
left off. It gives a short overview of what is good to know when writing
{anvl} code, and points to the articles that cover each topic in detail.

``` r

library(anvl)
```

## Passing R Values

R values can be passed to jitted functions in two ways. R `numeric(1)`,
`logical(1)` and `array`s are automatically converted when being passed
as arguments that expect `AnvlArray`s. We also refer do such inputs as
being *dynamic*. When doing this conversion, the R object takes the data
type of the `AnvlArray` it is combined with, so `x / 3` keeps the
precision of `x`:

``` r

nv_array(c(1, 2), dtype = "f64") / 3
```

    ## AnvlArray
    ##  0.3333
    ##  0.6667
    ## [ CPUf64{2} ]

Note that vectors of length greater than one are not accepted, as
otherwise `c(1)` would have 0 axes but `c(1, 2)` would have one axis,
which would lead to surprising results and unpredictable behavior.
Instead, use R arrays with a single axis to represent vectors. Wrap them
in [`array()`](https://rdrr.io/r/base/array.html), or use
[`anvl::arr()`](https://r-xla.github.io/anvl/dev/reference/arr.md),
which saves the [`c()`](https://rdrr.io/r/base/c.html).

``` r

nv_add(nv_scalar(1, dtype = "f64"), arr(2, 3))
```

    ## AnvlArray
    ##  3
    ##  4
    ## [ CPUf64{2} ]

When there is no other operand to infer the data type from, the default
data type is used, which can also be configured via the
`anvl.default_dtypes` option.

``` r

nv_add(1, 2)
```

    ## AnvlArray
    ##  3
    ## [ CPUf32{} ]

See the [Data Types and Promotion
Rules](https://r-xla.github.io/anvl/dev/articles/type-promotion.md)
article for the full promotion rules.

When you write your jitted function, you might also want to take in R
values without auto-converting them to `AnvlArray`s. This is what the
`static` argument of
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) is for. It
names function parameters that stay regular R values and this can, for
example, be used for `if`-statements within compiled functions.

``` r

loss <- function(y_hat, y, average) {
  se <- (y_hat - y)^2
  if (average) mean(se) else sum(se)
}
y_hat <- nv_array(c(1, 2, 3))
y <- nv_array(c(1, 2, 5))

mse <- jit(loss, static = "average")
mse(y_hat, y, average = TRUE)
```

    ## AnvlArray
    ##  1.3333
    ## [ CPUf32{} ]

Without `static`, `average` would be converted to an `AnvlArray` like
any other logical and could thus not be used in the if-conditional:

``` r

mse_dynamic <- jit(loss)
mse_dynamic(y_hat, y, average = TRUE)
```

    ## Error in `if (average) ...`:
    ## ! the condition has length > 1

The value of a static argument is fixed in the compiled program, so
every new value compiles a new version of the function. Static arguments
should therefore only take few different values. See the [compilation
cache](https://r-xla.github.io/anvl/dev/articles/jit.html#the-compilation-cache)
section of the JIT deep dive for details.

## Nested Lists of `AnvlArray`s

Jitted functions can take and return (nested) lists of `AnvlArray`s,
e.g. the parameters of a model.
[`map_tree()`](https://r-xla.github.io/pjrt/reference/map_tree.html) and
[`pmap_tree()`](https://r-xla.github.io/pjrt/reference/pmap_tree.html)
apply a function to every `AnvlArray` in such lists, like
[`lapply()`](https://rdrr.io/r/base/lapply.html) and
[`Map()`](https://rdrr.io/r/base/funprog.html), but keeping the nesting:

``` r

params <- list(w = nv_array(c(1, 2)), layer = list(b = nv_scalar(0.5)))
scale_params <- jit(function(params) map_tree(params, \(p) p * 2))
scale_params(params)
```

    ## $w
    ## AnvlArray
    ##  2
    ##  4
    ## [ CPUf32{2} ] 
    ## 
    ## $layer
    ## $layer$b
    ## AnvlArray
    ##  1
    ## [ CPUf32{} ]

See the [nested inputs and
outputs](https://r-xla.github.io/anvl/dev/articles/internals.html#nested-inputs-and-outputs)
section of the internals article for more.

## Subsetting

`AnvlArray`s are subset with `[`, like in R:

``` r

y <- nv_array(1:12, shape = c(3, 4))
y[arr(1L, 3L), 2:3]
```

    ## AnvlArray
    ##  4 7
    ##  6 9
    ## [ CPUi32{2,2} ]

Logical masks cannot be used for subsetting (see [shapes must be known
in advance](#shapes-must-be-known-in-advance) below), and indices
outside of the `AnvlArray` are moved to the nearest valid index instead
of raising an error (see [compiled code cannot throw
errors](#compiled-code-cannot-throw-errors) below). See the
[Subsetting](https://r-xla.github.io/anvl/dev/articles/subsetting.md)
article for the details.

## Value Semantics

Like R objects, `AnvlArray`s have value semantics: modifying an
`AnvlArray`, e.g. via subset assignment with `[<-`, never affects other
variables holding the same `AnvlArray`.

``` r

y2 <- y
y2[1, ] <- 0L
y[1, ]
```

    ## AnvlArray
    ##   1
    ##   4
    ##   7
    ##  10
    ## [ CPUi32{4} ]

``` r

y2[1, ]
```

    ## AnvlArray
    ##  0
    ##  0
    ##  0
    ##  0
    ## [ CPUi32{4} ]

Outside of a jitted function, this means that every subset assignment
copies the whole `AnvlArray`, while inside a jitted function, the
compiler avoids unnecessary copies.

For the same reason, a jitted function does not modify its inputs, so it
needs new memory for its outputs. If an input is not needed after the
call, e.g. the parameters in a model fitting loop, marking it via
`donate` allows the compiled program to reuse its memory for the
outputs:

``` r

update <- jit(function(x, delta) x + delta, donate = "x")
x <- nv_array(c(1, 2, 3))
x <- update(x, 0.1)
x
```

    ## AnvlArray
    ##  1.1000
    ##  2.1000
    ##  3.1000
    ## [ CPUf32{3} ]

The donated `AnvlArray` cannot be used anymore after the call. See the
[donation](https://r-xla.github.io/anvl/dev/articles/efficiency.html#donation)
section of the efficiency article for more.

The copy of an eager subset assignment can be avoided in the same way:
`x[i, inplace = TRUE] <- value` donates `x` to the update, which then
writes into its memory (see [in-place
updates](https://r-xla.github.io/anvl/dev/articles/subsetting.html#in-place-updates)).

## When Functions Are Compiled

When {anvl} compiles a function, it runs the R code once with
placeholders that only know their shape and data type, and records the
operations on `AnvlArray`s that are performed. Everything else in the R
code is resolved during that single run, which has consequences that the
following sections go through. See the
[tracing](https://r-xla.github.io/anvl/dev/articles/jit.html#tracing)
section of the JIT deep dive for the details.

A jitted function is compiled again for every new combination of input
shapes, data types, and static argument values. Keep the shapes fixed
where possible, and only make arguments static that take few different
values. See [the compilation
cache](https://r-xla.github.io/anvl/dev/articles/jit.html#the-compilation-cache)
and [padding inputs to avoid
recompilation](https://r-xla.github.io/anvl/dev/articles/efficiency.html#padding-inputs-to-avoid-recompilation)
for more.

## Control Flow

An R `if` picks one branch while the function is traced, so its
condition cannot depend on the values of `AnvlArray`s, as we have
already seen for the non-static `average` argument in [passing R
values](#passing-r-values). An R loop is unrolled into one copy of its
body per iteration, which slows down compilation for many iterations.
See [R loops are
unrolled](https://r-xla.github.io/anvl/dev/articles/jit.html#r-loops-are-unrolled)
and [R `if` statements pick one
branch](https://r-xla.github.io/anvl/dev/articles/jit.html#r-if-statements-pick-one-branch)
for examples.

It might be tempting to convert the condition to an R value first,
e.g. with [`as.logical()`](https://rdrr.io/r/base/logical.html), as one
would do outside of
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md):

``` r

abs_value <- jit(function(x) {
  if (as.logical(x < 0)) -x else x
})
abs_value(nv_scalar(-2))
```

    ## Error:
    ## ! `as.logical()` is not defined for a <GraphBox>.
    ## ✖ A traced array has no values: it stands for the shape and data type of
    ##   something the compiled program only computes when it is run.
    ## ℹ You can only convert AnvlArrays to R objects outside of jit().

This fails, because while the function is traced, `x` is only a
placeholder that describes the shape and data type of an `AnvlArray`.
Its values only exist once the compiled program runs, so there is
nothing that could be converted to an R value.

[`nv_while()`](https://r-xla.github.io/anvl/dev/reference/nv_while.md)
and [`nv_if()`](https://r-xla.github.io/anvl/dev/reference/nv_if.md)
instead run the loop or branch within the compiled program, where it can
depend on the values of `AnvlArray`s:

``` r

double_until <- jit(function(x, limit) {
  nv_while(list(x = x), \(x) x < limit, \(x) list(x = x * 2))
})
double_until(nv_scalar(1), 100)
```

    ## $x
    ## AnvlArray
    ##  128
    ## [ CPUf32{} ]

Alternatively, an R loop around a jitted function, as in the [Get
Started](https://r-xla.github.io/anvl/dev/articles/anvl.md) article, can
also be an option.

## Shapes Must Be Known in Advance

The shape of every intermediate result must be known when compiling, so
operations such as `x[x > 0]`,
[`which()`](https://rdrr.io/r/base/which.html), or
[`unique()`](https://rdrr.io/r/base/unique.html) cannot be used. Often,
one can instead keep all values and ignore the unwanted ones,
e.g. `sum(nv_ifelse(x > 0, x, 0))` instead of `sum(x[x > 0])`. See the
[Static Shape
Restriction](https://r-xla.github.io/anvl/dev/articles/static_shapes.md)
article for more such patterns.

## Compiled Code Cannot Throw Errors

Once a program is compiled, it cannot throw errors. Errors and
assertions can therefore only depend on the values of static arguments,
or on the shapes and data types of the other inputs, and are raised
while the function is compiled. This also applies to eager code, which
runs each operation as a small compiled program.

Where R would raise an error depending on the values of the inputs,
{anvl} therefore returns some value instead. For example, an index
outside of the `AnvlArray` is moved to the nearest valid index, and the
Cholesky decomposition of a matrix that is not positive definite
contains `NaN`s:

``` r

x <- nv_array(c(1, 2, 3))
x[nv_scalar(10L)]
```

    ## AnvlArray
    ##  3
    ## [ CPUf32{} ]

``` r

m <- matrix(c(1, 2, 2, 1), nrow = 2)
chol(nv_array(m))
```

    ## AnvlArray
    ##  nan nan
    ##    0 nan
    ## [ CPUf32{2,2} ]

## Random Numbers

R’s [`rnorm()`](https://rdrr.io/r/stats/Normal.html) and friends are
only called once, while tracing, so their result is fixed in the
compiled function. Use {anvl}’s own generators instead, which take the
random state as an explicit input and return a new one:

``` r

draw <- jit(function(state) nv_rnorm(state, shape = 2L))
res <- draw(nv_rng_state(42L))
res$values
```

    ## AnvlArray
    ##  0.4299
    ##  0.3097
    ## [ CPUf32{2} ]

See the [Random Number
Generation](https://r-xla.github.io/anvl/dev/articles/random-numbers.md)
article for more.

## Debugging

For the same reason, [`print()`](https://rdrr.io/r/base/print.html)
inside a jitted function only shows a placeholder, once.
[`nv_print()`](https://r-xla.github.io/anvl/dev/reference/nv_print.md)
prints the actual values on every call, and returns its input so it can
be placed anywhere. To step through the code, e.g. with
[`browser()`](https://rdrr.io/r/base/browser.html), call the function
without [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).
See [side effects only fire during
tracing](https://r-xla.github.io/anvl/dev/articles/jit.html#side-effects-only-fire-during-tracing)
for more.

## Differences from Base R

While {anvl} aims to offer an interface that is familiar to R users,
there are various differences to be aware of. For an overview, see the
[Gotchas](https://r-xla.github.io/anvl/dev/articles/gotchas.md) article.

## Saving `AnvlArray`s

An `AnvlArray` only holds a pointer to its data, so
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) does not work for it.
Use [`nv_save()`](https://r-xla.github.io/anvl/dev/reference/nv_save.md)
and [`nv_read()`](https://r-xla.github.io/anvl/dev/reference/nv_read.md)
instead, which use the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

``` r

path <- tempfile(fileext = ".safetensors")
nv_save(list(w = nv_array(c(1, 2)), b = nv_scalar(0.5)), path)
nv_read(path)$w
```

    ## AnvlArray
    ##  1
    ##  2
    ## [ CPUf32{2} ]

## `nv_*` Functions and Primitives

The `nv_*` functions and R operators are built from *primitives*, named
`prim_*`. The primitives are deliberately strict – for example, they do
not combine `AnvlArray`s of different data types – but can express
things for which there is no `nv_*` function, such as reducing an
`AnvlArray` with an arbitrary function:

``` r

logsumexp <- jit(function(x) {
  prim_reduce(x, init = -Inf, axes = 1L, reducer = function(a, b) {
    nv_pmax(a, b) + log1p(exp(-abs(a - b)))
  })
})
logsumexp(nv_array(c(1, 2, 3)))
```

    ## AnvlArray
    ##  3.4076
    ## [ CPUf32{} ]

Anything expressed in terms of primitives is compiled into a single
optimized program, so even algorithms without a built-in operation run
fast. See the [Primitives
Reference](https://r-xla.github.io/anvl/dev/articles/primitives.md) for
the available primitives, and the [Adding a
Primitive](https://r-xla.github.io/anvl/dev/articles/extending_primitive.md)
article for how to add new ones.
