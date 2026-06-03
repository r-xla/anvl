# Static Shape Restriction

This vignette covers the static shape restriction within
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md). First, we
describe what this means and then we discuss how to work around it.

Whenever the XLA compiler that underpins
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md) compiles a
program, it must know the shape of every intermediate value at compile
time. This means that functions such as
[`unique()`](https://rdrr.io/r/base/unique.html) cannot be part of a
jit-compiled function, because their output shape depends on the runtime
values (`unique(c(1, 1))` outputs a length-1 vector, but
`unique(c(1, 2))` outputs a length-2 vector). Other examples are
`which(x > 0)` or `x[x > 0]`.

While this restriction is less ergonomic, it means the compiler knows
more about your program and your compiled executable will run faster.
The cost is that some operations require rethinking, and a few cannot
currently be expressed inside
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md) at all.

For many operations, you can work around the restriction by keeping
shapes fixed and using a logical mask – the same idea presented in the
*Padding* section of the efficiency vignette. We cover some common
patterns below. For operations we cannot yet handle in-graph (such as
[`which()`](https://rdrr.io/r/base/which.html) or
[`unique()`](https://rdrr.io/r/base/unique.html)), you currently need to
convert the `AnvlArray` back to R, apply the operation there, and then
convert the result back to an `AnvlArray` to resume the computation. We
hope to lift the static-shape restriction in the long term and add
functions like [`which()`](https://rdrr.io/r/base/which.html) and
[`unique()`](https://rdrr.io/r/base/unique.html) to {anvl}’s eager API
to make this more ergonomic.

## The masking pattern

The usual workaround is to keep the output shape equal to the input
shape – so that everything remains known at trace time – and carry a
logical mask that tells us which positions count. Any downstream
operation is then modified to ignore the masked-out positions.

Throughout, we will use a single example vector, `x`, and compute things
that in plain R we would write as `sum(x[x > 0])`, `max(x[x > 0])`, and
so on:

``` r

library(anvl)
x <- nv_array(c(-2, 1, 3, -4, 2, -1, 5), dtype = "f32")
```

### Masked sum

For a sum over the positive entries, we don’t have to filter at all – we
can replace each non-matching entry with `0` and sum the whole vector.
Because adding `0` does nothing, the masked-out positions don’t
contribute to the total:

``` r

sum_positive <- jit(function(x) {
  nv_reduce_sum(nv_ifelse(x > 0, x, nv_fill_like(x, 0)), dims = 1L)
})

sum_positive(x)
#> AnvlArray
#>  11
#> [ CPUf32{} ]
```

This trick works because `0` is the *neutral value* of addition. The
same idea generalizes to any reduction whose operation has such a
neutral value:

| Operation                              | Neutral value                   |
|----------------------------------------|---------------------------------|
| `sum` (and additive reductions)        | `0`                             |
| `prod` (and multiplicative reductions) | `1`                             |
| `max`                                  | `-Inf` (or the dtype’s minimum) |
| `min`                                  | `+Inf` (or the dtype’s maximum) |
| `any`                                  | `FALSE`                         |
| `all`                                  | `TRUE`                          |

### Masked mean

The mean is the case where this trick alone is not enough:
`mean(x[x > 0])` divides by the number of *matching* entries, which
depends on the data. The matching count is a scalar, though, so we can
compute it inside
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md) by summing the
boolean mask, and then divide separately:

``` r

mean_positive <- jit(function(x) {
  mask <- x > 0
  total <- nv_reduce_sum(nv_ifelse(mask, x, 0), dims = 1L)
  n <- nv_reduce_sum(mask, dims = 1L)
  total / n
})

mean_positive(x)
#> AnvlArray
#>  11
#> [ CPUf32{} ]
```

Note that the divisor is the matching count, not `length(x)`. This is
the workaround for `length(x[cond])` more generally: convert the mask to
a numeric type and sum it.

### Subset assignment

The static shape restriction also prevents calls of the form
`x[mask] <- update`. In some cases, this can be replaced by
[`nv_ifelse()`](https://r-xla.github.io/anvl/reference/nv_ifelse.md).
For example, you might want to replace all values `< 1` with 1:

``` r

nv_ifelse(x < 1, 1, x)
#> AnvlArray
#>  1
#>  1
#>  3
#>  1
#>  2
#>  1
#>  5
#> [ CPUf32{7} ]
```

What is currently not possible is to actually subset `x` so that it only
contains values `>= 1`.
