# Vectorize a Function

**\[experimental\]**

Return a new function that maps `f` over an axis of its arguments: it
applies `f` to every slice along that axis, by default the last, so to
every column of a matrix. This is in the spirit of
[`base::Vectorize()`](https://rdrr.io/r/base/Vectorize.html), but unlike
it, `vectorize()` maps over slices, not elements. Instead of looping
over the slices, the function is rewritten to operate on the whole batch
at once.

This is analogous to `vmap` found in other array computing frameworks.

See the
[Vectorization](https://r-xla.github.io/anvl/articles/vectorize.html)
article for more details.

## Usage

``` r
vectorize(f, in_axes = -1L, out_axes = -1L, randomness = "independent")
```

## Arguments

- f:

  (`function`)  
  Function to vectorize.

- in_axes:

  (`integer(1)` \| named [`list()`](https://rdrr.io/r/base/list.html))  
  Specify the batch axis. If `in_axes` is a single integer, it maps over
  that axis of every argument that is an array with at least one axis,
  an R array as well as an
  [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md).
  Every application gets the other arguments as they are, such as a
  scalar, which has no axis to map over. If it is a named list, it must
  be compatible with the argument tree of the function. The values of
  the tree must either be integers, indicating the batch axis (negative
  values count from the end), or `NULL`, indicating no batching. Values
  are applied to all leaves of the node. Arguments or named elements the
  list leaves out are not batched either. Every argument that the list
  gives an axis must have it, so it cannot give one to a scalar.

- out_axes:

  (`integer(1)` \| [`list()`](https://rdrr.io/r/base/list.html))  
  Along which axis to stack the outputs of `f`. Specified like
  `in_axes`, but a tree follows the structure of what `f` returns.
  `NULL` returns an output as it is, which only works for one that does
  not differ between the slices: it depends neither on the arguments
  mapped over nor, with `randomness = "independent"`, on random numbers.

- randomness:

  (`character(1)`)  
  How the slices draw random numbers. With `"independent"` (the
  default), they draw independent random numbers: every slice draws the
  numbers a loop applying `f` to one slice after the other draws for it,
  and the random state advances as after that loop, so replacing such a
  loop with `vectorize()` gives the same results. With `"common"`, they
  draw common random numbers: every slice draws those a single
  application of `f` draws, and the random state advances as after that
  application.

## Value

(`function`)  
Has the same formals as `f` and must be called inside
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md). It returns
what `f` returns, with the batch axis added to every output where
`out_axes` puts it; an output whose entry in `out_axes` is `NULL` is
returned as it is.

## Differences from base R

Unlike [`base::Vectorize()`](https://rdrr.io/r/base/Vectorize.html) and
the apply functions:

- `vectorize()` maps over slices along an axis, not over elements: over
  the columns of a matrix by default, not over its numbers.

- Arguments are not recycled, as elsewhere in anvl: every argument
  mapped over must have the same number of slices.

- Results are not flattened: every output keeps the shape `f` gives it,
  with the batch axis added, where
  [`sapply()`](https://rdrr.io/r/base/lapply.html) and
  [`apply()`](https://rdrr.io/r/base/apply.html) would turn a matrix
  result into a column.

- A negative axis counts from the end: the default `in_axes = -1L` maps
  over the last axis, while `x[-1]` leaves out the first element.

## See also

[`rule_vectorize()`](https://r-xla.github.io/anvl/dev/reference/rule_vectorize.md),
[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md),
the
[Vectorization](https://r-xla.github.io/anvl/articles/vectorize.html)
article

## Examples

``` r
# f is written for a single vector
f <- function(x, y) sum(x * y)
x <- nv_array(matrix(1:6, nrow = 2L), dtype = "f32")
y <- nv_array(c(1, 2), dtype = "f32")

# one dot product per column of x, against the same y
jit(vectorize(f, in_axes = list(x = -1L)))(x, y)
#> AnvlArray
#>   5
#>  11
#>  17
#> [ CPUf32{3} ] 

# map over the rows instead: each row divided by its sum
jit(vectorize(function(x) x / sum(x), in_axes = 1L, out_axes = 1L))(x)
#> AnvlArray
#>  0.1111 0.3333 0.5556
#>  0.1667 0.3333 0.5000
#> [ CPUf32{2,3} ] 

# a nested list in in_axes maps over a part of a nested argument: the
# columns of params$w, while params$b is the same for every column
affine <- function(params, y) sum(params$w * y) + params$b
params <- list(w = x, b = nv_scalar(10, dtype = "f32"))
jit(vectorize(affine, in_axes = list(params = list(w = -1L))))(params, y)
#> AnvlArray
#>  15
#>  21
#>  27
#> [ CPUf32{3} ] 

# out_axes as a list: stack the dot products, and return the norm of y,
# which does not depend on the columns, as it is
dot_norm <- function(x, y) list(dot = sum(x * y), norm = sqrt(sum(y^2)))
jit(vectorize(
  dot_norm,
  in_axes = list(x = -1L),
  out_axes = list(dot = -1L, norm = NULL)
))(x, y)
#> $dot
#> AnvlArray
#>   5
#>  11
#>  17
#> [ CPUf32{3} ] 
#> 
#> $norm
#> AnvlArray
#>  2.2361
#> [ CPUf32{} ] 
#> 

# vectorize() composes with other transformations, e.g. one gradient of a
# loss per column of x
loss <- function(w, x) sum((x * w)^2)
jit(vectorize(gradient(loss, wrt = "w"), in_axes = list(x = -1L)))(y, x)
#> $w
#> AnvlArray
#>    2  18  50
#>   16  64 144
#> [ CPUf32{2,3} ] 
#> 

# and with itself: every column of x against every column of x, which is
# t(x) %*% x
f_all <- vectorize(vectorize(f, in_axes = list(x = -1L)), in_axes = list(y = -1L))
jit(f_all)(x, x)
#> AnvlArray
#>   5 11 17
#>  11 25 39
#>  17 39 61
#> [ CPUf32{3,3} ] 

# every column gets noise of its own, the noise a loop over the columns
# would draw
noisy <- function(x) x + nv_rnorm(shape = 2L)
nv_set_seed(1L)
jit(vectorize(noisy))(x)
#> AnvlArray
#>  1.8863 4.0593 5.4281
#>  0.7194 4.7140 7.4333
#> [ CPUf32{2,3} ] 
# or the same noise for every column
jit(vectorize(noisy, randomness = "common"))(x)
#> AnvlArray
#>  -0.7255  1.2745  3.2745
#>   2.4136  4.4136  6.4136
#> [ CPUf32{2,3} ] 
```
