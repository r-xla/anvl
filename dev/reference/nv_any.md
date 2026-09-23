# Any Reduction

Performs logical OR along the specified axes. Returns `TRUE` if any
element is `TRUE`.

## Usage

``` r
nv_any(x, axes = NULL, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Must be a boolean or an R logical.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce over. Negative values count from the end, i.e. `-1`
  refers to the last axis. If `NULL` (default), reduces over all axes.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced axes: removed from the output shape if
  `TRUE`, set to 1 if `FALSE`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the boolean data type. The shape is the input's with the reduced
axes removed (`drop = TRUE`) or set to 1 (`drop = FALSE`).

## The [`any()`](https://rdrr.io/r/base/any.html) generic

[`any()`](https://rdrr.io/r/base/any.html) reduces over all axes and,
like [`base::any()`](https://rdrr.io/r/base/any.html), takes several
data arguments: `any(x, y)` asks about both arrays. It is *logical*, so
– unlike base R – a non-boolean argument is an error rather than a
comparison against zero. Beyond base R, named arguments are passed on,
so `any(x, axes = 1L)` reduces a single axis – but only when `x` is the
only data argument.

## See also

[`prim_any()`](https://r-xla.github.io/anvl/dev/reference/prim_any.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
# no axes given: reduce over all of them
nv_any(x)
#> AnvlArray
#>  1
#> [ CPUbool{} ] 

# reducing axis 1 removes it, drop = FALSE keeps it at size 1
nv_any(x, axes = 1L)
#> AnvlArray
#>  1
#>  1
#> [ CPUbool{2} ] 
nv_any(x, axes = 1L, drop = FALSE)
#> AnvlArray
#>  1 1
#> [ CPUbool{1,2} ] 
```
