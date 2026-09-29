# Compare AbstractArray Types

Compare two abstract arrays for type equality.

An [`RData`](https://r-xla.github.io/anvl/reference/RData.md) has no
data type to compare, so it is an error here, just as
[`dtype()`](https://r-xla.github.io/xlamisc/reference/dtype.html) is.
Give it a data type first, e.g. with
[`nv_convert()`](https://r-xla.github.io/anvl/reference/nv_convert.md).

## Usage

``` r
eq_type(e1, e2)

neq_type(e1, e2)
```

## Arguments

- e1:

  ([`AbstractArray`](https://r-xla.github.io/anvl/reference/AbstractArray.md))  
  First array to compare. Must not be an
  [`RData`](https://r-xla.github.io/anvl/reference/RData.md).

- e2:

  ([`AbstractArray`](https://r-xla.github.io/anvl/reference/AbstractArray.md))  
  Second array to compare. Must not be an
  [`RData`](https://r-xla.github.io/anvl/reference/RData.md).

## Value

(`logical(1)`)

## Examples

``` r
a <- nv_aval("f32", c(2L, 3L))
b <- nv_aval("f32", c(2L, 3L))

# same dtype and shape
eq_type(a, b)
#> [1] TRUE

# different dtype
eq_type(a, nv_aval("i32", c(2L, 3L)))
#> [1] FALSE

# different shape
eq_type(a, nv_aval("f32", c(3L, 2L)))
#> [1] FALSE

# neq_type is the negation of eq_type
neq_type(a, b)
#> [1] FALSE

# an RData has no data type, so it cannot be compared
r <- RData(c(2L, 3L), "double")
try(eq_type(a, r))
#> Error : `eq_type()` is undefined for an <RData>.
#> ℹ An R value has no data type of its own until it is used, so there is nothing
#>   to compare.
#> ℹ Give it one explicitly with `nv_convert()`.
try(neq_type(r, r))
#> Error : `eq_type()` is undefined for an <RData>.
#> ℹ An R value has no data type of its own until it is used, so there is nothing
#>   to compare.
#> ℹ Give it one explicitly with `nv_convert()`.
```
