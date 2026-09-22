# Compare AbstractArray Types

Compare two abstract arrays for type equality.

An [`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md) has no
data type to compare, so it is an error here, just as
[`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html) is.
Commit it first, e.g. with
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md).

## Usage

``` r
eq_type(e1, e2)

neq_type(e1, e2)
```

## Arguments

- e1:

  ([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  First array to compare. Must not be an
  [`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md).

- e2:

  ([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  Second array to compare. Must not be an
  [`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md).

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
```
