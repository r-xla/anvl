# Compare AbstractArray Types

Compare two abstract arrays for type equality.

## Usage

``` r
eq_type(e1, e2)

neq_type(e1, e2)
```

## Arguments

- e1:

  ([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  First array to compare.

- e2:

  ([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  Second array to compare.

## Value

`logical(1)` - `TRUE` if the arrays are equal, `FALSE` otherwise.

## Examples

``` r
a <- nv_aval("f32", c(2L, 3L))
b <- nv_aval("f32", c(2L, 3L))

# Same dtype and shape
eq_type(a, b)
#> [1] TRUE

# Different dtype
eq_type(a, nv_aval("i32", c(2L, 3L)))
#> [1] FALSE

# Different shape
eq_type(a, nv_aval("f32", c(3L, 2L)))
#> [1] FALSE

# neq_type is the negation of eq_type
neq_type(a, b)
#> [1] FALSE
```
