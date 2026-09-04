# Array-like Objects

A `arrayish` value is anything that represents an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
or can be converted to one.

Specifically, these values are `arrayish`:

- [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md):
  a concrete array holding data on a device.

- R objects:

  - `numeric(1)` and `logical(1)` which represent scalars.

  - `numeric` and `logical` R arrays.

- [`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md):
  this is how dynamic
  [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)s
  are represented during
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

Use `is_arrayish()` to check whether a value is arrayish.

## Usage

``` r
is_arrayish(x, convert_ok = TRUE)
```

## Arguments

- x:

  (`any`)  
  Object to check.

- convert_ok:

  (`logical(1)`)  
  Whether to accept `numeric(1)` and `logical(1)` and R arrays of type
  `numeric` and `logical`.

## Value

`logical(1)`

## See also

[AnvlArray](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
[GraphBox](https://r-xla.github.io/anvl/dev/reference/GraphBox.md)

## Examples

``` r
# AnvlArrays are arrayish
is_arrayish(nv_array(1:4))
#> [1] TRUE

# Scalar R literals are arrayish by default
is_arrayish(1.5)
#> [1] TRUE
# R arrays are arrayish by default
is_arrayish(array(1.5))
#> [1] TRUE

# R arrays
is_arrayish(array(1:4), convert_ok = TRUE)
#> [1] TRUE
is_arrayish(array(1:4), convert_ok = FALSE)
#> [1] FALSE

# Length 1 vectors
is_arrayish(1.5, convert_ok = FALSE)
#> [1] FALSE
is_arrayish(1.5, convert_ok = TRUE)
#> [1] TRUE
```
