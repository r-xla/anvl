# Array-Like Objects

An `arrayish` value is anything that represents an
[`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md) or
can be converted to one.

Specifically, these values are `arrayish`:

- [`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md): a
  concrete array holding data on a device.

- [`GraphBox`](https://r-xla.github.io/anvl/reference/GraphBox.md): the
  representation of an array during tracing, e.g. inside a
  [`jit()`](https://r-xla.github.io/anvl/reference/jit.md)ted function.

- R objects:

  - `numeric(1)` and `logical(1)` which represent scalars.

  - `numeric` and `logical` R arrays.

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

(`logical(1)`)  
Whether `x` is arrayish.

## See also

[AnvlArray](https://r-xla.github.io/anvl/reference/AnvlArray.md),
[GraphBox](https://r-xla.github.io/anvl/reference/GraphBox.md)

## Examples

``` r
# AnvlArray objects are arrayish
is_arrayish(nv_array(1:4))
#> [1] TRUE

# R arrays and literals are arrayish by default
is_arrayish(array(1:4), convert_ok = TRUE)
#> [1] TRUE
is_arrayish(array(1:4), convert_ok = FALSE)
#> [1] FALSE

# length 1 vectors
is_arrayish(1.5, convert_ok = FALSE)
#> [1] FALSE
is_arrayish(1.5, convert_ok = TRUE)
#> [1] TRUE
```
