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

Use `is_arrayish()` to check whether a value is arrayish.

The group words the parameter descriptions use are listed below;
[`dtypes`](https://r-xla.github.io/anvl/dev/reference/dtypes.md) gives
the categories they are built from, and
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
the default an R value materializes at.

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

## Details

During [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md),
[`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md) is
also arrayish, but it is simply the trace-time representation of an
`AnvlArray`.

## Data Type Vocabulary

Where a page says which data types an argument takes, it names a group
of them with a single word:

- *any data type* – all of them: `bool`, the signed and unsigned
  integers, and the floats.

- *numeric* – signed integer, unsigned integer and float.

- *integer* – signed and unsigned integer.

- *integerish* – boolean and integer, signed or unsigned.

- *signed numeric* – signed integer and float.

- *float* – the whole float category: `f32` and `f64`.

- *boolean* – `bool`, the only member of its category.

## See also

[AnvlArray](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
[GraphBox](https://r-xla.github.io/anvl/dev/reference/GraphBox.md)

## Examples

``` r
# AnvlArray objects are arrayish
is_arrayish(nv_array(1:4))
#> [1] TRUE

# scalar R literals are arrayish by default
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

# length 1 vectors
is_arrayish(1.5, convert_ok = FALSE)
#> [1] FALSE
is_arrayish(1.5, convert_ok = TRUE)
#> [1] TRUE
```
