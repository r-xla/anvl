# Run a Function at Given Data Types

`with_dtypes()` wraps `f` into a function that works at the data types
`dtypes` names: on each call every array argument of a category `dtypes`
names is converted to that data type, the defaults (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
are set to the `float` / `int` entries for the duration of the call, and
every returned array of a named category is converted as well.

    nv_add_f64 <- with_dtypes(nv_add, c(float = "f64"))

A category `dtypes` does not name is left alone, in the arguments, in
the body and in the result.

## Usage

``` r
with_dtypes(f, dtypes)
```

## Arguments

- f:

  (`function`)  
  The function to wrap.

- dtypes:

  (named [`character()`](https://rdrr.io/r/base/character.html) \| named
  [`list()`](https://rdrr.io/r/base/list.html))  
  A mapping of the data type categories (`float`, `int` and `uint`) to
  data types, e.g. `c(float = "f64", int = "i64")`. Each may be a string
  or a
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html).
  A category it does not name is left as it is.

## Value

A `function` with the same arguments as `f`, a `JitFunction` if `f` was
one.

## Details

Note that `f` itself can also change the default data types, which
overrides the defaults configured by `with_dtypes()`.

## See also

[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md),
[`local_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md)

## Examples

``` r
add_f64 <- with_dtypes(nv_add, c(float = "f64"))
# An `f32` argument is converted, and the result comes back as `f64`
dtype(add_f64(nv_array(1, dtype = "f32"), 2.5))
#> <f64>
# A category that is not named is untouched
dtype(add_f64(nv_array(1L, dtype = "i32"), 2L))
#> <i32>
# `uint` is converted too, but sets no default
dtype(with_dtypes(nv_add, c(uint = "ui32"))(nv_array(1L, dtype = "ui8"), 2L))
#> <ui32>
```
