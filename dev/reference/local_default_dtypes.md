# Set the Default Data Types

Set the default data types (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
for a scope: `local_default_dtypes()` until the calling frame exits,
`with_default_dtypes()` for the duration of `code`. Both write the
`anvl.default_dtypes` option for one backend, and change only the
categories they name.

## Usage

``` r
local_default_dtypes(dtypes, backend = NULL, envir = parent.frame())

with_default_dtypes(dtypes, code, backend = NULL)
```

## Arguments

- dtypes:

  (named [`character()`](https://rdrr.io/r/base/character.html) \| named
  [`list()`](https://rdrr.io/r/base/list.html))  
  A mapping of the data type categories (`float` and `int`) to data
  types, e.g. `c(float = "f64", int = "i32")`. Each may be a string or a
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html).
  Can also be a partial override, such as `c(float = "f64")`, in which
  case the category it does not name is left as it is.

- backend:

  (`NULL` \| `character(1)`)  
  The backend whose defaults to set. Uses
  [`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md)
  by default.

- envir:

  (`environment`)  
  The environment to scope the change to.

- code:

  An expression to evaluate with the given defaults.

## Value

`local_default_dtypes()` returns the previous values of the options it
set, invisibly. `with_default_dtypes()` returns the result of evaluating
`code`.

## Details

Inside a [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
body the defaults the program was keyed on are the *baseline* and an
override applies to its scope, so one program can use different
precisions in different parts of itself. Only that baseline is part of
the compilation cache key, so **an override in a body must not change
between calls: write it out literally rather than reading it from a
variable.** `with_default_dtypes(c(float = prec), ...)` with a `prec`
that later changes keeps serving the program traced at the first value,
exactly as a changing `dtype` argument would – and just as silently.
Nothing checks this for you.

## See also

[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md),
[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md),
[`with_backend()`](https://r-xla.github.io/anvl/dev/reference/with_backend.md)

## Examples

``` r
with_default_dtypes(c(float = "f64"), dtype(nv_array(1.5)))
#> <f64>
# A value that meets a typed array still takes that array's data type
with_default_dtypes(c(float = "f64"), dtype(nv_array(1, dtype = "f32") + 1.5))
#> <f32>
# Untyped values in one program can commit at different precisions
jit(function() {
  list(single = nv_fill(0, 2), double = with_default_dtypes(c(float = "f64"), nv_fill(0, 2)))
})()
#> $single
#> AnvlArray
#>  0
#>  0
#> [ CPUf32{2} ] 
#> 
#> $double
#> AnvlArray
#>  0
#>  0
#> [ CPUf64{2} ] 
#> 
```
