# Default Data Types

The default data types for the [active
backend](https://r-xla.github.io/anvl/dev/reference/active_backend.md).
They are the data types an R value settles on when it meets no typed
array, e.g. `nv_array(1)` or `prim_exp(1)`.

`default_dtypes()` reports both categories at once; `default_float()`
and `default_int()` report one each.

Each backend registers its own – `f32` / `i32` for `"pjrt"`, `f64` /
`i32` for `"quickr"` – and the `anvl.default_dtypes` option overrides
them.
[`local_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md)
and
[`with_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md)
set the option for a scope.

The option maps the categories `float` and `int` to data types, e.g.
`c(float = "f64", int = "i64")`, and applies to every backend. It may
name only one of them, like `c(float = "f64")`, in which case the other
category keeps the backend's default. To set the defaults of a single
backend, give a list with an entry named after that backend instead:
`list(float = "f64", pjrt = list(int = "i64"))` sets `f64` for every
backend and additionally `i64` for `"pjrt"`. An entry that names a
backend wins over the categories beside it.

When the option is not set, the `ANVL_DEFAULT_DTYPES` environment
variable (read once, when anvl is loaded) is used instead, written as
`category=dtype` pairs that apply to every backend, e.g.
`ANVL_DEFAULT_DTYPES="float=f64,int=i64"`.

## Usage

``` r
default_dtypes()

default_float()

default_int()
```

## Value

`default_dtypes()`: (named `list`)  
Elements `float` and `int`, each a
[`DataType`](https://r-xla.github.io/xlamisc/reference/DataType.html).

`default_float()`, `default_int()`:
([`DataType`](https://r-xla.github.io/xlamisc/reference/DataType.html))  
The default of one category.

## See also

[`local_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md),
[`with_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md),
[`with_dtypes()`](https://r-xla.github.io/anvl/dev/reference/with_dtypes.md)

## Examples

``` r
with_backend("quickr", default_dtypes())
#> $float
#> <f64>
#> 
#> $int
#> <i32>
#> 
with_backend("pjrt", default_dtypes())
#> $float
#> <f32>
#> 
#> $int
#> <i32>
#> 
default_dtypes()
#> $float
#> <f32>
#> 
#> $int
#> <i32>
#> 
default_float()
#> <f32>
default_int()
#> <i32>
dtype(nv_array(1.5))
#> <f32>
```
