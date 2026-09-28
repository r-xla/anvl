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

Below, we configure every backend to use the default `f64` for floats
and `i64` for integers:

    options(anvl.default_dtypes = c(float = "f64", int = "i64"))

You can also only change the float dtype, leaving the backend's default
integer dtype unchanged:

    options(anvl.default_dtypes = c(float = "f64"))

It is also possible to specify the defaults per backend. Below, every
backend uses `f64` for floats, and `"pjrt"` also uses `i64` for
integers:

    options(anvl.default_dtypes = list(
      float = "f64",
      pjrt = list(int = "i64")
    ))

An entry that names a backend wins over the categories beside it.

When the option is not set, the `ANVL_DEFAULT_DTYPES` environment
variable (read once, when anvl is loaded) is used instead, written as
`category=dtype` pairs that apply to every backend, e.g.
`ANVL_DEFAULT_DTYPES="float=f64,int=i64"`.

The defaults decide only what a value becomes when *nothing else does*:
an R value that meets a typed array of its own category still takes that
array's data type, whatever the default (see the [Data Types and
Promotion
Rules](https://r-xla.github.io/anvl/articles/type-promotion.html)
article). The data type you name is taken on trust, so one that does not
fit is an error where the data is allocated or the program compiled
rather than where it is set. Which ones fit is the backend's own
business – see the *Supported data types* section of
[`AnvlBackendPjrt()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackendPjrt.md)
and of
[`AnvlBackendQuickr()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackendQuickr.md),
which has only `f64`, `i32` and `bool`, so both `"f32"` and `"i64"` are
errors there. A compiled program is keyed on the defaults it was
compiled under, so changing them never serves a stale program.

## Usage

``` r
default_dtypes()

default_float()

default_int()
```

## Value

`default_dtypes()`: (named `list`)  
Elements `float` and `int`, each a
[`DataType`](https://r-xla.github.io/tengen/reference/DataType.html).

`default_float()`, `default_int()`:
([`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
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
