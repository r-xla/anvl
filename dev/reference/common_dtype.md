# Type Promotion Rules

Compute the common data type.

Two integer data types meet at one that holds every value of both: a
signed and an unsigned one at the narrowest signed data type wide enough
for the unsigned side (`ui8` and `i8` at `i16`, `ui32` and `i32` at
`i64`). `ui64` is the exception – no signed data type holds it, and an
integer does not become a float on its own – so `ui64` and a signed
integer have no common data type and the pair is an error. Convert one
side with
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
to decide what they meet at.

See the *Type Promotion* article for more information.

## Usage

``` r
common_dtype(lhs_dtype, rhs_dtype)
```

## Arguments

- lhs_dtype, rhs_dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  The two data types.

## Value

([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
The narrowest common data type.

## Examples

``` r
common_dtype("i32", "f32")
#> <f32>
common_dtype("i32", "i64")
#> <i64>
try(common_dtype("ui64", "i8"))
#> Error : "ui64" and "i8" have no common data type.
#> ✖ No integer data type holds every value of both, and an integer does not
#>   become a float on its own.
#> ℹ Convert one of them with `nv_convert()` -- "f64" holds both, exactly up to
#>   2^53.
```
