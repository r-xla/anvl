# Select JIT device from a function argument

Pass the result to
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)'s `device`
argument to indicate that the device should be read from a formal
argument of the function being compiled. At call time, the value of that
argument is used to derive the backend via
[`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md)
dispatch and is forwarded to the backend-specific JIT as the compilation
device.

This is intended for functions that have no dynamic array inputs from
which the backend could otherwise be detected (e.g. array constructors
like
[`prim_fill()`](https://r-xla.github.io/anvl/dev/reference/prim_fill.md)
or
[`prim_iota()`](https://r-xla.github.io/anvl/dev/reference/prim_iota.md)).

## Usage

``` r
device_arg(argname)
```

## Arguments

- argname:

  (`character(1)`)  
  Name of a formal argument of the function passed to
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

## Value

(`AnvlDeviceArg`)  
An object recognized by
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

## See also

[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md),
[`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md)

## Examples

``` r
f <- function(x) nv_scalar(1, device = x)
g <- jit(f, backend = "auto", device = device_arg("x"))
g(nv_device("cpu", "pjrt"))
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
