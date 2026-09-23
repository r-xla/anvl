# Primitive Clamp

Clamps every element of `x` to the range `[min, max]`.

## Usage

``` r
prim_clamp(x, min, max)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array to clamp. Can be any data type. `x`, `min` and `max` must
  have the same data type. An R value among them assumes the data type
  of the others when it is in its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

- min, max:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Lower and upper bound. Each must be scalar or the same shape as `x`,
  and shares its data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s shape and the data type the operands agreed on.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_clamp()`](https://r-xla.github.io/stablehlo/reference/hlo_clamp.html),
specified under [clamp](https://openxla.org/stablehlo/spec#clamp).

## See also

[`nv_clamp()`](https://r-xla.github.io/anvl/dev/reference/nv_clamp.md)

## Examples

``` r
x <- nv_array(c(-1, 0.5, 2))
# the R bounds take x's data type
prim_clamp(x, 0, 1)
#> AnvlArray
#>  0.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 

# an integer array takes integer bounds
prim_clamp(nv_array(1:5), 0L, 3L)
#> AnvlArray
#>  1
#>  2
#>  3
#>  3
#>  3
#> [ CPUi32{5} ] 

# the f64 bound settles it: x and max are built at f64 too
prim_clamp(1, nv_scalar(0, "f64"), 2)
#> AnvlArray
#>  1
#> [ CPUf64{} ] 
```
