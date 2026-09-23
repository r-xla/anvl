# Primitive Arctangent 2

Element-wise atan2 operation: the angle (in radians) between the
positive x-axis and the point `(x, y)`.

The operands are named `y` and `x`, in that order, after
[`base::atan2()`](https://rdrr.io/r/base/Trig.html), rather than `lhs` /
`rhs`.

## Usage

``` r
prim_atan2(y, x)
```

## Arguments

- y, x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Ordinate and abscissa of the point, of the same shape. Can be any
  float data type. `y` and `x` must have the same data type. An R value
  among them assumes the data type of the others when it is in its [data
  type category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  and its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' shape and data type.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_atan2()`](https://r-xla.github.io/stablehlo/reference/hlo_atan2.html),
specified under [atan2](https://openxla.org/stablehlo/spec#atan2).

## See also

[`nv_atan2()`](https://r-xla.github.io/anvl/dev/reference/nv_atan2.md)

## Examples

``` r
# two R values: both take an R double's default data type
prim_atan2(1, 1)
#> AnvlArray
#>  0.7854
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_atan2(1, nv_scalar(1, "f64"))
#> AnvlArray
#>  0.7854
#> [ CPUf64{} ] 
```
