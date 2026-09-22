# Primitive Polygamma

Element-wise polygamma function: the `(n+1)`-th derivative of the
log-gamma function. Both `n` and `x` must have the same shape; `n`
typically holds non-negative integer values.

## Usage

``` r
prim_polygamma(n, x)
```

## Arguments

- n, x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same shape. Can be any float data type. `n` and `x`
  must have the same data type. An R value among them assumes the data
  type of the others when it is in its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
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
[`hlo_polygamma()`](https://r-xla.github.io/stablehlo/reference/hlo_polygamma.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.polygamma](https://openxla.org/stablehlo/generated/chlo#chlopolygamma_chlopolygammaop).

## See also

[`nv_polygamma()`](https://r-xla.github.io/anvl/dev/reference/nv_polygamma.md)

## Examples

``` r
# both operands are floats, as the primitive requires
n <- nv_array(c(1, 1, 2))
x <- nv_array(c(0.5, 1, 2))
prim_polygamma(n, x)
#> AnvlArray
#>   4.9348
#>   1.6449
#>  -0.4041
#> [ CPUf32{3} ] 
```
