# Primitive Psigamma

Element-wise psigamma function: the `deriv`-th derivative of the digamma
function, i.e. the `(deriv + 1)`-th derivative of the log-gamma
function. Both `x` and `deriv` must have the same shape; `deriv`
typically holds non-negative integer values.

## Usage

``` r
prim_psigamma(x, deriv)
```

## Arguments

- x, deriv:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same shape. Can be any float data type. `x` and
  `deriv` must have the same data type. An R value among them assumes
  the data type of the others when it is in its [data type
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

[`nv_psigamma()`](https://r-xla.github.io/anvl/dev/reference/nv_psigamma.md)

## Examples

``` r
# both operands are floats, as the primitive requires
x <- nv_array(c(0.5, 1, 2))
deriv <- nv_array(c(1, 1, 2))
prim_psigamma(x, deriv)
#> AnvlArray
#>   4.9348
#>   1.6449
#>  -0.4041
#> [ CPUf32{3} ] 
```
