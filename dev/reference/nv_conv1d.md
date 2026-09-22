# 1D Convolution

Torch-style 1D convolution in NCW layout: `x` is
`[batch, in_channels, width]`, `weight` is
`[out_channels, in_channels / groups, kW]`, output is
`[batch, out_channels, out_w]`. Symmetric zero padding.

## Usage

``` r
nv_conv1d(
  x,
  weight,
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L,
  precision = "highest"
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  `[N, C_in, W]`. Can be any data type; `x` and `weight` are [promoted
  to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  An R value assumes the other operand's data type, and materializes at
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when that has none either.

- weight:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  `[C_out, C_in / groups, kW]`. Promoted together with `x` – see `x`.

- stride, padding, dilation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Length 1.

- groups:

  (`integer(1)`)  
  Grouped/depthwise convolution.

- precision:

  (`character(1)`)  
  `"highest"`, `"high"` or `"default"`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type, and shape `[N, C_out, out_W]`.

## See also

[`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md),
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md).

## Examples

``` r
# one batch, one channel, width 5, convolved with a width-3 kernel
x <- nv_array(1:5, shape = c(1, 1, 5), dtype = "f32")
weight <- nv_array(c(1, 0, -1), shape = c(1, 1, 3), dtype = "f32")
nv_conv1d(x, weight)
#> AnvlArray
#> (1,.,.) =
#>  -2 -2 -2
#> [ CPUf32{1,1,3} ] 

# `padding = 1` keeps the input width, `stride = 2` visits every other
# window position
nv_conv1d(x, weight, padding = 1L)
#> AnvlArray
#> (1,.,.) =
#>  -2 -2 -2 -2  4
#> [ CPUf32{1,1,5} ] 
nv_conv1d(x, weight, stride = 2L)
#> AnvlArray
#> (1,.,.) =
#>  -2 -2
#> [ CPUf32{1,1,2} ] 
```
