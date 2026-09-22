# 2D Convolution

Torch-style 2D convolution in NCHW layout: `x` is
`[batch, in_channels, height, width]`, `weight` is
`[out_channels, in_channels / groups, kh, kw]`, output is
`[batch, out_channels, out_h, out_w]`. Symmetric zero padding.

## Usage

``` r
nv_conv2d(
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
  `[N, C_in, H, W]`. Can be any data type; `x` and `weight` are
  [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  An R value assumes the other operand's data type, and materializes at
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when that has none either.

- weight:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  `[C_out, C_in / groups, kH, kW]`. Promoted together with `x` – see
  `x`.

- stride:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Length 1 or 2.

- padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Symmetric padding, length 1 or 2.

- dilation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Kernel dilation, length 1 or 2.

- groups:

  (`integer(1)`)  
  Grouped/depthwise convolution.

- precision:

  (`character(1)`)  
  `"highest"`, `"high"` or `"default"`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type, and shape
`[N, C_out, out_H, out_W]`.

## See also

[`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md),
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md).

## Examples

``` r
# one batch, one channel, 4x4, convolved with a 3x3 kernel
x <- nv_array(1:16, shape = c(1, 1, 4, 4), dtype = "f32")
weight <- nv_fill(1, shape = c(1, 1, 3, 3), dtype = "f32")
nv_conv2d(x, weight)
#> AnvlArray
#> (1,1,.,.) =
#>  54 90
#>  63 99
#> [ CPUf32{1,1,2,2} ] 

# two output channels give a result with two channels
weight2 <- nv_fill(1, shape = c(2, 1, 3, 3), dtype = "f32")
shape(nv_conv2d(x, weight2))
#> [1] 1 2 2 2
```
