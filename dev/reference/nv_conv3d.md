# 3D Convolution

Torch-style 3D convolution in NCDHW layout. `x` is
`[batch, in_channels, depth, height, width]`, `kernel` is
`[out_channels, in_channels / groups, kD, kH, kW]`. Asymmetric padding
(e.g. causal temporal padding) is available via
[`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md).

## Usage

``` r
nv_conv3d(
  x,
  kernel,
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
  `[N, C_in, D, H, W]`. Can be any data type; `x` and `kernel` are
  [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  An R value assumes the other operand's data type, and materializes at
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when that has none either.

- kernel:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  `[C_out, C_in / groups, kD, kH, kW]`. Promoted together with `x` – see
  `x`.

- stride, padding, dilation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Length 1 or 3.

- groups:

  (`integer(1)`)  
  Grouped/depthwise convolution.

- precision:

  (`character(1)`)  
  `"highest"`, `"high"` or `"default"`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type, and shape
`[N, C_out, out_D, out_H, out_W]`.

## See also

[`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md),
[`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md).

## Examples

``` r
# one batch, one channel, 2x3x3, convolved with a 1x2x2 kernel
x <- nv_array(1:18, shape = c(1, 1, 2, 3, 3), dtype = "f32")
kernel <- nv_fill(1, shape = c(1, 1, 1, 2, 2), dtype = "f32")
shape(nv_conv3d(x, kernel))
#> [1] 1 1 2 2 2
```
