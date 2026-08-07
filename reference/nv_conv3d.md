# 3D Convolution

Torch-style 3D convolution in NCDHW layout. `x` is
`[batch, in_channels, depth, height, width]`, `weight` is
`[out_channels, in_channels / groups, kD, kH, kW]`. Asymmetric padding
(e.g. causal temporal padding) is available via
[`prim_convolution()`](https://r-xla.github.io/anvl/reference/prim_convolution.md).

## Usage

``` r
nv_conv3d(
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

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  `[N, C_in, H, W]`.

- weight:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  `[C_out, C_in / groups, kH, kW]`.

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

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)
`[N, C_out, out_D, out_H, out_W]`.

## See also

[`nv_conv1d()`](https://r-xla.github.io/anvl/reference/nv_conv1d.md),
[`nv_conv2d()`](https://r-xla.github.io/anvl/reference/nv_conv2d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/reference/prim_convolution.md).
