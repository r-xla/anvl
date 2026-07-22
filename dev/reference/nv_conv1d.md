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
  `[N, C_in, W]`.

- weight:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  `[C_out, C_in / groups, kW]`.

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

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
`[N, C_out, out_W]`.

## See also

[`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md),
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md).
