# 2D Convolution

Torch-style 2D convolution in NCHW layout: `input` is
`[batch, in_channels, height, width]`, `weight` is
`[out_channels, in_channels / groups, kh, kw]`, output is
`[batch, out_channels, out_h, out_w]`. Symmetric zero padding.

## Usage

``` r
nv_conv2d(
  input,
  weight,
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L,
  precision = "highest"
)
```

## Arguments

- input:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  `[N, C_in, H, W]`.

- weight:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  `[C_out, C_in / groups, kH, kW]`.

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

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
`[N, C_out, out_H, out_W]`.

## See also

[`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md),
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md).
