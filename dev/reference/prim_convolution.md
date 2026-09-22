# Primitive Convolution

General N-D windowed convolution. The axis arguments say which axis of
each operand plays which role, so any layout can be described. Most
users want
[`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md)
/
[`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md)
/
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md)
instead.

## Usage

``` r
prim_convolution(
  x,
  kernel,
  input_batch_axis,
  input_feature_axis,
  input_spatial_axes,
  kernel_input_feature_axis,
  kernel_output_feature_axis,
  kernel_spatial_axes,
  output_batch_axis,
  output_feature_axis,
  output_spatial_axes,
  window_strides,
  padding,
  x_dilation,
  kernel_dilation,
  feature_group_count = 1L,
  batch_group_count = 1L,
  precision = "highest"
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input, e.g. `[batch, channels, *spatial]`. Can be any data type. `x`
  and `kernel` must have the same data type. An R value among them
  assumes the data type of the others when it is in its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

- kernel:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Kernel, e.g. `[out_ch, in_ch/groups, *spatial]`. Shares `x`'s data
  type – see `x`.

- input_batch_axis, input_feature_axis:

  (`integer(1)`)  
  Batch and feature axis of `x`.

- input_spatial_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Spatial axes of `x`.

- kernel_input_feature_axis, kernel_output_feature_axis:

  (`integer(1)`)  
  Input and output feature axis of `kernel`.

- kernel_spatial_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Spatial axes of `kernel`.

- output_batch_axis, output_feature_axis:

  (`integer(1)`)  
  Batch and feature axis of the output.

- output_spatial_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Spatial axes of the output.

- window_strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Stride per spatial axis.

- padding:

  (`matrix`)  
  `[n_spatial, 2]` of `(low, high)` padding.

- x_dilation, kernel_dilation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Input/kernel dilation.

- feature_group_count, batch_group_count:

  (`integer(1)`)  
  Grouping.

- precision:

  (`character(1)`)  
  One of `"highest"`, `"high"`, `"default"`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the data type `x` and `kernel` agreed on. Its shape is given by the
output axis arguments: the batch axis holds `x`'s batch size divided by
`batch_group_count`, the feature axis `kernel`'s output feature size,
and each spatial axis the number of window positions along it.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_convolution()`](https://r-xla.github.io/stablehlo/reference/hlo_convolution.html),
specified under
[convolution](https://openxla.org/stablehlo/spec#convolution). The axis
numbers are converted on the way down.

## See also

[`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md),
[`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md),
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md)

## Examples

``` r
# a 1-D convolution in NCW layout: one batch, one channel, width 5,
# convolved with a width-3 kernel, giving 3 window positions
x <- nv_array(1:5, shape = c(1, 1, 5), dtype = "f32")
kernel <- nv_array(c(1, 0, -1), shape = c(1, 1, 3), dtype = "f32")
prim_convolution(
  x, kernel,
  input_batch_axis = 1L, input_feature_axis = 2L, input_spatial_axes = 3L,
  kernel_output_feature_axis = 1L, kernel_input_feature_axis = 2L,
  kernel_spatial_axes = 3L,
  output_batch_axis = 1L, output_feature_axis = 2L, output_spatial_axes = 3L,
  window_strides = 1L, padding = matrix(0L, nrow = 1, ncol = 2),
  x_dilation = 1L, kernel_dilation = 1L
)
#> AnvlArray
#> (1,.,.) =
#>  -2 -2 -2
#> [ CPUf32{1,1,3} ] 
```
