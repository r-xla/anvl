# Primitive Convolution

General N-D windowed convolution, lowering to StableHLO's `convolution`
op. Axis numbers are given 1-based (anvl convention) and converted to
StableHLO's 0-based layout internally. Most users want
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
  Input, e.g. `[batch, channels, *spatial]`.

- kernel:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Kernel, e.g. `[out_ch, in_ch/groups, *spatial]`.

- input_batch_axis, input_feature_axis:

  (`integer(1)`)  
  1-based batch/feature axis of `x`.

- input_spatial_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  1-based spatial axes of `x`.

- kernel_input_feature_axis, kernel_output_feature_axis:

  (`integer(1)`)  
  1-based input/output feature axis of `kernel`.

- kernel_spatial_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  1-based spatial axes of `kernel`.

- output_batch_axis, output_feature_axis:

  (`integer(1)`)  
  1-based batch/feature axis of the output.

- output_spatial_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  1-based spatial axes of the output.

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

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
