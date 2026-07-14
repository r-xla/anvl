# Primitive Convolution

General N-D windowed convolution, lowering to StableHLO's `convolution`
op. Dimension numbers are given 1-based (anvl convention) and converted
to StableHLO's 0-based layout internally. Most users want
[`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md)
/
[`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md)
/
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md)
instead.

## Usage

``` r
prim_convolution(
  lhs,
  rhs,
  input_batch_dimension,
  input_feature_dimension,
  input_spatial_dimensions,
  kernel_input_feature_dimension,
  kernel_output_feature_dimension,
  kernel_spatial_dimensions,
  output_batch_dimension,
  output_feature_dimension,
  output_spatial_dimensions,
  window_strides,
  padding,
  lhs_dilation,
  rhs_dilation,
  feature_group_count = 1L,
  batch_group_count = 1L,
  precision = "highest"
)
```

## Arguments

- lhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input, e.g. `[batch, channels, *spatial]`.

- rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Kernel, e.g. `[out_ch, in_ch/groups, *spatial]`.

- input_batch_dimension, input_feature_dimension:

  (`integer(1)`)  
  1-based batch/feature dimension of `lhs`.

- input_spatial_dimensions:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  1-based spatial dims of `lhs`.

- kernel_input_feature_dimension, kernel_output_feature_dimension:

  (`integer(1)`)  
  1-based input/output feature dimension of `rhs`.

- kernel_spatial_dimensions:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  1-based spatial dims of `rhs`.

- output_batch_dimension, output_feature_dimension:

  (`integer(1)`)  
  1-based batch/feature dimension of the output.

- output_spatial_dimensions:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  1-based spatial dims of the output.

- window_strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Stride per spatial dim.

- padding:

  (`matrix`)  
  `[n_spatial, 2]` of `(low, high)` padding.

- lhs_dilation, rhs_dilation:

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
