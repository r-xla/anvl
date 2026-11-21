# 3D Convolution

3D convolution like torch's `nnf_conv3d()`, but with the axes in reverse
order: width, height, depth, channels, batch.

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

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input with the shape given under Shapes. Can be any data type; `x` and
  `kernel` are [promoted to a common data
  type](https://r-xla.github.io/anvl/reference/nv_promote_to_common.md).
  An R value assumes the other operand's data type, and materializes at
  its [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md) when
  that has none either.

- kernel:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Kernel with the shape given under Shapes. Promoted together with `x` –
  see `x`.

- stride:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step between neighbouring window positions, as `(W, H, D)` or one
  value for all three spatial axes.

- padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Number of zeros added on each side of each spatial axis, as
  `(W, H, D)` or one value for all three.

- dilation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Spacing between the kernel's elements, as `(W, H, D)` or one value for
  all three.

- groups:

  (`integer(1)`)  
  Number of groups the channels are split into: the `g`-th block of
  input channels is convolved with the `g`-th block of output channels
  only. It must divide `C_in` and `C_out`, and `groups = C_in` is a
  depthwise convolution.

- precision:

  (`character(1)`)  
  `"highest"`, `"high"` or `"default"`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the operands' common data type and the shape given under Shapes.

## Shapes

- `x`: `(W, H, D, C_in, N)`

- `kernel`: `(kW, kH, kD, C_in / groups, C_out)`

- output: `(W_out, H_out, D_out, C_out, N)`, with each spatial axis as
  in
  [`nv_conv1d()`](https://r-xla.github.io/anvl/reference/nv_conv1d.md)

## See also

[`nv_conv1d()`](https://r-xla.github.io/anvl/reference/nv_conv1d.md),
[`nv_conv2d()`](https://r-xla.github.io/anvl/reference/nv_conv2d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/reference/prim_convolution.md).

## Examples

``` r
# a 3x3x2 volume with one channel, in a batch of one, convolved with a
# 2x2x1 kernel
x <- nv_array(1:18, shape = c(3, 3, 2, 1, 1), dtype = "f32")
kernel <- nv_fill(1, shape = c(2, 2, 1, 1, 1), dtype = "f32")
shape(nv_conv3d(x, kernel))
#> [1] 2 2 2 1 1
```
