# 2D Convolution

2D convolution like torch's `nnf_conv2d()`, but with the axes in reverse
order: width, height, channels, batch.

## Usage

``` r
nv_conv2d(
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
  Input with the shape given under Shapes. Can be any data type; `x` and
  `kernel` are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  An R value assumes the other operand's data type, and materializes at
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when that has none either.

- kernel:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Kernel with the shape given under Shapes. Promoted together with `x` –
  see `x`.

- stride:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step between neighbouring window positions, as `(W, H)` or one value
  for both spatial axes.

- padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Number of zeros added on each side of each spatial axis, as `(W, H)`
  or one value for both.

- dilation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Spacing between the kernel's elements, as `(W, H)` or one value for
  both.

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

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type and the shape given under Shapes.

## Shapes

- `x`: `(W, H, C_in, N)`

- `kernel`: `(kW, kH, C_in / groups, C_out)`

- output: `(W_out, H_out, C_out, N)`, with each spatial axis as in
  [`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md)

## See also

[`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md),
[`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md).

## Examples

``` r
# a 4x4 image with one channel, in a batch of one, convolved with a 3x3
# kernel
x <- nv_array(1:16, shape = c(4, 4, 1, 1), dtype = "f32")
kernel <- nv_fill(1, shape = c(3, 3, 1, 1), dtype = "f32")
nv_conv2d(x, kernel)
#> AnvlArray
#> (1,1,.,.) =
#>  54
#> 
#> (1,2,.,.) =
#>  90
#> 
#> (2,1,.,.) =
#>  63
#> 
#> (2,2,.,.) =
#>  99
#> [ CPUf32{2,2,1,1} ] 

# a batch of two 6x4 images with three channels, and a kernel with two
# output channels; the stride is 2 along the width and 1 along the height
x <- nv_fill(1, shape = c(6, 4, 3, 2), dtype = "f32")
kernel <- nv_fill(1, shape = c(3, 3, 3, 2), dtype = "f32")
shape(nv_conv2d(x, kernel, stride = c(2L, 1L)))
#> [1] 2 2 2 2
```
