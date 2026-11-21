# 1D Convolution

1D convolution like torch's `nnf_conv1d()`, but with the axes in reverse
order: width, channels, batch.

## Usage

``` r
nv_conv1d(
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

  (`integer(1)`)  
  Step between neighbouring window positions.

- padding:

  (`integer(1)`)  
  Number of zeros added on each side of the spatial axis.

- dilation:

  (`integer(1)`)  
  Spacing between the kernel's elements.

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

- `x`: `(W, C_in, N)`

- `kernel`: `(kW, C_in / groups, C_out)`

- output: `(W_out, C_out, N)`, where
  `W_out = floor((W + 2 * padding - dilation * (kW - 1) - 1) / stride) + 1`

## See also

[`nv_conv2d()`](https://r-xla.github.io/anvl/reference/nv_conv2d.md),
[`nv_conv3d()`](https://r-xla.github.io/anvl/reference/nv_conv3d.md),
[`prim_convolution()`](https://r-xla.github.io/anvl/reference/prim_convolution.md).

## Examples

``` r
# width 5, one channel, a batch of one, convolved with a width-3 kernel
x <- nv_array(1:5, shape = c(5, 1, 1), dtype = "f32")
kernel <- nv_array(c(1, 0, -1), shape = c(3, 1, 1), dtype = "f32")
nv_conv1d(x, kernel)
#> AnvlArray
#> (1,.,.) =
#>  -2
#> 
#> (2,.,.) =
#>  -2
#> 
#> (3,.,.) =
#>  -2
#> [ CPUf32{3,1,1} ] 

# `padding = 1` keeps the input width, `stride = 2` visits every other
# window position
nv_conv1d(x, kernel, padding = 1L)
#> AnvlArray
#> (1,.,.) =
#>  -2
#> 
#> (2,.,.) =
#>  -2
#> 
#> (3,.,.) =
#>  -2
#> 
#> (4,.,.) =
#>  -2
#> 
#> (5,.,.) =
#>  4
#> [ CPUf32{5,1,1} ] 
nv_conv1d(x, kernel, stride = 2L)
#> AnvlArray
#> (1,.,.) =
#>  -2
#> 
#> (2,.,.) =
#>  -2
#> [ CPUf32{2,1,1} ] 
```
