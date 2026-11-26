# Parameter Kinds

How a parameter of a primitive changes when its operands gain a trailing
batch axis of size `size`, for
[`rule_vectorize()`](https://r-xla.github.io/anvl/dev/reference/rule_vectorize.md).
A parameter that names axes needs none of these: the batch axis comes
after all of them, so they keep their positions.

- `vec_shape()`: a shape, which gains the batch axis (`c(shape, size)`).

- `vec_axis_map()`: for each axis of the output, the operand axis it
  takes, such as a permutation; the batch axis maps to itself
  (`c(axes, length(axes) + 1L)`).

- `vec_per_axis()`: one entry per axis of an operand, with `value` for
  the batch axis.

## Usage

``` r
vec_shape()

vec_axis_map()

vec_per_axis(value)
```

## Arguments

- value:

  (`integer(1)` \| `function`)  
  The entry for the batch axis, or a function of `size` that returns it.

## Value

(`anvl_param_kind`)  
A function of the parameter's value and `size` that returns its new
value.

## Examples

``` r
vec_shape()(c(2L, 3L), size = 5L)
#> [1] 2 3 5
vec_axis_map()(c(2L, 1L), size = 5L)
#> [1] 2 1 3
vec_per_axis(0L)(c(1L, 2L), size = 5L)
#> [1] 1 2 0
```
