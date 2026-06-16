# Primitive Clamp

Clamps every element of `operand` to the range `[min_val, max_val]`,
i.e. `max(min_val, min(operand, max_val))`.

## Usage

``` r
prim_clamp(min_val, operand, max_val)
```

## Arguments

- min_val:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Minimum value. Must be scalar or the same shape as `operand`.

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- max_val:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Maximum value. Must be scalar or the same shape as `operand`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type and shape as `operand`. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_clamp()`](https://r-xla.github.io/stablehlo/reference/hlo_clamp.html).

## See also

[`nv_clamp()`](https://r-xla.github.io/anvl/dev/reference/nv_clamp.md)

## Examples

``` r
x <- nv_array(c(-1, 0.5, 2))
prim_clamp(nv_scalar(0), x, nv_scalar(1))
#> AnvlArray
#>  0.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 
```
