# Primitive Ifelse

Element-wise selection based on a boolean predicate, like R's
[`ifelse()`](https://rdrr.io/r/base/ifelse.html). For each element,
returns the corresponding element from `true_value` where `pred` is
`TRUE` and from `false_value` where `pred` is `FALSE`.

## Usage

``` r
prim_ifelse(pred, true_value, false_value)
```

## Arguments

- pred:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  of boolean type)  
  Predicate array. Must be scalar or have the same shape as
  `true_value`.

- true_value, false_value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Values to select from. Must have the same dtype and shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same dtype and shape as `true_value`.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_select()`](https://r-xla.github.io/stablehlo/reference/hlo_select.html).

## See also

[`nv_ifelse()`](https://r-xla.github.io/anvl/dev/reference/nv_ifelse.md)

## Examples

``` r
pred <- nv_array(c(TRUE, FALSE, TRUE))
prim_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#> AnvlArray
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
```
