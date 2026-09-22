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

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Predicate array. Must be a boolean or an R logical, and scalar or the
  same shape as `true_value`.

- true_value, false_value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Values to select from, of the same shape. Can be any data type.
  `true_value` and `false_value` must have the same data type. An R
  value among them assumes the data type of the others when it is in its
  [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the shape of `true_value` and `false_value`, and the data type they
agreed on.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_select()`](https://r-xla.github.io/stablehlo/reference/hlo_select.html),
specified under [select](https://openxla.org/stablehlo/spec#select).

## See also

[`nv_ifelse()`](https://r-xla.github.io/anvl/dev/reference/nv_ifelse.md)

## Examples

``` r
# the result takes the branches' data type; `pred` only selects
pred <- nv_array(c(TRUE, FALSE, TRUE))
prim_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#> AnvlArray
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
```
