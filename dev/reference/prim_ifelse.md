# Primitive Ifelse

Element-wise selection based on a boolean predicate, mirroring R's
[`ifelse()`](https://rdrr.io/r/base/ifelse.html). For each element,
returns the corresponding element from `yes` where `test` is `TRUE` and
from `no` where `test` is `FALSE`.

[`prim_if()`](https://r-xla.github.io/anvl/dev/reference/prim_if.md) is
the other conditional: it mirrors R's `if` construct and branches
between two *functions*, evaluating only the selected one.

## Usage

``` r
prim_ifelse(test, yes, no)
```

## Arguments

- test:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Predicate array. Must be a boolean or an R logical, and scalar or the
  same shape as `yes`.

- yes, no:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Values to select from, of the same shape. Can be any data type. `yes`
  and `no` must have the same data type. An R value among them assumes
  the data type of the others when it is in its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the shape of `yes` and `no`, and the data type they agreed on.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_select()`](https://r-xla.github.io/stablehlo/reference/hlo_select.html),
specified under [select](https://openxla.org/stablehlo/spec#select).

## See also

[`nv_ifelse()`](https://r-xla.github.io/anvl/dev/reference/nv_ifelse.md),
[`prim_if()`](https://r-xla.github.io/anvl/dev/reference/prim_if.md)

## Examples

``` r
# the result takes the branches' data type; `test` only selects
test <- nv_array(c(TRUE, FALSE, TRUE))
prim_ifelse(test, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#> AnvlArray
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
```
