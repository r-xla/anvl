# Primitive Greater Than or Equal

Element-wise greater than or equal comparison.

## Usage

``` r
prim_ge(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any data type. R
  values take the other operand's data type when it is in their [data
  type
  category](https://r-xla.github.io/anvl/reference/dtype_categories.md),
  and their [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md) when
  neither operand has one. An R value outside the other operand's
  category is an error, as are two R values of different storage types.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the inputs' shape and boolean data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html),
specified under [compare](https://openxla.org/stablehlo/spec#compare).
The comparison direction is `GE`.

## See also

[`nv_ge()`](https://r-xla.github.io/anvl/reference/nv_ge.md), `>=`

## Examples

``` r
# two R values: both take an R double's default data type
prim_ge(2, 1)
#> AnvlArray
#>  1
#> [ CPUbool{} ] 

# the R value is built at the array's data type instead
prim_ge(2, nv_scalar(1, "f64"))
#> AnvlArray
#>  1
#> [ CPUbool{} ] 
```
