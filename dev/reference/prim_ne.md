# Primitive Not Equal

Element-wise inequality comparison.

## Usage

``` r
prim_ne(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any data type. R
  values assume the other operand's data type when it is in their [data
  type category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  and their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when neither operand has one.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' shape and boolean data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html),
specified under [compare](https://openxla.org/stablehlo/spec#compare).
The comparison direction is `NE`.

## See also

[`nv_ne()`](https://r-xla.github.io/anvl/dev/reference/nv_ne.md), `!=`

## Examples

``` r
# two R values: both take an R double's default data type
prim_ne(1, 2)
#> AnvlArray
#>  1
#> [ CPUbool{} ] 

# the R value is built at the array's data type instead
prim_ne(1, nv_scalar(2, "f64"))
#> AnvlArray
#>  1
#> [ CPUbool{} ] 
```
