# Primitive Addition

Adds two arrays element-wise.

## Usage

``` r
prim_add(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any data type. R
  values take the other operand's data type when it is in their [data
  type
  category](https://r-xla.github.io/anvl/dev/reference/dtype_categories.md),
  and their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when neither operand has one. An R value outside the other operand's
  category is an error, as are two R values of different storage types.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' shape and data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_add()`](https://r-xla.github.io/stablehlo/reference/hlo_add.html),
specified under [add](https://openxla.org/stablehlo/spec#add).

## See also

[`nv_add()`](https://r-xla.github.io/anvl/dev/reference/nv_add.md), `+`

## Examples

``` r
# two R values: both take an R double's default data type
prim_add(1, 2)
#> AnvlArray
#>  3
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_add(1, nv_scalar(2, "f64"))
#> AnvlArray
#>  3
#> [ CPUf64{} ] 
```
