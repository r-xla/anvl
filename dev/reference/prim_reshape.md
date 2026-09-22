# Primitive Reshape

Reshapes an array to a new shape without changing the underlying data.
Note that row-major order is used, which differs from R's column-major
order.

## Usage

``` r
prim_reshape(x, shape)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Must have the same number of elements as `x`. At most
  one entry may be `-1`, in which case its extent is inferred from the
  remaining entries and the number of elements of `x`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the same data type as the input and the given `shape`.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_reshape()`](https://r-xla.github.io/stablehlo/reference/hlo_reshape.html),
specified under [reshape](https://openxla.org/stablehlo/spec#reshape).

## See also

[`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)

## Examples

``` r
# the elements are reread in row-major order; the data type is untouched
x <- nv_array(1:6)
prim_reshape(x, shape = c(2, 3))
#> AnvlArray
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
```
