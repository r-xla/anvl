# Primitive Cumulative Maximum

Running maximum of array elements along a single dimension along with
the index of the last occurrence of the running maximum. At output
position `j`, the values output is `max(input[1:j])` and the indices
output is the largest `i` in `1:j` with `input[i] == values[j]`
(last-occurrence tiebreak).

## Usage

``` r
prim_cummax(x, dim)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dim:

  (`integer(1)`)  
  Dimension along which to accumulate.

## Value

`list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
values:  
The running maximum (same dtype as `x`) and the running argmax (dtype
`i32`, 1-based). Both have the same shape as `x`.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to a variadic
[`hlo_reduce_window()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce_window.html)
over `(values, iota)`.

## See also

[`nv_cummax()`](https://r-xla.github.io/anvl/dev/reference/nv_cummax.md)

## Examples

``` r
x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
prim_cummax(x, dim = 1L)
#> [[1]]
#> AnvlArray
#>  3 4 5
#>  3 4 9
#> [ CPUf32{2,3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1 1 1
#>  1 1 2
#> [ CPUi32{2,3} ] 
#> 
```
