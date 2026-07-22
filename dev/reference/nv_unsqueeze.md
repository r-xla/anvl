# Unsqueeze

Inserts a dimension of size 1 at the specified position.

## Usage

``` r
nv_unsqueeze(x, dim)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- dim:

  (`integer(1)`)  
  Position at which to insert the new dimension.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as `x` with an extra dimension of size 1.

## See also

[`nv_squeeze()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md),
[`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3))
nv_unsqueeze(x, dim = 1L)
#> AnvlArray
#>  1 2 3
#> [ CPUf32{1,3} ] 
```
