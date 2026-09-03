# Convert to Abstract Array

Convert an object to its abstract array representation
([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)).

## Usage

``` r
to_abstract(x, pure = FALSE)
```

## Arguments

- x:

  (`any`)  
  Object to convert.

- pure:

  (`logical(1)`)  
  Whether to convert to a pure `AbstractArray` and not e.g.
  `LiteralArray` or `ConcreteArray`.

## Value

[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)

## Examples

``` r
# R literals become LiteralArrays
to_abstract(1.5)
#> RData(double, ()) 
to_abstract(1L)
#> RData(integer, ()) 
to_abstract(TRUE)
#> RData(logical, ()) 

# AnvlArrays become ConcreteArrays
to_abstract(nv_array(1:4))
#> ConcreteArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUi32{4} ] 

# Use pure = TRUE to strip subclass info
to_abstract(nv_array(1:4), pure = TRUE)
#> AbstractArray(dtype=i32, shape=4) 
```
