# Convert to Abstract Array

Convert an object to its abstract array representation
([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)).

## Usage

``` r
to_abstract(x, pure = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \|
  [`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  Object to convert.

- pure:

  (`logical(1)`)  
  Whether to convert to a pure `AbstractArray` and not e.g. `RData` or
  `ConcreteArray`.

## Value

([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
A
[`ConcreteArray`](https://r-xla.github.io/anvl/dev/reference/ConcreteArray.md)
for an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
an [`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md) for an
R value, the abstract array of a
[`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md),
and `x` itself for an
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md).
With `pure = TRUE`, a plain `AbstractArray` of the same shape and data
type.

## Examples

``` r
# an R value becomes `RData`: it has no data type of its own yet
to_abstract(1.5)
#> RData(double, ()) 
to_abstract(1L)
#> RData(integer, ()) 
to_abstract(TRUE)
#> RData(logical, ()) 

# an AnvlArray becomes a ConcreteArray
to_abstract(nv_array(1:4))
#> ConcreteArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUi32{4} ] 

# use pure = TRUE to strip subclass info
to_abstract(nv_array(1:4), pure = TRUE)
#> AbstractArray(dtype=i32, shape=4) 
```
