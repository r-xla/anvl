# Concrete Array Class

An
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
that also holds a reference to the actual array data. Usually represents
a closed-over constant in a program. Inherits from
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md).

## Usage

``` r
ConcreteArray(data)
```

## Arguments

- data:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  The actual array data.

## Lowering

When lowering to XLA, these become inputs to the executable instead of
embedding them into programs as constants. This is to avoid increasing
compilation time and bloating the size of the executable.

## Examples

``` r
y <- nv_array(c(0.5, 0.6))
x <- ConcreteArray(y)
x
#> ConcreteArray
#>  0.5000
#>  0.6000
#> [ CPUf32{2} ] 
ambiguous(x)
#> [1] FALSE
shape(x)
#> [1] 2
naxes(x)
#> [1] 1
dtype(x)
#> <f32>

# How it appears during tracing
graph <- trace_fn(function() y, list())
graph
#> <AnvlGraph>
#>   Inputs: (none)
#>   Constants:
#>     %c1: f32[2]
#>   Body: (empty)
#>   Outputs:
#>     %c1: f32[2] 
graph$outputs[[1]]$aval
#> ConcreteArray
#>  0.5000
#>  0.6000
#> [ CPUf32{2} ] 
```
