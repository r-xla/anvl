# Block until an async operation completes

Block until the array's underlying computation has finished, and return
the object invisibly. Useful for benchmarking, where the dispatch of an
asynchronous operation should not be confused with its execution.

## Usage

``` r
await(x, ...)
```

## Arguments

- x:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  or other awaitable)  
  An object with an `await()` method.

- ...:

  Additional arguments passed to methods (unused).

## Value

`x`, invisibly.

## Details

Implemented via the generic
[`pjrt::await()`](https://r-xla.github.io/pjrt/reference/await.html).
For backends without asynchronous execution (e.g. `"quickr"`), this is a
no-op.

## See also

[`pjrt::await()`](https://r-xla.github.io/pjrt/reference/await.html),
[`map_tree()`](https://r-xla.github.io/anvl/dev/reference/map_tree.md)
(to await a tree of outputs)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
await(x)

# Await all leaves of a (possibly nested) list of arrays.
map_tree(list(x, list(y = x)), await)
#> [[1]]
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUf32{4} ] 
#> 
#> [[2]]
#> [[2]]$y
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUf32{4} ] 
#> 
#> 
```
