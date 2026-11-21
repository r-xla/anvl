# Optimization Barrier

Returns the input arrays unchanged, converting any R values to arrays,
and prevents the compiler from moving computations across the barrier.
Operations that produce the input arrays must complete before operations
that use the returned arrays.

A barrier prevents fusion between these operations. This can improve
performance when fusion would prevent vectorization or repeat an
expensive computation. It can also increase memory traffic, so benchmark
the result before using a barrier for performance.

[`prim_optimization_barrier()`](https://r-xla.github.io/anvl/dev/reference/prim_optimization_barrier.md)
takes a flat list of arrays. This function also accepts a single array
or a nested list, and returns it with the same structure and names. All
arrays share a single barrier.

## Usage

``` r
nv_optimization_barrier(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \| `list`)  
  An array or a list of arrays, nested to any depth (see
  [`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html)).
  The arrays may have different data types and shapes. R values are
  materialized at their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) \|
`list`)  
An array or nested list of arrays with the same structure and names as
`x`. Existing arrays retain their values, data types and shapes; R
values are materialized as described above. A list without any elements
to pass through, such as [`list()`](https://rdrr.io/r/base/list.html) or
`list(a = list())`, is returned unchanged.

## Gradients

[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
passes the gradients through unchanged, with a separate barrier in the
backward pass. See
[`prim_optimization_barrier()`](https://r-xla.github.io/anvl/dev/reference/prim_optimization_barrier.md).

## See also

[`prim_optimization_barrier()`](https://r-xla.github.io/anvl/dev/reference/prim_optimization_barrier.md)
for the underlying primitive.

## Examples

``` r
# Prevent fusion of the square root and multiplication with the addition.
f <- jit(function(x) {
  out <- nv_optimization_barrier(list(a = sqrt(x), b = x * 2))
  out$a + out$b
})
f(nv_array(c(1, 4, 9)))
#> AnvlArray
#>   3
#>  10
#>  21
#> [ CPUf32{3} ] 

# A single array is returned as an array.
nv_optimization_barrier(nv_array(1:3))
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
```
