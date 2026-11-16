# Primitive Optimization Barrier

Returns the input arrays unchanged and prevents the compiler from moving
computations across the barrier. Operations that produce the input
arrays must complete before operations that use the returned arrays.

A barrier prevents fusion between these operations. This can improve
performance when fusion would prevent vectorization or repeat an
expensive computation. It can also increase memory traffic, so benchmark
the result before using a barrier for performance.

During reverse-mode differentiation, gradients pass through unchanged,
with a separate barrier in the backward pass.

## Usage

``` r
prim_optimization_barrier(xs)
```

## Arguments

- xs:

  (`list` of
  [`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  A non-empty list of arrays. The arrays may have different data types
  and shapes.

## Value

(`list` of
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
The input arrays, with their values, data types, shapes, order and names
preserved.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_optimization_barrier()`](https://r-xla.github.io/stablehlo/reference/hlo_optimization_barrier.html),
specified under
[optimization_barrier](https://openxla.org/stablehlo/spec#optimization_barrier).
All arrays in `xs` share a single barrier.

## See also

[`nv_optimization_barrier()`](https://r-xla.github.io/anvl/reference/nv_optimization_barrier.md)

## Examples

``` r
# Prevent fusion of the square root and multiplication with the addition.
f <- jit(function(x) {
  out <- prim_optimization_barrier(list(a = sqrt(x), b = x * 2))
  out$a + out$b
})
f(nv_array(c(1, 4, 9)))
#> AnvlArray
#>   3
#>  10
#>  21
#> [ CPUf32{3} ] 
```
