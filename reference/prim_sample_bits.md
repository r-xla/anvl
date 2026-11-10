# Primitive Random Bits

Draws random bits from the active random state (see
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md))
with the ThreeFry-2x32 counter-based generator of Salmon et al. (2011),
and advances that random state by one step.

See the [Random Number
Generation](https://r-xla.github.io/anvl/articles/random-numbers.html)
article for more details.

## Usage

``` r
prim_sample_bits(shape, dtype)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the result.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/xlamisc/reference/DataType.html))  
  An unsigned integer data type: `"ui8"`, `"ui16"`, `"ui32"` or
  `"ui64"`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
The random bits, of data type `dtype` and shape `shape`. Every value is
made from an encrypted 64-bit block of its own: a `"ui64"` value is the
block, and a narrower value the xor of the block's two 32-bit words,
truncated to its low bits, as JAX's `jax.random.bits()` does by default.
The value at a position thus does not depend on the number of values
drawn, so a draw of `3` is the start of a draw of `6`. The draws fill
`shape` in row-major order, as JAX's do, so a draw of `c(2, 3)` is not a
draw of `6` filled into `c(2, 3)` in R's column-major order.

## Random State

The draws come from the active random state: the global random state
(see
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md)),
or the random state that
[`with_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md)
or
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md)
make the default for a scope. They advance it in place.

## Implemented Rules

- `stablehlo`

## References

Salmon J, Moraes M, Dror R, Shaw D (2011). “Parallel Random Numbers: As
Easy as 1, 2, 3.” In *Proceedings of the 2011 International Conference
for High Performance Computing, Networking, Storage and Analysis*.
[doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405) .

Claessen K, Pałka M (2013). “Splittable Pseudorandom Number Generators
using Cryptographic Hashing.” In *Proceedings of the 2013 ACM SIGPLAN
Symposium on Haskell*, 47–58.
[doi:10.1145/2503778.2503784](https://doi.org/10.1145/2503778.2503784) .

Bradbury J, Frostig R, Hawkins P, Johnson M, Katariya Y, Leary C,
Maclaurin D, Necula G, Paszke A, VanderPlas J, Wanderman-Milne S, Zhang
Q (2018). *JAX: composable transformations of Python+NumPy programs*.
Version 0.11.2, <http://github.com/jax-ml/jax>.

## See also

[`nv_runif()`](https://r-xla.github.io/anvl/reference/nv_uniform.md),
[`nv_rnorm()`](https://r-xla.github.io/anvl/reference/nv_normal.md)

## Examples

``` r
nv_get_rng_state()
#> <AnvlRngState> key 0x000000000000002a, counter 0x0000000000000003
prim_sample_bits(2L, "ui32")
#> AnvlArray
#>  1.6708e+09
#>  3.6823e+08
#> [ CPUui32{2} ] 
# The counter is incremented by 1
nv_get_rng_state()
#> <AnvlRngState> key 0x000000000000002a, counter 0x0000000000000004
# A random state of its own is drawn from within with_nv_rng_state():
custom_rng <- nv_rng_state(1L)
custom_rng
#> <AnvlRngState> key 0x0000000000000001, counter 0x0000000000000000
with_nv_rng_state(custom_rng, prim_sample_bits(2L, "ui32"))
#> AnvlArray
#>  3.0177e+09
#>  1.0086e+09
#> [ CPUui32{2} ] 
# And it is still updated
custom_rng
#> <AnvlRngState> key 0x0000000000000001, counter 0x0000000000000001
# And does not affect the global RNG
nv_get_rng_state()
#> <AnvlRngState> key 0x000000000000002a, counter 0x0000000000000004
```
