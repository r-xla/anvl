# Create a Random State

Creates a random state of its own, which the samplers draw from in place
of the global random state within
[`with_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md)
or after
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md).
Draws advance it in place. A function compiled with
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md) can draw from a
random state it takes as an argument or closes over, but cannot create
or return one.

See the [Random Number
Generation](https://r-xla.github.io/anvl/articles/random-numbers.html)
article for more details.

## Usage

``` r
nv_rng_state(seed)
```

## Arguments

- seed:

  (`integer(1)`)  
  The seed, a 32-bit integer, which gives the same random state as
  [`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md)
  gives the global random state.

## Value

(`AnvlRngState`)  
A random state of the key that `seed` gives and the counter 0.

## See also

Other rng:
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md),
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md),
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md),
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md),
[`nv_uniform`](https://r-xla.github.io/anvl/reference/nv_uniform.md)

## Examples

``` r
rng <- nv_rng_state(42L)
with_nv_rng_state(rng, nv_runif(3L))
#> AnvlArray
#>  0.5303
#>  0.3134
#>  0.9015
#> [ CPUf32{3} ] 
# the random state was advanced, so the next draw differs
with_nv_rng_state(rng, nv_runif(3L))
#> AnvlArray
#>  0.7277
#>  0.7879
#>  0.1817
#> [ CPUf32{3} ] 
rng
#> <AnvlRngState> key 0x000000000000002a, counter 0x0000000000000002

# a jitted function takes a random state as an argument
f <- jit(function(x, rng) with_nv_rng_state(rng, x + nv_rnorm(2L)))
f(nv_array(c(1, 2)), nv_rng_state(1L))
#> AnvlArray
#>  1.8863
#>  0.7194
#> [ CPUf32{2} ] 
```
