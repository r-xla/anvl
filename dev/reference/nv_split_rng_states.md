# Split Random States Off a Random State

Splits `n` random states of their own off a random state, e.g. one per
chain of a simulation, so that one seed decides the draws of all of
them. The split is one sampling call of `rng_state`, which advances its
counter by one however large `n` is: it derives a fresh key from
`rng_state`, and the first `n` keys split off that fresh key are the
keys of the random states, each with the counter 0. The first `m` random
states of a split into `n` are therefore those of a split into `m`, so a
chain draws the same numbers however many chains there are. A function
compiled with
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) cannot
split a random state, but it can draw from the random states split off
one.

See the [Random Number
Generation](https://r-xla.github.io/anvl/articles/random-numbers.html)
article for more details.

## Usage

``` r
nv_split_rng_states(n, rng_state = NULL)
```

## Arguments

- n:

  (`integer(1)`)  
  The number of random states to split off.

- rng_state:

  (`AnvlRngState` \| `NULL`)  
  The random state to split, which the split advances in place. `NULL`,
  the default, splits the random state the samplers draw from: the
  global random state, or the one that
  [`with_nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/local_nv_rng_state.md)
  or
  [`local_nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/local_nv_rng_state.md)
  set.

## Value

(`list` of `AnvlRngState`)  
`n` random states, each of a key of its own and the counter 0.

## See also

Other rng:
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/local_nv_rng_state.md),
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
[`nv_set_seed()`](https://r-xla.github.io/anvl/dev/reference/nv_set_seed.md),
[`nv_uniform`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)

## Examples

``` r
nv_set_seed(42L)
chains <- nv_split_rng_states(3L)
chains
#> [[1]]
#> <AnvlRngState> key 0x3c54dd4abbebf007, counter 0x0000000000000000
#> 
#> [[2]]
#> <AnvlRngState> key 0x65ae5e0e3596dfce, counter 0x0000000000000000
#> 
#> [[3]]
#> <AnvlRngState> key 0xf423f62512e94690, counter 0x0000000000000000
#> 
# the split was one sampling call of the global random state
nv_get_rng_state()
#> <AnvlRngState> key 0x000000000000002a, counter 0x0000000000000001

# one draw per chain
lapply(chains, function(rng) with_nv_rng_state(rng, nv_runif(2L)))
#> [[1]]
#> AnvlArray
#>  0.9433
#>  0.5275
#> [ CPUf32{2} ] 
#> 
#> [[2]]
#> AnvlArray
#>  0.2955
#>  0.1507
#> [ CPUf32{2} ] 
#> 
#> [[3]]
#> AnvlArray
#>  0.3759
#>  0.5921
#> [ CPUf32{2} ] 
#> 

# a larger split starts with the same random states
with_nv_seed(42L, nv_split_rng_states(5L))[1:3]
#> [[1]]
#> <AnvlRngState> key 0x3c54dd4abbebf007, counter 0x0000000000000000
#> 
#> [[2]]
#> <AnvlRngState> key 0x65ae5e0e3596dfce, counter 0x0000000000000000
#> 
#> [[3]]
#> <AnvlRngState> key 0xf423f62512e94690, counter 0x0000000000000000
#> 
```
