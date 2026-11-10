# Set the Global Random State

`nv_set_seed()` seeds the global random state, like base R's
[`set.seed()`](https://rdrr.io/r/base/Random.html). Until it is set, its
first use derives it from base R's RNG. `nv_get_rng_state()` and
`nv_set_rng_state()` save and restore it, and `local_nv_seed()` and
`with_nv_seed()` seed it for a scope.

See the [Random Number
Generation](https://r-xla.github.io/anvl/articles/random-numbers.html)
article for more details.

## Usage

``` r
nv_set_seed(seed)

nv_get_rng_state()

nv_set_rng_state(rng_state)

local_nv_seed(seed, envir = parent.frame())

with_nv_seed(seed, code)
```

## Arguments

- seed:

  (`integer(1)`)  
  The seed, a 32-bit integer.

- rng_state:

  (`AnvlRngState`)  
  A random state, as `nv_get_rng_state()` returns.

- envir:

  (`environment`)  
  The scope whose exit restores the global random state.

- code:

  (any)  
  An expression to evaluate with the seeded global random state.

## Value

`nv_set_seed()`, `local_nv_seed()` and `nv_set_rng_state()` return
`NULL`, invisibly; `with_nv_seed()` returns the result of evaluating
`code`. `nv_get_rng_state()` returns a copy of the global random state,
an `AnvlRngState`.

## See also

Other rng:
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md),
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md),
[`nv_uniform`](https://r-xla.github.io/anvl/reference/nv_uniform.md)

## Examples

``` r
nv_set_seed(42L)
nv_runif(3L)
#> AnvlArray
#>  0.5303
#>  0.3134
#>  0.9015
#> [ CPUf32{3} ] 
nv_set_seed(42L)
nv_runif(3L)
#> AnvlArray
#>  0.5303
#>  0.3134
#>  0.9015
#> [ CPUf32{3} ] 

# save the random state and continue from it later
rng_state <- nv_get_rng_state()
nv_runif(2L)
#> AnvlArray
#>  0.7277
#>  0.7879
#> [ CPUf32{2} ] 
nv_set_rng_state(rng_state)
nv_runif(2L)
#> AnvlArray
#>  0.7277
#>  0.7879
#> [ CPUf32{2} ] 

with_nv_seed(1L, nv_runif(2L))
#> AnvlArray
#>  0.7026
#>  0.2348
#> [ CPUf32{2} ] 
```
