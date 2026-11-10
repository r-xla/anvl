# Draw From a Random State for a Scope

`local_nv_rng_state()` and `with_nv_rng_state()` make `rng_state` the
random state the samplers draw from, for the calling scope or for `code`
only, and restore the one from before afterwards. The draws advance
`rng_state` in place. In a function traced by
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md), they cannot be
called in the body of
[`nv_while()`](https://r-xla.github.io/anvl/reference/nv_while.md) or
[`nv_scan()`](https://r-xla.github.io/anvl/reference/nv_scan.md): called
before the loop, they reach its body.

See the [Random Number
Generation](https://r-xla.github.io/anvl/articles/random-numbers.html)
article for more details.

## Usage

``` r
local_nv_rng_state(rng_state, envir = parent.frame())

with_nv_rng_state(rng_state, code)
```

## Arguments

- rng_state:

  (`AnvlRngState`)  
  The random state to draw from, as
  [`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md)
  returns.

- envir:

  (`environment`)  
  The scope whose exit restores the default from before.

- code:

  (any)  
  An expression to evaluate drawing from `rng_state`.

## Value

`local_nv_rng_state()` returns `NULL`, invisibly; `with_nv_rng_state()`
returns the result of evaluating `code`.

## See also

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md),
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md),
[`nv_uniform`](https://r-xla.github.io/anvl/reference/nv_uniform.md)

## Examples

``` r
rng <- nv_rng_state(1L)
with_nv_rng_state(rng, list(nv_runif(2L), nv_rnorm(2L)))
#> [[1]]
#> AnvlArray
#>  0.7026
#>  0.2348
#> [ CPUf32{2} ] 
#> 
#> [[2]]
#> AnvlArray
#>  -0.3611
#>  -0.9505
#> [ CPUf32{2} ] 
#> 
# both samplers drew from `rng`, three sampling calls in all
rng
#> <AnvlRngState> key 0x0000000000000001, counter 0x0000000000000003

# in a jitted function, from a random state it takes as an argument
step <- jit(function(x, rng) {
  local_nv_rng_state(rng)
  x + nv_rnorm(2L) + nv_runif(2L)
})
step(nv_array(c(0, 0)), rng)
#> AnvlArray
#>  -0.4693
#>   0.1911
#> [ CPUf32{2} ] 
rng
#> <AnvlRngState> key 0x0000000000000001, counter 0x0000000000000006
```
