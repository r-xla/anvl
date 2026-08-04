# Random Number Generation

In this vignette, you will learn how to generate random numbers in
{anvl}, which is different from base R, where random number generation
uses a global state (`.Random.seed`) that is automatically updated after
each call:

``` r

set.seed(42)
.Random.seed[2:4]
#> [1]        624  507561766 1260545903
rnorm(3)
#> [1]  1.3709584 -0.5646982  0.3631284
.Random.seed[2:4]
#> [1]           6 -1577024373  1699409082
rnorm(3)
#> [1]  0.6328626  0.4042683 -0.1061245
.Random.seed[2:4]
#> [1]          12 -1577024373  1699409082
```

In {anvl}, the random state must be explicitly passed around. This is
because we are following a functional programming paradigm where
functions are pure and don’t have side effects.

**Note:** This explicit state-passing behavior might change in the
future to provide a more R-like experience, but for now you need to
manage the state yourself.

To generate random numbers, you first need to create an initial RNG
state, which is simply a `ui64[2]`. You can convert a seed into a state
using
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md):

``` r

library(anvl)
state <- nv_rng_state(42L)
state
#> AnvlArray
#>  42
#>   0
#> [ CPUui64{2} ]
```

The main functions for generating random numbers are
[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md),
[`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
and
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md).
All those functions return a list with two elements:

1.  The **new** RNG state (to be used for subsequent random number
    generation).
2.  The generated random numbers.

Let’s generate some uniform random numbers:

``` r

result <- nv_runif(state, dtype = "f32", shape = c(2, 3))
result[[1]]  # new state
#> AnvlArray
#>  42
#>   3
#> [ CPUui64{2} ]
result[[2]]  # random numbers
#> AnvlArray
#>  0.8690 0.1506 0.5203
#>  0.3103 0.9928 0.1065
#> [ CPUf32{2,3} ]
```

For normally distributed random numbers:

``` r

result <- nv_rnorm(state, dtype = "f32", shape = c(2, 3), mean = 0, sd = 1)
result[[2]]
#> AnvlArray
#>  -0.0675  0.9489  1.9457
#>  -0.5255  1.2002  0.0008
#> [ CPUf32{2,3} ]
```

`mean` and `sd` are arrayish, so they may vary across the sample, as
long as they have the same shape as it:

``` r

sds <- nv_matrix(c(0.01, 0.1, 1, 10, 100, 1000), nrow = 2)
nv_rnorm(state, dtype = "f32", shape = c(2, 3), sd = sds)[[2]]
#> AnvlArray
#>   -0.0007   0.9489 194.5720
#>   -0.0526  12.0017   0.7665
#> [ CPUf32{2,3} ]
```

To draw integers, use
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
the counterpart to R’s
[`sample.int()`](https://rdrr.io/r/base/sample.html):

``` r

# roll six dice
nv_sample_int(6L, state, 6L)[[2]]
#> AnvlArray
#>  4
#>  6
#>  2
#>  5
#>  1
#>  2
#> [ CPUi32{6} ]
```

[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md)
draws from an arbitrary population instead. Unlike R’s
[`sample()`](https://rdrr.io/r/base/sample.html), its population
argument is never overloaded with a count, so
`nv_sample(5L, state, nv_array(6))` samples from the population `6`
rather than from `1:6`:

``` r

population <- nv_array(c(10, 20, 30))
nv_sample(8L, state, population)[[2]]
#> AnvlArray
#>  20
#>  30
#>  10
#>  30
#>  10
#>  10
#>  10
#>  20
#> [ CPUf32{8} ]
```

One thing to avoid is to reuse the same state for multiple calls as done
in the example below:

``` r

result1 <- nv_runif(state, dtype = "f32", shape = 3L)
result2 <- nv_runif(state, dtype = "f32", shape = 3L)
list(first = result1[[2]], second = result2[[2]])
#> $first
#> AnvlArray
#>  0.8690
#>  0.3103
#>  0.1506
#> [ CPUf32{3} ] 
#> 
#> $second
#> AnvlArray
#>  0.8690
#>  0.3103
#>  0.1506
#> [ CPUf32{3} ]
```

As you can see, both calls produced identical random numbers because we
used the same state for both. To get different random numbers in
subsequent calls, you need to pass the **new** state returned by the
previous call:

``` r

result1 <- nv_runif(state, dtype = "f32", shape = 3L)
new_state <- result1[[1]]
result2 <- nv_runif(new_state, dtype = "f32", shape = 3L)
list(first = result1[[2]], second = result2[[2]])
#> $first
#> AnvlArray
#>  0.8690
#>  0.3103
#>  0.1506
#> [ CPUf32{3} ] 
#> 
#> $second
#> AnvlArray
#>  0.5203
#>  0.1065
#>  0.2499
#> [ CPUf32{3} ]
```

Now we get different random numbers because we properly propagated the
state.
