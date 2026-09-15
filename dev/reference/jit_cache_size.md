# Number of cached programs of a jitted function

The number of compiled programs a function returned by
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) currently
holds for one backend, i.e. how many entries of its compilation cache
are filled. A call whose inputs hit an existing entry leaves this
unchanged; a call that misses adds one, up to the `cache_size` cap
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) was given,
beyond which the least recently used entry is evicted.

Each backend a jitted function has run on keeps its own cache, so this
is reported for one backend at a time.

## Usage

``` r
jit_cache_size(f, backend = active_backend())
```

## Arguments

- f:

  (`function`)  
  A function returned by
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

- backend:

  (`character(1)`)  
  The backend whose cache to report on. Defaults to the active one
  ([`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md)).

## Value

`integer(1)`. A backend that `f` has not run on yet reports `0`, since
the caches are created on first use.

## See also

[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)

## Examples

``` r
f <- jit(function(x, y) x + y)
jit_cache_size(f)
#> [1] 0

f(nv_scalar(1), nv_scalar(2))
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
jit_cache_size(f)
#> [1] 1

# same dtypes and shapes -- a cache hit, no new entry
f(nv_scalar(3), nv_scalar(4))
#> AnvlArray
#>  7
#> [ CPUf32{} ] 
jit_cache_size(f)
#> [1] 1

# a different shape -- a second program is compiled
f(nv_array(c(1, 2)), nv_array(c(3, 4)))
#> AnvlArray
#>  4
#>  6
#> [ CPUf32{2} ] 
jit_cache_size(f)
#> [1] 2
```
