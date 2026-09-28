# JIT Compile a Function

Wraps a function so that it is traced and compiled on first call.
Subsequent calls with the same input structure, shapes, and dtypes hit
an LRU cache and skip recompilation.

## Usage

``` r
jit(f, static = character(), cache_size = 100L, device = NULL, ...)
```

## Arguments

- f:

  (`function`)  
  Function to compile.

- static:

  ([`character()`](https://rdrr.io/r/base/character.html) \|
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  Names or positions of parameters of `f` that are *not* arrays. Static
  values are embedded as constants in the compiled program; a new
  compilation is triggered whenever a static value changes. For example
  useful when you want R control flow in your function.

  A static value must not have reference semantics: an environment or an
  external pointer, also inside a `list`, is an error, since it could be
  mutated in place while the cache key stays equal. Closures are
  allowed, but their enclosing environment must not change in a way that
  modifies their behavior.

- cache_size:

  (`integer(1)`)  
  Maximum number of compiled executables to keep in the LRU cache.

- device:

  (`NULL` \| `character(1)` \|
  [device](https://r-xla.github.io/anvl/dev/reference/nv_device.md))  
  Target device, of the active backend. When a device is specified, all
  arrays are moved to it.

  The default (`NULL`) infers the device at call time from the array
  inputs, falling back to
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md).

- ...:

  Backend-specific options:

  - `donate` ([`character()`](https://rdrr.io/r/base/character.html),
    default [`character()`](https://rdrr.io/r/base/character.html)),
    `"pjrt"` only: names of arguments whose buffers the compiled program
    may reuse for its outputs. A donated array must not be used again
    after the call; this can save memory and copies for large inputs.
    Must not overlap with `static`.

  - `unwrap` (`logical(1)`, default `FALSE`), `"quickr"` only: if
    `TRUE`, the compiled function returns plain R arrays instead of
    [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)s,
    which is useful when the output is consumed by non-anvl R code.

  An option no backend takes is rejected here; one that only another
  backend takes is rejected when the function is called on a backend
  that does not, since the backend is not known until then.

## Value

(`JitFunction`)  
A `function` with the same formals as `f`.

## Backend and device

A jitted function runs on the active backend *when it is called*
([`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md),
set with
[`with_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
/
[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)),
so one `JitFunction` serves every backend, and a function created under
one backend and called under another runs on the latter. Array inputs
must belong to that backend; an array of another backend is rejected.
Each backend keeps its own compilation cache.

The device is a choice within that backend. Setting `device` explicitly
enforces that the function always uses it, e.g. `"cuda:0"`, and copies
every array input to it. With `device = NULL` (default) the device is
inferred from the input arrays and the constants within the program;
conflicting devices are an error, and with no array to read a device
from the default device is used.

## Default Data Types

It is possible to configure the default data types for `float`s and
`int`s via the `anvl.default_dtypes` option, see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).
The defaults are read at *call time*, not when `jit()` is called.

To pin a jitted function to a pair of data types instead of letting it
follow the configured defaults, wrap it in
[`with_dtypes()`](https://r-xla.github.io/anvl/dev/reference/with_dtypes.md):
the wrapper converts the array arguments and results of a category it
names, and sets the defaults for the duration of the call, so
`f_f64 <- with_dtypes(f, c(float = "f64"))` runs `f` at `f64`, unless
`f` itself changes the default data types.

## See also

[`jit_cache_size()`](https://r-xla.github.io/anvl/dev/reference/jit_cache_size.md)
for how many programs a jitted function has cached.

## Examples

``` r
f <- jit(function(x, y) x + y)
f(nv_array(1), nv_array(2))
#> AnvlArray
#>  3
#> [ CPUf32{1} ] 

# A non-static R value is an input of the program, just like an array: a
# new value reuses the compiled program
scale <- jit(function(x, n) x * n)
scale(nv_array(1:3), 2)
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ] 
scale(nv_array(1:3), 3)
#> AnvlArray
#>  3
#>  6
#>  9
#> [ CPUf32{3} ] 
jit_cache_size(scale)
#> [1] 1

# A static R value is embedded as a constant, so each new value compiles a
# new program
scale_static <- jit(function(x, n) x * n, static = "n")
scale_static(nv_array(1:3), 2)
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ] 
scale_static(nv_array(1:3), 3)
#> AnvlArray
#>  3
#>  6
#>  9
#> [ CPUf32{3} ] 
jit_cache_size(scale_static)
#> [1] 2

# static arguments enable data-dependent control flow
g <- jit(function(x, flag) {
  if (flag) x + 1 else x * 2
}, static = "flag")
g(nv_array(3), TRUE)
#> AnvlArray
#>  4
#> [ CPUf32{1} ] 
g(nv_array(3), FALSE)
#> AnvlArray
#>  6
#> [ CPUf32{1} ] 

# R values that meet no typed array take the default data types, which are
# read when the function is called
half <- jit(function(x) x / 2)
half(1)
#> AnvlArray
#>  0.5000
#> [ CPUf32{} ] 
with_default_dtypes(c(float = "f64"), half(1))
#> AnvlArray
#>  0.5000
#> [ CPUf64{} ] 
# or pin the data types of a function with `with_dtypes()`
half_f64 <- with_dtypes(half, c(float = "f64"))
half_f64(nv_array(1, dtype = "f32"))
#> AnvlArray
#>  0.5000
#> [ CPUf64{1} ] 

# a donated input's buffer is reused for the output, so the input must not
# be used afterwards
step <- jit(function(x) x + 1L, donate = "x")
x <- nv_array(1:3)
step(x)
#> AnvlArray
#>  2
#>  3
#>  4
#> [ CPUi32{3} ] 
try(as_array(x))
#> Error : called on deleted or donated buffer
# the same function runs on whichever backend is active when it is called
f <- jit(function(x, y) x + y)
with_backend("quickr", f(nv_array(1), nv_array(2)))
#> AnvlArray
#> [1] 3
#> [ CPUf64{1} ] 
```
