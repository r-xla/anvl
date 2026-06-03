# Ahead-of-time compile a function to XLA

Compiles a function to an XLA executable via tracing.

Returns a callable R function that executes the compiled binary. Unlike
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md), compilation
happens eagerly at definition time rather than on first call, so the
input shapes and dtypes must be specified upfront via abstract arrays
(see
[`nv_aval()`](https://r-xla.github.io/anvl/reference/AbstractArray.md)).

## Usage

``` r
xla(f, args, donate = character(), device = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to compile. Must accept and return
  [`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)s.

- args:

  (`list`)  
  List of abstract array specifications (e.g. from
  [`nv_aval()`](https://r-xla.github.io/anvl/reference/AbstractArray.md))
  describing the expected shapes and dtypes of `f`'s arguments.

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated.

- device:

  (`character(1)` \| `PJRTDevice`)  
  Target device such as `"cpu"` (default) or `"cuda"`.

## Value

(`function`)  
A function that accepts
[`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
arguments (matching the flat inputs) and returns the result as
[`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)s.

## Details

Traces `f` with the given abstract `args` (via
[`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md)),
lowers the resulting graph via
[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md) and
then compiles it to an XLA executable via
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).

## See also

[`jit()`](https://r-xla.github.io/anvl/reference/jit.md) for lazy
compilation,
[`compile_xla()`](https://r-xla.github.io/anvl/reference/compile_xla.md)
for the lower-level API.

## Examples

``` r
f_compiled <- xla(function(x, y) x + y,
  args = list(x = nv_aval("f32", c(2, 2)), y = nv_aval("f32", c(2, 2)))
)
a <- nv_matrix(1:4, nrow = 2, dtype = "f32")
b <- nv_matrix(5:8, nrow = 2, dtype = "f32")
f_compiled(a, b)
#> AnvlArray
#>   6 10
#>   8 12
#> [ CPUf32{2,2} ] 
```
