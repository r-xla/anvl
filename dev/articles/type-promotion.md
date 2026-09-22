# Type Promotion

## Type Promotion Rules

When combining arrays of different types (e.g., adding an `f32` to an
`i32`), {anvl} needs to determine a common type. For example, below we
are adding an `f32` to an `f64`, where the former is promoted to the
latter’s type, because it’s more expressive.

``` r

library(anvl)
nv_add(
  nv_scalar(1.0, dtype = "f32"),
  nv_scalar(1.0, dtype = "f64")
)
```

    ## AnvlArray
    ##  2
    ## [ CPUf64{} ]

The rules are defined by the
[`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md)
function.

``` r

common_dtype("f64", "f32")
```

    ## <f64>

``` r

common_dtype("i64", "f32")
```

    ## <f32>

A table with the promotion rules is below.

|          | bool | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
|:---------|:-----|:----|:----|:----|:----|:-----|:-----|:-----|:-----|:----|:----|
| **bool** | bool | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
| **i8**   | i8   | i8  | i16 | i32 | i64 | i16  | i32  | i64  | –    | f32 | f64 |
| **i16**  | i16  | i16 | i16 | i32 | i64 | i16  | i32  | i64  | –    | f32 | f64 |
| **i32**  | i32  | i32 | i32 | i32 | i64 | i32  | i32  | i64  | –    | f32 | f64 |
| **i64**  | i64  | i64 | i64 | i64 | i64 | i64  | i64  | i64  | –    | f32 | f64 |
| **ui8**  | ui8  | i16 | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
| **ui16** | ui16 | i32 | i32 | i32 | i64 | ui16 | ui16 | ui32 | ui64 | f32 | f64 |
| **ui32** | ui32 | i64 | i64 | i64 | i64 | ui32 | ui32 | ui32 | ui64 | f32 | f64 |
| **ui64** | ui64 | –   | –   | –   | –   | ui64 | ui64 | ui64 | ui64 | f32 | f64 |
| **f32**  | f32  | f32 | f32 | f32 | f32 | f32  | f32  | f32  | f32  | f32 | f64 |
| **f64**  | f64  | f64 | f64 | f64 | f64 | f64  | f64  | f64  | f64  | f64 | f64 |

Type promotion rules (row × column) {.table}

Two integer types meet at a type that holds every value of both, so `i8`
and `ui8` meet at `i16`. The exception is `ui64`: no signed integer type
holds its values, and an integer never becomes a float on its own, so a
`ui64` and a signed integer have no common type at all (the `--` cells
above). This is where JAX and NumPy fall back to a float – {anvl}
instead asks you to say what you want, by converting one of the operands
with
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md).

The biggest differentiator between our type system and the one from JAX
is the handling of array objects from the host language, which is R in
our case. While introducing more complexity, these rules prevent the
loss of precision present in JAX’s type system. We describe it below.

## R Values Have No Data Type

In {anvl}’s type system, R objects do not have a concrete data type.

``` r

dtype(1)
```

    ## Error:
    ## ! An R value has no data type of its own until it is used.
    ## ℹ `dtype()` is undefined here for the same reason `dtype(1.5)` is: the value
    ##   only takes a data type when it meets a typed array, or when it materializes
    ##   at the default ("f32").
    ## ℹ Give it one explicitly with `nv_convert()`.

The type promotion table from above therefore does not apply to them.
However, it is possible to apply an {anvl} function to R values
(length-1 vectors and arrays):

``` r

nv_exp(1)
```

    ## AnvlArray
    ##  2.7183
    ## [ CPUf32{} ]

``` r

nv_add(1, nv_scalar(1, "f64"))
```

    ## AnvlArray
    ##  2
    ## [ CPUf64{} ]

``` r

nv_add(1, 2)
```

    ## AnvlArray
    ##  3
    ## [ CPUf32{} ]

It is therefore important to understand the rules that govern the
materialization of R objects as `AnvlArray`s. Generally, there are two
routes:

1.  The R value is materialized at its default data type, which
    [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
    reports: a `double` and an `integer` take the defaults of the active
    backend, configurable through the `anvl.default_dtypes` option, and
    a `logical` takes `bool`. This is e.g. the case in unary functions
    such as `nv_exp`.
2.  The R value’s data type is inferred from the other arguments, as in
    the `nv_add` call above. When no concrete data type is present,
    `nv_add(1, 2)` falls back to that default again.

Either way the value is *built at* that data type rather than converted
into it, so one the data type cannot hold is an error:

``` r

jit(function(x) x + (-2L))(nv_scalar(1L, "ui8"))
```

    ## Error:
    ## ! Cannot build the R value -2 at data type "ui8".
    ## ✖ It is outside the range of "ui8" (0 to 255).
    ## ℹ An R value is built at the data type it meets rather than converted into it,
    ##   so that data type has to hold it.
    ## ℹ Convert an array with `nv_convert()` where the wraparound is what you want.

Converting an array is a different matter and keeps XLA’s wraparound
semantics:

``` r

nv_convert(nv_scalar(-2L, "i32"), "ui8")
```

    ## AnvlArray
    ##  254
    ## [ CPUui8{} ]

Note that these rules are not universal and exceptions exist. Some
functions, such as `nv_clamp`, prioritize the data type of a specific
argument, in this case `x`, the value that is being clamped. It fails if
the the boundary values cannot be promoted to `x` without loss of
precision:

``` r

nv_clamp(
  min_val = nv_scalar(0, "f64"),
  x = nv_scalar(0, "f32"),
  max_val = nv_scalar(0, "f64")
)
```

    ## Error:
    ## ! Cannot bring `min_val` to data type "f32".
    ## ✖ "f64" is not promotable to "f32".
    ## ℹ Convert it explicitly with `nv_convert()`.

Functions document their behavior, so consult their respective help page
for more information.

You can canonicalize and promote inputs to a function via the `.promote`
field of
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md).
It takes a `PromotionRule`, which is a special function that takes in
the arguments and outputs a data type for each one.

``` r

as_anvl_arrays(1, 2, .promote = promotion_dtype("f64"))
```

    ## [[1]]
    ## AnvlArray
    ##  1
    ## [ CPUf64{} ] 
    ## 
    ## [[2]]
    ## AnvlArray
    ##  2
    ## [ CPUf64{} ]

One common rule is
[`promotion_common()`](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md),
which is used by functions such as `nv_add` above. It computes the
common data type of the inputs. In this case, it returns the default
data type of an R `double`, as neither input has one of its own. We can
override that default via
[`with_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md),
which we use below to ask for double precision. The two elements
contrast what the override does and does not touch: `materialized` is an
R literal with no data type of its own, so it follows the new default,
while `yielded` meets a typed `f32` array and takes *its* data type – a
default only decides what happens when nothing else does.

``` r

with_default_dtypes(c(float = "f64"), {
  list(
    materialized = jit(\() 1.0)(),
    yielded = nv_array(1, dtype = "f32") + 1.5
  )
})
```

    ## $materialized
    ## AnvlArray
    ##  1
    ## [ CPUf64{} ] 
    ## 
    ## $yielded
    ## AnvlArray
    ##  2.5000
    ## [ CPUf32{1} ]

Below, we compute the common data type of the arguments.

``` r

promotion_fn <- promotion_common()
args <- list(1, 2, nv_scalar(1L, "i8"))
promotion_fn(args)
```

    ## [[1]]
    ## <f32>
    ## 
    ## [[2]]
    ## <f32>
    ## 
    ## [[3]]
    ## <f32>

And when used in
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
it moves the inputs there:

``` r

do.call(as_anvl_arrays, c(args, list(.promote = promotion_fn)))
```

    ## [[1]]
    ## AnvlArray
    ##  1
    ## [ CPUf32{} ] 
    ## 
    ## [[2]]
    ## AnvlArray
    ##  2
    ## [ CPUf32{} ] 
    ## 
    ## [[3]]
    ## AnvlArray
    ##  1
    ## [ CPUf32{} ]

For more information about the available rules, see
[`?promotion_rule`](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md).
