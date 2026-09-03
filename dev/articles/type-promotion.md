# Type Promotion

## Type Promotion Rules

When combining arrays of different types (e.g., adding an `f32` to an
`i32`), {anvl} needs to determine a common type. For example, below we
are adding an `f32` to an `f64`, where the former is promoted to the
latter’s type, because it’s more expressive.

``` r

library(anvl)
jit(nv_add)(
  nv_scalar(1.0, dtype = "f32"),
  nv_scalar(1.0, dtype = "f64")
)
```

    ## AnvlArray
    ##  2
    ## [ CPUf64{} ]

The type-promotion rules are inspired by JAX, and they are designed for
execution on accelerators like GPUs, where one often wants speed instead
of precision.

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

|      | bool | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
|:-----|:-----|:----|:----|:----|:----|:-----|:-----|:-----|:-----|:----|:----|
| bool | bool | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
| i8   | i8   | i8  | i16 | i32 | i64 | i16  | i32  | i64  | i64  | f32 | f64 |
| i16  | i16  | i16 | i16 | i32 | i64 | i16  | i32  | i64  | i64  | f32 | f64 |
| i32  | i32  | i32 | i32 | i32 | i64 | i32  | i32  | i64  | i64  | f32 | f64 |
| i64  | i64  | i64 | i64 | i64 | i64 | i64  | i64  | i64  | i64  | f32 | f64 |
| ui8  | ui8  | i16 | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
| ui16 | ui16 | i32 | i32 | i32 | i64 | ui16 | ui16 | ui32 | ui64 | f32 | f64 |
| ui32 | ui32 | i64 | i64 | i64 | i64 | ui32 | ui32 | ui32 | ui64 | f32 | f64 |
| ui64 | ui64 | i64 | i64 | i64 | i64 | ui64 | ui64 | ui64 | ui64 | f32 | f64 |
| f32  | f32  | f32 | f32 | f32 | f32 | f32  | f32  | f32  | f32  | f32 | f64 |
| f64  | f64  | f64 | f64 | f64 | f64 | f64  | f64  | f64  | f64  | f64 | f64 |

Type promotion rules (row × column) {.table}

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
    ##   only takes a data type when it meets a typed array, or when it commits to the
    ##   default ("f32").
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

It is therefore important to understand to understand the rules that
govern the materialization of R objects as `AnvlArray`s. Generally,
there are two routes:

1.  An R values it commited at its default data type (`double -> f32`,
    `integer -> i32`, `logical -> bool`). This is e.g. the case in unary
    functions such as `nv_exp`.
2.  The R values data type is inferred from other arguments, as is the
    case of the `nv_add` call above. When no concrete data type is
    present, `nv_add(1, 2)` falls back to the default, which is `f32`
    for `double`s.

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

as_anvl_arrays(1, 2, .promote = promote_dtype("f64"))
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
[`promote_common()`](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md),
which is used by functions such as `nv_add` above. It computes the
common data type of the inputs. In this case, it returns `f32`, which is
the default data type of R `double`s.

``` r

promote_fn <- promote_common()
args <- list(1, 2, nv_scalar(1L, "i8"))
promote_fn(args)
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

do.call(as_anvl_arrays, c(args, list(.promote = promote_fn)))
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
