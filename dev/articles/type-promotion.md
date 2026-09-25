# Data Types and Promotion Rules

## Data Types

Every `AnvlArray` has a data type, which
[`dtype()`](https://r-xla.github.io/anvl/dev/reference/dtype.md)
reports. The data types fall into three categories, where the number is
the width in bits:

| Category | Data types |
|----|----|
| Boolean | `bool` |
| Integer | signed: `i8`, `i16`, `i32`, `i64`; unsigned: `ui8`, `ui16`, `ui32`, `ui64` |
| Float | `f32`, `f64` |

Other floating-point data types, such as `f16`, and complex numbers are
currently not supported.

The data type of a new `AnvlArray` is chosen with the `dtype` argument,
and
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
converts an existing one:

``` r

library(anvl)
x <- nv_array(1:3, dtype = "i16")
dtype(x)
```

    ## <i16>

``` r

nv_convert(x, "f64")
```

    ## AnvlArray
    ##  1
    ##  2
    ##  3
    ## [ CPUf64{3} ]

Without a `dtype`, an R `logical` becomes a `bool`, and R `double`s and
`integer`s become the default float and integer data types, which
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
reports and the `anvl.default_dtypes` option configures:

``` r

default_dtypes()
```

    ## $float
    ## <f32>
    ## 
    ## $int
    ## <i32>

Converting an `AnvlArray` back to R with
[`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
gives the R type that can represent its values, e.g. a `double` for an
`f32`; see
[`?as_array`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
for the full list. Help pages name groups of data types that an argument
accepts with single words, such as *numeric* or *integerish*, which
[`?dtypes`](https://r-xla.github.io/anvl/dev/reference/dtypes.md)
defines.

The rest of this article covers which data type the result of an
operation has when its inputs have different ones.

## Type Promotion Rules

When combining arrays of different types (e.g., adding an `f32` to an
`i32`), {anvl} needs to determine a common type. For example, below we
are adding an `f32` to an `f64`, where the former is promoted to the
latter’s type, because it’s more expressive.

``` r

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

The table below shows the common data type of every pair of data types.
The cells are colored by the category of the result: boolean, signed
integer, unsigned integer, float, and none.

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

Common data type of two data types {.table}

Two integer types meet at a type that holds every value of both, so `i8`
and `ui8` meet at `i16`. The exception is `ui64`: no signed integer type
holds its values, and an integer never becomes a float on its own, so a
`ui64` and a signed integer have no common type at all (the red cells
above). This is where JAX and NumPy fall back to a float – {anvl}
instead asks you to say what you want, by converting one of the operands
with
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md).

R values are handled differently: {anvl}’s type system keeps their full
precision, and lets the author of a function decide how they are
promoted, without losing precision along the way. We describe how this
works below.

## R Values Have No Data Type

An R number is a 64-bit double, while {anvl} computes in `f32` by
default. There are two simple ways to reconcile the two, and both have a
drawback:

- Converting R doubles always to `f64` means that every R number, such
  as a step size or `pi`, pulls the computation to `f64`, which is
  considerably slower on GPUs.
- Converting R doubles always to the default `f32` means that an R
  number loses precision before it is even used, also when it is
  combined with an `f64` `AnvlArray`.

Below, `exact` is what {anvl} computes, and `rounded` is what the second
approach would give, where `pi` is rounded to an `f32` first:

``` r

exact <- nv_add(pi, nv_scalar(2, dtype = "f64"))
rounded <- nv_add(
  nv_convert(nv_scalar(pi, dtype = "f32"), "f64"),
  nv_scalar(2, dtype = "f64")
)
print(c(exact = as_array(exact), rounded = as_array(rounded), base_r = pi + 2), digits = 16)
```

    ##             exact           rounded            base_r 
    ## 5.141592653589793 5.141592741012573 5.141592653589793

{anvl} avoids both drawbacks: an R value has no data type of its own,
and takes the data type of the `AnvlArray` it is combined with. Combined
with an `f64`, `pi` keeps its full precision and gives the same result
as base R, while combined with an `f32`, the computation stays fast.
This makes {anvl} well suited for statistical computations, where
precision often matters a lot.

In {anvl}’s type system, R objects do not have a concrete data type.

``` r

dtype(1)
```

    ## Error:
    ## ! An R value has no data type of its own until it is used.
    ## ℹ `dtype()` is undefined here for the same reason `dtype(1.5)` is: the value
    ##   only takes a data type when it meets a typed array, or when it materializes
    ##   at the default.
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
materialization of R objects as `AnvlArray`s. There is no uniform rule
how this is done, functions can decide this for themselves, but the two
most common routes are:

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
the boundary values cannot be promoted to `x` without loss of precision:

``` r

nv_clamp(
  x = nv_scalar(0, "f32"),
  min = nv_scalar(0, "f64"),
  max = nv_scalar(0, "f64")
)
```

    ## Error:
    ## ! Cannot bring `min` to data type "f32".
    ## ✖ "f64" is not promotable to "f32".
    ## ℹ Convert it explicitly with `nv_convert()`.

Functions document their behavior, so consult their respective help page
for more information on how they promote their inputs.

When you are writing your own {anvl} functions, you can decide the
promotion rules yourself. This is possible via the `.promote` argument
of
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
common data type of the inputs, so below, the R value `1` takes the data
type of the `f64` input:

``` r

as_anvl_arrays(1, nv_scalar(2, dtype = "f64"), .promote = promotion_common())
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

When none of the inputs has a data type of its own,
e.g. `as_anvl_arrays(1, 2, .promote = promotion_common())`, it falls
back to the default data type of an R `double`. We can override that
default via
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

When used in
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md),
it converts the inputs to that data type:

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
