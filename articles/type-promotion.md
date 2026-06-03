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
[`common_dtype()`](https://r-xla.github.io/anvl/reference/common_dtype.md)
function. It returns a [`list()`](https://rdrr.io/r/base/list.html) with
two values: the common dtype and a flag indicating whether the result is
ambiguous, which we will cover later.

``` r

common_dtype("f64", "f32")$dtype
```

    ## <f64>

``` r

common_dtype("i64", "f32")$dtype
```

    ## <f32>

A table with the promotion rules is below.

|      | bool | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
|:-----|:-----|:----|:----|:----|:----|:-----|:-----|:-----|:-----|:----|:----|
| bool | i1   | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
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

## Literals as Ambiguous Types

Usually, the types in an {anvl} program can be deterministically
inferred from the input types. The only case where this is not possible
is when you use R literals. The default types for literals are as
follows:

- [`double()`](https://rdrr.io/r/base/double.html) -\> `f32`
- [`integer()`](https://rdrr.io/r/base/integer.html) -\> `i32`
- [`logical()`](https://rdrr.io/r/base/logical.html) -\> `i1` (bool)

``` r

jit(\() list(1L, 1.0, TRUE))()
```

    ## [[1]]
    ## AnvlArray
    ##  1
    ## [ CPUi32?{} ] 
    ## 
    ## [[2]]
    ## AnvlArray
    ##  1
    ## [ CPUf32?{} ] 
    ## 
    ## [[3]]
    ## AnvlArray
    ##  1
    ## [ CPUbool{} ]

However, because this is just a guess, they behave differently than
known types during promotion. Therefore, the `common_dtype` function has
two arguments indicating which of the data types are ambiguous. Below,
the first type is a known `f64` and the second is an ambiguous `f32`.
Within anvl, we denote the latter as `i32?`. The result is an `f64`,
although we would promote to an `f64` if both were known. If both types
are ambiguous, the result is generally the same as if both were known.

``` r

common_dtype("f32", "f64", FALSE, TRUE)
```

    ## $dtype
    ## <f32>
    ## 
    ## $ambiguous
    ## [1] FALSE

``` r

common_dtype("f32", "f64", TRUE, TRUE)
```

    ## $dtype
    ## <f64>
    ## 
    ## $ambiguous
    ## [1] TRUE

``` r

common_dtype("f32", "f64", FALSE, FALSE)
```

    ## $dtype
    ## <f64>
    ## 
    ## $ambiguous
    ## [1] FALSE

The promotion rules only change when one type is ambiguous and the other
is not. There, we usually promote the ambiguous type to the known type,
unless:

1.  The ambiguous type is a float and the known type is not.
2.  The known type is a bool but the ambiguous type is not.

In both case, we promote the known type to the default type of the
ambiguous type. The table below shows the promotion rules, where the
rows are ambiguous and the columns are known.

|      | bool | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
|:-----|:-----|:----|:----|:----|:----|:----|:-----|:-----|:-----|:----|:----|
| bool | i1   | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i8   | i8   | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i16  | i16  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i32  | i32  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i64  | i64  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui8  | ui8  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui16 | ui16 | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui32 | ui32 | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui64 | ui64 | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| f32  | f32  | f32 | f32 | f32 | f32 | f32 | f32  | f32  | f32  | f32 | f64 |
| f64  | f64  | f64 | f64 | f64 | f64 | f64 | f64  | f64  | f64  | f32 | f64 |

Promotion rules: ambiguous (row) × known (column) {.table}

## Creating Tensors with Different Ambiguity

Both
[`nv_scalar()`](https://r-xla.github.io/anvl/reference/AnvlArray.md) and
[`nv_array()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
create **non-ambiguous** arrays by default. You can explicitly control
ambiguity using the `ambiguous` parameter:

``` r

s1 <- nv_scalar(1.0)
ambiguous(s1)
```

    ## [1] FALSE

``` r

s2 <- nv_scalar(1.0, ambiguous = TRUE)
ambiguous(s2)
```

    ## [1] TRUE

``` r

t1 <- nv_array(c(1.0, 2.0, 3.0))
ambiguous(t1)
```

    ## [1] FALSE

``` r

t2 <- nv_array(c(1.0, 2.0, 3.0), ambiguous = TRUE)
ambiguous(t2)
```

    ## [1] TRUE

## Propagating Ambiguity

Ambiguity is propagated through operations. Consider the following
example:

``` r

f <- jit(function(x, y) {
  z <- x + 1L
  z * y
})
f(nv_scalar(TRUE), nv_scalar(2L, dtype = "i16"))
```

    ## AnvlArray
    ##  4
    ## [ CPUi16{} ]

The type of `z` is `i32?`, because `x` is promoted to an `i32`, the
default type of the `1L` literal. If `z` was not ambiguous, the output
would be an `i32`, because the `y` would be promoted to an `i32` in the
multiplication. Because we propagate the ambiguity, the `z` is actually
down-promoted to an `i16`, because the `z` is ambiguous, while the `y`
is known.
