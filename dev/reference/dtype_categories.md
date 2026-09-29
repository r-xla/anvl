# Data Type Categories

For promotion, every data type belongs to one of three categories,
ordered boolean \< integer \< float:

- **boolean** – `bool`

- **integer** – `i8`, `i16`, `i32`, `i64` and their unsigned
  counterparts `ui8`, `ui16`, `ui32`, `ui64`

- **float** – `f32` and `f64`.

These are the categories promotion works in, where signed and unsigned
integers count as one.
[`xlamisc::dtype_category()`](https://r-xla.github.io/xlamisc/reference/dtype_category.html)
reports a finer split that names `int` and `uint` separately.

## Data Type Vocabulary

Where a page says which data types an argument takes, it names a group
of them with a single word:

- *any data type* – all of them: `bool`, the signed and unsigned
  integers, and the floats.

- *numeric* – signed integer, unsigned integer and float.

- *integer* – signed and unsigned integer.

- *integerish* – boolean and integer, signed or unsigned.

- *signed numeric* – signed integer and float.

## Data Types for R Values

An R value has no data type of its own. Within its own category, it
takes the data type of the array it meets, and is built at it directly
rather than converted to it, which is what keeps
`nv_scalar(1, "f64") / sqrt(2)` exact. Where it meets nothing, it
settles on the default of its category, which
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
reports and the `anvl.default_dtypes` option configures.
[`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md)
reports the data type a given R value would take.

The same defaults are used wherever a function needs a data type that
its inputs do not give it.
[`nv_exp()`](https://r-xla.github.io/anvl/dev/reference/nv_exp.md)
computes an integer input at the default float, and
[`nv_add()`](https://r-xla.github.io/anvl/dev/reference/nv_add.md) of an
integer array and a double R value promotes to the default float.

The primitives require operands that have a data type to agree on it;
the `nv_*` functions promote them to a common one.

## See also

[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md),
[`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md),
[`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md),
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md),
the [Data Types and Promotion
Rules](https://r-xla.github.io/anvl/articles/type-promotion.html)
article

## Examples

``` r
x <- nv_array(1:3, dtype = "i16")
# an R value takes the data type of the array it meets
dtype(x + 1L)
#> <i16>
# and settles on the default of its category when it meets nothing
dtype(nv_add(1L, 2L))
#> <i32>
peek_dtype(1)
#> <f32>
# a double meeting an integer array promotes to the default float
dtype(x + 0.5)
#> <f32>
# an integer input to a float function computes at the default float
dtype(nv_exp(x))
#> <f32>
with_default_dtypes(c(float = "f64"), dtype(nv_exp(x)))
#> <f64>
```
