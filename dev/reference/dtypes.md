# Data Type Categories

For promotion, every data type belongs to one of three categories,
ordered boolean \< integer \< float:

- **boolean** – `bool`

- **integer** – `i8`, `i16`, `i32`, `i64` and their unsigned
  counterparts `ui8`, `ui16`, `ui32`, `ui64`

- **float** – `f32` and `f64`.

These are the categories promotion works in, where signed and unsigned
integers count as one.
[`tengen::dtype_category()`](https://r-xla.github.io/tengen/reference/dtype_category.html)
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

- *float* – the whole float category: `f32` and `f64`.

- *boolean* – `bool`, the only member of its category.

## Where a Data Type Comes From

An R value has no data type of its own. Where nothing in the program
says which one it should take, it materializes at the default of its
category, which
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
reports and the `anvl.default_dtypes` option configures.
[`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md)
reports the default a given R value would materialize at.

The same defaults settle the data type of a result anvl chooses on its
own, where no R value is involved at all: an index
([`nv_which_max()`](https://r-xla.github.io/anvl/dev/reference/nv_which_max.md),
[`nv_order()`](https://r-xla.github.io/anvl/dev/reference/nv_order.md),
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
the cumulative extrema,
[`nv_lu()`](https://r-xla.github.io/anvl/dev/reference/nv_lu.md)'s
pivots), the accumulator a boolean input is counted at
([`nv_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_sum.md),
[`nv_prod()`](https://r-xla.github.io/anvl/dev/reference/nv_prod.md),
[`nv_cumsum()`](https://r-xla.github.io/anvl/dev/reference/nv_cumsum.md),
[`nv_cumprod()`](https://r-xla.github.io/anvl/dev/reference/nv_cumprod.md),
[`nv_trace()`](https://r-xla.github.io/anvl/dev/reference/nv_trace.md)),
and the float a non-float input is averaged or interpolated at
([`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md),
[`nv_var()`](https://r-xla.github.io/anvl/dev/reference/nv_var.md),
[`nv_sd()`](https://r-xla.github.io/anvl/dev/reference/nv_sd.md),
[`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md),
[`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md)).

Within its own category an R value assumes the data type it meets
instead, and is built at it directly rather than converted to it, which
is what keeps `nv_scalar(1, "f64") / sqrt(2)` exact. The primitives
require operands that have a data type to agree on it; the `nv_*`
functions promote them to a common one.

## See also

[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md),
[`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md),
[`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md),
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md),
[`vignette("type-promotion")`](https://r-xla.github.io/anvl/dev/articles/type-promotion.md)
