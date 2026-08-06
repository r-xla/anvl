# AnvlArray

The main array object. Its type is determined by a data type and a
shape.

## Usage

``` r
nv_array(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  ambiguous = NULL,
  backend = NULL,
  byrow = FALSE,
  check = FALSE
)

nv_scalar(
  data,
  dtype = NULL,
  device = NULL,
  ambiguous = NULL,
  backend = NULL,
  check = FALSE
)

nv_matrix(
  data,
  nrow = NULL,
  ncol = NULL,
  dtype = NULL,
  device = NULL,
  ambiguous = NULL,
  backend = NULL,
  byrow = FALSE
)

nv_empty(dtype, shape, device = NULL, ambiguous = FALSE, backend = NULL)

nv_array_like(
  like,
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  ambiguous = NULL,
  backend = NULL
)

nv_scalar_like(
  like,
  data,
  dtype = NULL,
  device = NULL,
  ambiguous = NULL,
  backend = NULL
)

nv_empty_like(
  like,
  dtype = NULL,
  shape = NULL,
  device = NULL,
  ambiguous = NULL
)
```

## Arguments

- data:

  (any)  
  [`integer()`](https://rdrr.io/r/base/integer.html),
  [`double()`](https://rdrr.io/r/base/double.html), or
  [`logical()`](https://rdrr.io/r/base/logical.html) scalar, vector, or
  array.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  One of bool, i8, i16, i32, i64, ui8, ui16, ui32, ui64, f32, f64 or a
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html).
  The default (`NULL`) uses the current backend's default dtype: `f32`
  for numeric data on `"pjrt"`, `f64` for numeric data on `"quickr"`,
  `i32` for integer data, and `bool` for logical data.

- device:

  (`NULL` \| `character(1)` \|
  [device](https://r-xla.github.io/anvl/dev/reference/nv_device.md))  
  The device the data lives on, given either as:

  - a *device string* naming the platform (e.g. `"cpu"`, `"cuda"`,
    `"cuda:<n>"`), which is resolved against the backend in use, or

  - a *device object* as returned by
    [`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md):
    a
    [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html)
    for the `"pjrt"` backend or a
    [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
    for the `"quickr"` backend. Because a device object is
    backend-specific, it also determines the backend.

  The default (`NULL`) uses
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md):
  the CPU, or the platform named by the `PJRT_PLATFORM` environment
  variable on the `"pjrt"` backend.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The output shape of the array. The default (`NULL`) is to infer it
  from the data if possible. Note that `nv_array` interprets length 1
  vectors as having shape `(1)`. To create a "scalar" with no axes
  (shape `()`), use `nv_scalar` or explicitly specify `shape = c()`.

- ambiguous:

  (`NULL` \| `logical(1)`)  
  Whether the dtype should be marked as ambiguous. Defaults to `FALSE`
  for new arrays.

- backend:

  (`NULL` \| `character(1)`)  
  Backend the array belongs to (`"pjrt"` or `"quickr"`). The default
  (`NULL`) is inferred from `device` when `device` is a backend-specific
  device object, and otherwise falls back to
  [`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md).
  Must not be specified inside
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

- byrow:

  (`logical(1)`)  
  When constructing from an R object and the result has at least two
  axes, fill the array in row-major order rather than the default
  column-major order, mirroring
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html)'s `byrow`. Only
  allowed when `data` is an R object — passing an existing `AnvlArray`
  together with `byrow = TRUE` is an error.

- check:

  (`logical(1)`)  
  If `TRUE`, error when `data` contains any `NA` values. XLA has no
  representation for missing values, so they are otherwise silently
  coerced to the closest available value of the target dtype (e.g. `NaN`
  for floats, the bit pattern `-2147483648` for `i32`, `TRUE` for
  `bool`). Defaults to `FALSE`. See the "Gotchas" vignette.

- nrow:

  (`NULL` \| `integer(1)`)  
  Number of rows. Inferred from `ncol` and the data length if `NULL`.
  Defaults to `1` when `data` is a scalar.

- ncol:

  (`NULL` \| `integer(1)`)  
  Number of columns. Inferred from `nrow` and the data length if `NULL`.
  Defaults to `1` when `data` is a scalar.

- like:

  (`AnvlArray`)  
  An existing array. Any of `dtype`, `device`, `shape`, `ambiguous`, and
  `backend` that are `NULL` (the default) are taken from `like`.

## Value

(`AnvlArray`)

## Terminology

An array's **axes** are the indices that identify its directions,
numbered `1`, `2`, `3`, ... The **size** of an axis (its *axis size*) is
the extent along that axis, and the **shape** is the vector of all axis
sizes. For example, `nv_array(1:6, shape = c(2, 3))` has two axes; the
size of axis `1` is `2` and the size of axis `2` is `3`, so its shape is
`c(2, 3)`. Use
[`naxes()`](https://r-xla.github.io/tengen/reference/naxes.html) for the
number of axes and
[`shape()`](https://r-xla.github.io/tengen/reference/shape.html) for the
axis sizes. We speak of the *size of an axis* rather than an array's
"dimensions", as the latter is generally overloaded as it is used to
refer to both the axis and it's size.

## Extractors

The following generic functions can be used to extract information from
an `AnvlArray`:

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html): Get
  the data type of the array.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html): Get
  the shape (axis sizes) of the array.

- [`naxes()`](https://r-xla.github.io/tengen/reference/naxes.html): Get
  the number of axes.

- [`device()`](https://r-xla.github.io/tengen/reference/device.html):
  Get the device of the array.

- [`platform()`](https://r-xla.github.io/anvl/dev/reference/platform.md):
  Get the platform (e.g. `"cpu"`, `"cuda"`).

- [`ambiguous()`](https://r-xla.github.io/anvl/dev/reference/ambiguous.md):
  Get whether the dtype is ambiguous.

## Serialization

Arrays can be serialized to and from the
[safetensors](https://huggingface.co/docs/safetensors/index) format:

- [`nv_save()`](https://r-xla.github.io/anvl/dev/reference/nv_save.md) /
  [`nv_read()`](https://r-xla.github.io/anvl/dev/reference/nv_read.md):
  Save/load arrays to/from a file.

- [`nv_serialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md)
  /
  [`nv_unserialize()`](https://r-xla.github.io/anvl/dev/reference/nv_unserialize.md):
  Serialize/deserialize arrays to/from raw vectors.

## Backend

An `AnvlArray` is backend-dependent: it belongs to exactly one backend
(`"pjrt"` or the experimental `"quickr"`) and lives on a device of that
backend. The supported data types and devices differ between backends.

## See also

[nv_fill](https://r-xla.github.io/anvl/dev/reference/nv_fill.md),
[nv_iota](https://r-xla.github.io/anvl/dev/reference/nv_iota.md),
[nv_seq](https://r-xla.github.io/anvl/dev/reference/nv_seq.md),
[as_array](https://r-xla.github.io/anvl/dev/reference/as_array.md),
[nv_serialize](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md)

## Examples

``` r
# A 1-d array (vector) with shape (4). Default type for integers is `i32`
nv_array(1:4)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUi32{4} ] 

# Specify a dtype
nv_array(c(1.5, 2.5, 3.5), dtype = "f64")
#> AnvlArray
#>  1.5000
#>  2.5000
#>  3.5000
#> [ CPUf64{3} ] 

# A 2x3 matrix
nv_array(1:6, shape = c(2L, 3L))
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 

# A 2x3 matrix filled by row, like `matrix(1:6, 2, 3, byrow = TRUE)`.
nv_array(1:6, shape = c(2L, 3L), byrow = TRUE)
#> AnvlArray
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 

# A scalar array.
nv_scalar(3.14)
#> AnvlArray
#>  3.1400
#> [ CPUf32{} ] 

# An uninitialized 2x3 array (contents are unspecified)
nv_empty("f32", shape = c(2L, 3L))
#> AnvlArray
#>  2.1019e-44 2.2421e-44 2.6625e-44
#>  2.9427e-44 3.0829e-44 3.0852e-41
#> [ CPUf32{2,3} ] 

# --- Extractors ---
x <- nv_array(1:6, shape = c(2L, 3L))
dtype(x)
#> <i32>
shape(x)
#> [1] 2 3
naxes(x)
#> [1] 2
device(x)
#> <CpuDevice(id=0)>
platform(x)
#> [1] "cpu"
ambiguous(x)
#> [1] FALSE

# --- Transforming arrays with jit ---
add_one <- jit(function(x) x + 1)
add_one(nv_array(1:4))
#> AnvlArray
#>  2
#>  3
#>  4
#>  5
#> [ CPUf32?{4} ] 

# --- Eager mode (calling operations directly) ---
nv_add(nv_array(1:3), nv_array(4:6))
#> AnvlArray
#>  5
#>  7
#>  9
#> [ CPUi32{3} ] 
```
