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
  for numeric data on `"xla"`, `f64` for numeric data on `"quickr"`,
  `i32` for integer data, and `bool` for logical data.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device for the array (`"cpu"`, `"cuda"`). Default is to use the
  CPU for new arrays. This can be changed by setting the `PJRT_PLATFORM`
  environment variable.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The output shape of the array. The default (`NULL`) is to infer it
  from the data if possible. Note that `nv_array` interprets length 1
  vectors as having shape `(1)`. To create a "scalar" with dimension
  `()`, use `nv_scalar` or explicitly specify `shape = c()`.

- ambiguous:

  (`NULL` \| `logical(1)`)  
  Whether the dtype should be marked as ambiguous. Defaults to `FALSE`
  for new arrays.

- backend:

  (`NULL` \| `character(1)`)  
  Backend to use (`"xla"` or `"quickr"`). Defaults to
  [`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md).
  Must not be specified inside
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

- byrow:

  (`logical(1)`)  
  When constructing from an R object and the result has at least two
  dimensions, fill the array in row-major order rather than the default
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

## Extractors

The following generic functions can be used to extract information from
an `AnvlArray`:

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html): Get
  the data type of the array.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html): Get
  the shape (dimensions) of the array.

- [`ndims()`](https://r-xla.github.io/tengen/reference/ndims.html): Get
  the number of dimensions.

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

# An uninitialized 2x3 array (contents are unspecified). Useful as a
# placeholder for outputs of jitted functions when donating buffers.
nv_empty("f32", shape = c(2L, 3L))
#> AnvlArray
#>  7.1655e-05 3.0785e-41 7.1654e-05
#>  3.0785e-41 8.0998e-05 3.0785e-41
#> [ CPUf32{2,3} ] 

# --- Extractors ---
x <- nv_array(1:6, shape = c(2L, 3L))
dtype(x)
#> <i32>
shape(x)
#> [1] 2 3
ndims(x)
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
