# Package index

## Array

Constructing and working with AnvlArrays

### Construction

Functions for creating and initializing arrays

- [`nv_array()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  [`nv_scalar()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  [`nv_matrix()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  [`nv_empty()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  [`nv_array_like()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  [`nv_scalar_like()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  [`nv_empty_like()`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  : AnvlArray
- [`nv_fill()`](https://r-xla.github.io/anvl/reference/nv_fill.md)
  [`nv_fill_like()`](https://r-xla.github.io/anvl/reference/nv_fill.md)
  : Fill Constant
- [`nv_iota()`](https://r-xla.github.io/anvl/reference/nv_iota.md)
  [`nv_iota_like()`](https://r-xla.github.io/anvl/reference/nv_iota.md)
  : Iota
- [`nv_seq()`](https://r-xla.github.io/anvl/reference/nv_seq.md)
  [`nv_seq_like()`](https://r-xla.github.io/anvl/reference/nv_seq.md) :
  Sequence
- [`nv_linspace()`](https://r-xla.github.io/anvl/reference/nv_linspace.md)
  [`nv_linspace_like()`](https://r-xla.github.io/anvl/reference/nv_linspace.md)
  : Evenly Spaced Sequence
- [`nv_diag()`](https://r-xla.github.io/anvl/reference/nv_diag.md) :
  Diagonal Matrix
- [`nv_eye()`](https://r-xla.github.io/anvl/reference/nv_eye.md)
  [`nv_eye_like()`](https://r-xla.github.io/anvl/reference/nv_eye.md) :
  Identity Matrix
- [`as_anvl_array()`](https://r-xla.github.io/anvl/reference/as_anvl_array.md)
  [`as_anvl_arrays()`](https://r-xla.github.io/anvl/reference/as_anvl_array.md)
  : Convert to AnvlArray

### Attributes

Functions for querying array properties

- [`backend()`](https://r-xla.github.io/anvl/reference/backend.md) : Get
  Backend Name of an Array
- [`dtype()`](https://r-xla.github.io/anvl/reference/dtype.md) : Get the
  Data Type of an Array
- [`shape()`](https://r-xla.github.io/anvl/reference/shape.md) : Get the
  Shape of an Array
- [`naxes()`](https://r-xla.github.io/anvl/reference/naxes.md) : Get the
  Number of Axes of an Array
- [`axes()`](https://r-xla.github.io/anvl/reference/axes.md) : Get the
  Axes of an Array
- [`device()`](https://r-xla.github.io/anvl/reference/device.md) : Get
  the Device of an Array
- [`platform()`](https://r-xla.github.io/anvl/reference/platform.md) :
  Get the Platform Name of an Array or Buffer

### Converters

Functions for converting arrays

- [`as_array()`](https://r-xla.github.io/anvl/reference/as_array.md) :
  Convert to an R Array
- [`as_raw()`](https://r-xla.github.io/anvl/reference/as_raw.md) :
  Convert an Array to a Raw Vector
- [`as.double(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/as-AnvlArray.md)
  [`as.integer(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/as-AnvlArray.md)
  [`as.integer64(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/as-AnvlArray.md)
  [`as.logical(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/as-AnvlArray.md)
  [`as.vector(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/as-AnvlArray.md)
  : Coerce AnvlArray to an R Vector

### Serialization

Functions for serializing and deserializing arrays

- [`nv_save()`](https://r-xla.github.io/anvl/reference/nv_save.md)
  [`nv_read()`](https://r-xla.github.io/anvl/reference/nv_save.md) :
  Save and Read Arrays in a File
- [`nv_serialize()`](https://r-xla.github.io/anvl/reference/nv_serialize.md)
  [`nv_unserialize()`](https://r-xla.github.io/anvl/reference/nv_serialize.md)
  : Serialize Arrays to Raw Bytes

### Miscellaneous

- [`is_arrayish()`](https://r-xla.github.io/anvl/reference/arrayish.md)
  : Array-Like Objects
- [`await()`](https://r-xla.github.io/anvl/reference/await.md) : Block
  Until an Async Operation Completes

### Base R Generics

Methods for base R generics that have no `nv_*` twin. Every other
generic is documented together with the `nv_*` function it delegates to.

- [`dim(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/dim.AnvlArray.md)
  : Shape of an Array
- [`length(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/length.AnvlArray.md)
  : Number of Elements

## API Functions

User-facing `nv_*` functions for array operations

### Type Conversion and Broadcasting

Functions for converting data types and broadcasting shapes

- [`nv_convert()`](https://r-xla.github.io/anvl/reference/nv_convert.md)
  : Convert Data Type
- [`nv_bitcast_convert()`](https://r-xla.github.io/anvl/reference/nv_bitcast_convert.md)
  : Bitcast Conversion
- [`nv_broadcast_scalars()`](https://r-xla.github.io/anvl/reference/nv_broadcast_scalars.md)
  : Broadcast Scalars to Common Shape
- [`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/reference/nv_broadcast_arrays.md)
  : Broadcast Arrays to a Common Shape
- [`nv_broadcast_to()`](https://r-xla.github.io/anvl/reference/nv_broadcast_to.md)
  : Broadcast to Shape

### Array Manipulation

Functions for reshaping and rearranging arrays

- [`nv_reshape()`](https://r-xla.github.io/anvl/reference/nv_reshape.md)
  : Reshape
- [`nv_flatten()`](https://r-xla.github.io/anvl/reference/nv_flatten.md)
  : Flatten
- [`nv_aperm()`](https://r-xla.github.io/anvl/reference/nv_aperm.md)
  [`nv_transpose()`](https://r-xla.github.io/anvl/reference/nv_aperm.md)
  [`t(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_aperm.md)
  [`aperm(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_aperm.md)
  : Transpose
- [`nv_concatenate()`](https://r-xla.github.io/anvl/reference/nv_concatenate.md)
  [`c(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_concatenate.md)
  : Concatenate
- [`nv_rbind()`](https://r-xla.github.io/anvl/reference/nv_bind.md)
  [`nv_cbind()`](https://r-xla.github.io/anvl/reference/nv_bind.md)
  [`rbind(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_bind.md)
  [`cbind(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_bind.md)
  : Combine Arrays by Rows or Columns
- [`nv_static_slice()`](https://r-xla.github.io/anvl/reference/nv_static_slice.md)
  : Static Slice
- [`nv_select()`](https://r-xla.github.io/anvl/reference/nv_select.md) :
  Select Elements Along an Axis
- [`nv_pad()`](https://r-xla.github.io/anvl/reference/nv_pad.md) : Pad
- [`nv_rev()`](https://r-xla.github.io/anvl/reference/nv_rev.md)
  [`rev(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_rev.md)
  : Reverse
- [`nv_squeeze()`](https://r-xla.github.io/anvl/reference/nv_squeeze.md)
  [`nv_drop()`](https://r-xla.github.io/anvl/reference/nv_squeeze.md) :
  Squeeze
- [`nv_unsqueeze()`](https://r-xla.github.io/anvl/reference/nv_unsqueeze.md)
  : Unsqueeze
- [`` `[`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_subset.md)
  [`nv_subset()`](https://r-xla.github.io/anvl/reference/nv_subset.md) :
  Subset an Array
- [`` `[<-`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_subset_assign.md)
  [`nv_subset_assign()`](https://r-xla.github.io/anvl/reference/nv_subset_assign.md)
  : Update Subset

### Arithmetic Operations

Basic arithmetic operations on arrays

- [`nv_add()`](https://r-xla.github.io/anvl/reference/nv_add.md)
  [`` `+`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_add.md)
  : Addition
- [`nv_sub()`](https://r-xla.github.io/anvl/reference/nv_sub.md)
  [`` `-`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sub.md)
  : Subtraction
- [`nv_mul()`](https://r-xla.github.io/anvl/reference/nv_mul.md)
  [`` `*`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_mul.md)
  : Multiplication
- [`nv_div()`](https://r-xla.github.io/anvl/reference/nv_div.md)
  [`` `/`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_div.md)
  : Division
- [`nv_pow()`](https://r-xla.github.io/anvl/reference/nv_pow.md)
  [`` `^`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_pow.md)
  : Power
- [`nv_negate()`](https://r-xla.github.io/anvl/reference/nv_negate.md) :
  Negation
- [`nv_remainder()`](https://r-xla.github.io/anvl/reference/nv_remainder.md)
  : Remainder (Truncating)
- [`nv_mod()`](https://r-xla.github.io/anvl/reference/nv_mod.md)
  [`` `%%`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_mod.md)
  : Modulo (Flooring Remainder)
- [`nv_floor_div()`](https://r-xla.github.io/anvl/reference/nv_floor_div.md)
  [`` `%/%`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_floor_div.md)
  : Flooring Division

### Comparison Operations

Element-wise comparison operations

- [`nv_eq()`](https://r-xla.github.io/anvl/reference/nv_eq.md)
  [`` `==`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_eq.md)
  : Equal
- [`nv_ne()`](https://r-xla.github.io/anvl/reference/nv_ne.md)
  [`` `!=`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_ne.md)
  : Not Equal
- [`nv_gt()`](https://r-xla.github.io/anvl/reference/nv_gt.md)
  [`` `>`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_gt.md)
  : Greater Than
- [`nv_ge()`](https://r-xla.github.io/anvl/reference/nv_ge.md)
  [`` `>=`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_ge.md)
  : Greater Than or Equal
- [`nv_lt()`](https://r-xla.github.io/anvl/reference/nv_lt.md)
  [`` `<`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_lt.md)
  : Less Than
- [`nv_le()`](https://r-xla.github.io/anvl/reference/nv_le.md)
  [`` `<=`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_le.md)
  : Less Than or Equal

### Mathematical Functions

Mathematical and trigonometric functions

- [`nv_abs()`](https://r-xla.github.io/anvl/reference/nv_abs.md)
  [`abs(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_abs.md)
  : Absolute Value
- [`nv_sqrt()`](https://r-xla.github.io/anvl/reference/nv_sqrt.md)
  [`sqrt(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sqrt.md)
  : Square Root
- [`nv_rsqrt()`](https://r-xla.github.io/anvl/reference/nv_rsqrt.md) :
  Reciprocal Square Root
- [`nv_cbrt()`](https://r-xla.github.io/anvl/reference/nv_cbrt.md) :
  Cube Root
- [`nv_exp()`](https://r-xla.github.io/anvl/reference/nv_exp.md)
  [`exp(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_exp.md)
  : Exponential
- [`nv_expm1()`](https://r-xla.github.io/anvl/reference/nv_expm1.md)
  [`expm1(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_expm1.md)
  : Exponential Minus One
- [`nv_log()`](https://r-xla.github.io/anvl/reference/nv_log.md)
  [`log(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_log.md)
  : Natural Logarithm
- [`nv_log1p()`](https://r-xla.github.io/anvl/reference/nv_log1p.md)
  [`log1p(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_log1p.md)
  : Log Plus One
- [`nv_log2()`](https://r-xla.github.io/anvl/reference/nv_log2.md)
  [`log2(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_log2.md)
  : Base-2 Logarithm
- [`nv_log10()`](https://r-xla.github.io/anvl/reference/nv_log10.md)
  [`log10(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_log10.md)
  : Base-10 Logarithm
- [`nv_sin()`](https://r-xla.github.io/anvl/reference/nv_sin.md)
  [`sin(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sin.md)
  : Sine
- [`nv_cos()`](https://r-xla.github.io/anvl/reference/nv_cos.md)
  [`cos(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_cos.md)
  : Cosine
- [`nv_tan()`](https://r-xla.github.io/anvl/reference/nv_tan.md)
  [`tan(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_tan.md)
  : Tangent
- [`nv_sinpi()`](https://r-xla.github.io/anvl/reference/nv_sinpi.md)
  [`sinpi(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sinpi.md)
  : Sine of a Multiple of Pi
- [`nv_cospi()`](https://r-xla.github.io/anvl/reference/nv_cospi.md)
  [`cospi(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_cospi.md)
  : Cosine of a Multiple of Pi
- [`nv_tanpi()`](https://r-xla.github.io/anvl/reference/nv_tanpi.md)
  [`tanpi(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_tanpi.md)
  : Tangent of a Multiple of Pi
- [`nv_asin()`](https://r-xla.github.io/anvl/reference/nv_asin.md)
  [`asin(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_asin.md)
  : Arc Sine
- [`nv_acos()`](https://r-xla.github.io/anvl/reference/nv_acos.md)
  [`acos(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_acos.md)
  : Arc Cosine
- [`nv_atan()`](https://r-xla.github.io/anvl/reference/nv_atan.md)
  [`atan(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_atan.md)
  : Arc Tangent
- [`nv_atan2()`](https://r-xla.github.io/anvl/reference/nv_atan2.md) :
  Arctangent 2
- [`nv_sinh()`](https://r-xla.github.io/anvl/reference/nv_sinh.md)
  [`sinh(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sinh.md)
  : Hyperbolic Sine
- [`nv_cosh()`](https://r-xla.github.io/anvl/reference/nv_cosh.md)
  [`cosh(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_cosh.md)
  : Hyperbolic Cosine
- [`nv_tanh()`](https://r-xla.github.io/anvl/reference/nv_tanh.md)
  [`tanh(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_tanh.md)
  : Hyperbolic Tangent
- [`nv_asinh()`](https://r-xla.github.io/anvl/reference/nv_asinh.md)
  [`asinh(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_asinh.md)
  : Inverse Hyperbolic Sine
- [`nv_acosh()`](https://r-xla.github.io/anvl/reference/nv_acosh.md)
  [`acosh(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_acosh.md)
  : Inverse Hyperbolic Cosine
- [`nv_atanh()`](https://r-xla.github.io/anvl/reference/nv_atanh.md)
  [`atanh(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_atanh.md)
  : Inverse Hyperbolic Tangent
- [`nv_sign()`](https://r-xla.github.io/anvl/reference/nv_sign.md)
  [`sign(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sign.md)
  : Sign
- [`nv_floor()`](https://r-xla.github.io/anvl/reference/nv_floor.md)
  [`floor(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_floor.md)
  : Floor
- [`nv_ceiling()`](https://r-xla.github.io/anvl/reference/nv_ceiling.md)
  [`ceiling(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_ceiling.md)
  : Ceiling
- [`nv_trunc()`](https://r-xla.github.io/anvl/reference/nv_trunc.md)
  [`trunc(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_trunc.md)
  : Truncate
- [`nv_round()`](https://r-xla.github.io/anvl/reference/nv_round.md) :
  Round
- [`nv_plogis()`](https://r-xla.github.io/anvl/reference/nv_plogis.md) :
  Logistic (Sigmoid)
- [`nv_erf()`](https://r-xla.github.io/anvl/reference/nv_erf.md) : Error
  Function
- [`nv_erfc()`](https://r-xla.github.io/anvl/reference/nv_erfc.md) :
  Complementary Error Function
- [`nv_erf_inv()`](https://r-xla.github.io/anvl/reference/nv_erf_inv.md)
  : Inverse Error Function
- [`nv_gamma()`](https://r-xla.github.io/anvl/reference/nv_gamma.md)
  [`gamma(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_gamma.md)
  : Gamma Function
- [`nv_digamma()`](https://r-xla.github.io/anvl/reference/nv_digamma.md)
  [`digamma(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_digamma.md)
  : Digamma
- [`nv_lgamma()`](https://r-xla.github.io/anvl/reference/nv_lgamma.md)
  [`lgamma(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_lgamma.md)
  : Log-Gamma
- [`nv_psigamma()`](https://r-xla.github.io/anvl/reference/nv_psigamma.md)
  [`trigamma(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_psigamma.md)
  : Psigamma
- [`nv_is_finite()`](https://r-xla.github.io/anvl/reference/nv_is_finite.md)
  [`is.finite(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_is_finite.md)
  : Is Finite
- [`nv_is_nan()`](https://r-xla.github.io/anvl/reference/nv_is_nan.md)
  [`is.nan(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_is_nan.md)
  : Is NaN
- [`nv_is_infinite()`](https://r-xla.github.io/anvl/reference/nv_is_infinite.md)
  [`is.infinite(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_is_infinite.md)
  : Is Infinite

### Reduction Operations

Operations that reduce array axes

- [`nv_sum()`](https://r-xla.github.io/anvl/reference/nv_sum.md)
  [`sum(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sum.md)
  : Sum Reduction
- [`nv_prod()`](https://r-xla.github.io/anvl/reference/nv_prod.md)
  [`prod(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_prod.md)
  : Product Reduction
- [`nv_max()`](https://r-xla.github.io/anvl/reference/nv_max.md)
  [`max(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_max.md)
  : Max Reduction
- [`nv_min()`](https://r-xla.github.io/anvl/reference/nv_min.md)
  [`min(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_min.md)
  : Min Reduction
- [`nv_range()`](https://r-xla.github.io/anvl/reference/nv_range.md)
  [`range(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_range.md)
  : Range Reduction
- [`nv_any()`](https://r-xla.github.io/anvl/reference/nv_any.md)
  [`any(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_any.md)
  : Any Reduction
- [`nv_all()`](https://r-xla.github.io/anvl/reference/nv_all.md)
  [`all(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_all.md)
  : All Reduction

### Statistical Summaries

Summary statistics over array axes

- [`nv_mean()`](https://r-xla.github.io/anvl/reference/nv_mean.md)
  [`mean(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_mean.md)
  : Mean
- [`nv_median()`](https://r-xla.github.io/anvl/reference/nv_median.md)
  [`median(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_median.md)
  : Median
- [`nv_quantile()`](https://r-xla.github.io/anvl/reference/nv_quantile.md)
  [`quantile(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_quantile.md)
  : Quantile
- [`nv_var()`](https://r-xla.github.io/anvl/reference/nv_var.md) :
  Variance
- [`nv_sd()`](https://r-xla.github.io/anvl/reference/nv_sd.md) :
  Standard Deviation

### Cumulative Operations

Cumulative (scan) operations along a single axis

- [`nv_cumsum()`](https://r-xla.github.io/anvl/reference/nv_cumsum.md)
  [`cumsum(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_cumsum.md)
  : Cumulative Sum
- [`nv_cumprod()`](https://r-xla.github.io/anvl/reference/nv_cumprod.md)
  [`cumprod(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_cumprod.md)
  : Cumulative Product
- [`nv_cummax()`](https://r-xla.github.io/anvl/reference/nv_cummax.md)
  [`cummax(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_cummax.md)
  : Cumulative Maximum
- [`nv_cummin()`](https://r-xla.github.io/anvl/reference/nv_cummin.md)
  [`cummin(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_cummin.md)
  : Cumulative Minimum

### Linear Algebra

Linear algebra operations

- [`nv_matmul()`](https://r-xla.github.io/anvl/reference/nv_matmul.md)
  [`` `%*%`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_matmul.md)
  : Matrix Multiplication
- [`nv_chol()`](https://r-xla.github.io/anvl/reference/nv_chol.md)
  [`chol(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_chol.md)
  : Cholesky Decomposition
- [`nv_qr()`](https://r-xla.github.io/anvl/reference/nv_qr.md)
  [`qr(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_qr.md)
  : QR Decomposition
- [`nv_lu()`](https://r-xla.github.io/anvl/reference/nv_lu.md) : LU
  Decomposition
- [`nv_svd()`](https://r-xla.github.io/anvl/reference/nv_svd.md) :
  Singular Value Decomposition
- [`nv_eigh()`](https://r-xla.github.io/anvl/reference/nv_eigh.md) :
  Symmetric Eigendecomposition
- [`nv_solve()`](https://r-xla.github.io/anvl/reference/nv_solve.md)
  [`solve(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_solve.md)
  : Solve Linear System
- [`nv_triangular_solve()`](https://r-xla.github.io/anvl/reference/nv_triangular_solve.md)
  : Triangular Solve
- [`nv_inv()`](https://r-xla.github.io/anvl/reference/nv_inv.md) :
  Matrix Inverse
- [`nv_det()`](https://r-xla.github.io/anvl/reference/nv_det.md) :
  Determinant
- [`nv_determinant()`](https://r-xla.github.io/anvl/reference/nv_determinant.md)
  [`determinant(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_determinant.md)
  : Determinant in Modulus/Sign Form
- [`nv_crossprod()`](https://r-xla.github.io/anvl/reference/nv_crossprod.md)
  [`crossprod(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_crossprod.md)
  : Cross Product (Matrix)
- [`nv_tcrossprod()`](https://r-xla.github.io/anvl/reference/nv_tcrossprod.md)
  [`tcrossprod(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_tcrossprod.md)
  : Transpose Cross Product (Matrix)
- [`nv_outer()`](https://r-xla.github.io/anvl/reference/nv_outer.md) :
  Outer Product
- [`nv_trace()`](https://r-xla.github.io/anvl/reference/nv_trace.md) :
  Matrix Trace
- [`nv_extract_diag()`](https://r-xla.github.io/anvl/reference/nv_extract_diag.md)
  : Extract Diagonal
- [`nv_tril()`](https://r-xla.github.io/anvl/reference/nv_tril.md) :
  Lower Triangular Matrix
- [`nv_triu()`](https://r-xla.github.io/anvl/reference/nv_triu.md) :
  Upper Triangular Matrix
- [`nv_lower_tri()`](https://r-xla.github.io/anvl/reference/nv_lower_tri.md)
  [`nv_lower_tri_like()`](https://r-xla.github.io/anvl/reference/nv_lower_tri.md)
  : Lower Triangular Mask
- [`nv_upper_tri()`](https://r-xla.github.io/anvl/reference/nv_upper_tri.md)
  [`nv_upper_tri_like()`](https://r-xla.github.io/anvl/reference/nv_upper_tri.md)
  : Upper Triangular Mask

### Convolution

N-dimensional convolution operations

- [`nv_conv1d()`](https://r-xla.github.io/anvl/reference/nv_conv1d.md) :
  1D Convolution
- [`nv_conv2d()`](https://r-xla.github.io/anvl/reference/nv_conv2d.md) :
  2D Convolution
- [`nv_conv3d()`](https://r-xla.github.io/anvl/reference/nv_conv3d.md) :
  3D Convolution

### Logical and Bitwise Operations

Logical and bitwise operations on arrays

- [`nv_and()`](https://r-xla.github.io/anvl/reference/nv_and.md)
  [`` `&`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_and.md)
  : Bitwise AND
- [`nv_or()`](https://r-xla.github.io/anvl/reference/nv_or.md)
  [`` `|`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_or.md)
  : Bitwise OR
- [`nv_xor()`](https://r-xla.github.io/anvl/reference/nv_xor.md) :
  Bitwise XOR
- [`nv_not()`](https://r-xla.github.io/anvl/reference/nv_not.md)
  [`` `!`( ``*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_not.md)
  : Bitwise Not
- [`nv_shift_left()`](https://r-xla.github.io/anvl/reference/nv_shift_left.md)
  : Shift Left
- [`nv_shift_right_logical()`](https://r-xla.github.io/anvl/reference/nv_shift_right_logical.md)
  : Logical Shift Right
- [`nv_shift_right_arithmetic()`](https://r-xla.github.io/anvl/reference/nv_shift_right_arithmetic.md)
  : Arithmetic Shift Right
- [`nv_popcnt()`](https://r-xla.github.io/anvl/reference/nv_popcnt.md) :
  Population Count

### Element-wise Operations

Other element-wise array operations

- [`nv_pmin()`](https://r-xla.github.io/anvl/reference/nv_pmin.md) :
  Parallel Minimum
- [`nv_pmax()`](https://r-xla.github.io/anvl/reference/nv_pmax.md) :
  Parallel Maximum
- [`nv_clamp()`](https://r-xla.github.io/anvl/reference/nv_clamp.md) :
  Clamp

### Sorting and Searching

Sorting arrays and finding extrema

- [`nv_sort()`](https://r-xla.github.io/anvl/reference/nv_sort.md)
  [`sort(`*`<AnvlArray>`*`)`](https://r-xla.github.io/anvl/reference/nv_sort.md)
  : Sort
- [`nv_order()`](https://r-xla.github.io/anvl/reference/nv_order.md) :
  Sort Order
- [`nv_top_k()`](https://r-xla.github.io/anvl/reference/nv_top_k.md) :
  Top-K Elements
- [`nv_which_max()`](https://r-xla.github.io/anvl/reference/nv_which_max.md)
  : Index of the Maximum
- [`nv_which_min()`](https://r-xla.github.io/anvl/reference/nv_which_min.md)
  : Index of the Minimum

### Control Flow

Control flow operations

- [`nv_if()`](https://r-xla.github.io/anvl/reference/nv_if.md) :
  Conditional Branching
- [`nv_while()`](https://r-xla.github.io/anvl/reference/nv_while.md) :
  While Loop
- [`nv_scan()`](https://r-xla.github.io/anvl/reference/nv_scan.md) :
  Scan (Loop with Per-Step Outputs)
- [`nv_ifelse()`](https://r-xla.github.io/anvl/reference/nv_ifelse.md) :
  Conditional Element Selection

### Distributions

Densities, distribution functions, and samplers for probability
distributions

- [`nv_dnorm()`](https://r-xla.github.io/anvl/reference/nv_normal.md)
  [`nv_pnorm()`](https://r-xla.github.io/anvl/reference/nv_normal.md)
  [`nv_qnorm()`](https://r-xla.github.io/anvl/reference/nv_normal.md)
  [`nv_rnorm()`](https://r-xla.github.io/anvl/reference/nv_normal.md) :
  The Normal Distribution
- [`nv_dunif()`](https://r-xla.github.io/anvl/reference/nv_uniform.md)
  [`nv_punif()`](https://r-xla.github.io/anvl/reference/nv_uniform.md)
  [`nv_qunif()`](https://r-xla.github.io/anvl/reference/nv_uniform.md)
  [`nv_runif()`](https://r-xla.github.io/anvl/reference/nv_uniform.md) :
  The Uniform Distribution
- [`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md) :
  Sample from a Binomial Distribution
- [`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md) :
  Sample from a Population
- [`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md)
  : Sample Integers

### Utilities

Helpers supporting the API functions

- [`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md)
  : Generate RNG State
- [`nv_print()`](https://r-xla.github.io/anvl/reference/nv_print.md) :
  Print Array

## Data Types and Promotion

Data types, their defaults, and the rules for promoting inputs to a
common one

### Data Types

- [`dtype_categories`](https://r-xla.github.io/anvl/reference/dtype_categories.md)
  : Data Type Categories
- [`as_dtype()`](https://r-xla.github.io/anvl/reference/as_dtype.md) :
  Convert to a DataType
- [`is_dtype()`](https://r-xla.github.io/anvl/reference/is_dtype.md) :
  Check If an Object Is a DataType
- [`common_dtype()`](https://r-xla.github.io/anvl/reference/common_dtype.md)
  : Common Data Type
- [`peek_dtype()`](https://r-xla.github.io/anvl/reference/peek_dtype.md)
  : Peek at a Data Type
- [`default_dtypes()`](https://r-xla.github.io/anvl/reference/default_dtypes.md)
  [`default_float()`](https://r-xla.github.io/anvl/reference/default_dtypes.md)
  [`default_int()`](https://r-xla.github.io/anvl/reference/default_dtypes.md)
  : Default Data Types
- [`local_default_dtypes()`](https://r-xla.github.io/anvl/reference/local_default_dtypes.md)
  [`with_default_dtypes()`](https://r-xla.github.io/anvl/reference/local_default_dtypes.md)
  : Set the Default Data Types
- [`with_dtypes()`](https://r-xla.github.io/anvl/reference/with_dtypes.md)
  : Run a Function at Given Data Types

### Promotion Rules

How `nv_*` functions promote their inputs; see the [Data Types and
Promotion
Rules](https://r-xla.github.io/anvl/articles/type-promotion.md) article
for how they are used.

- [`nv_promote_to_common()`](https://r-xla.github.io/anvl/reference/nv_promote_to_common.md)
  : Promote Arrays to a Common Data Type
- [`apply_promotion()`](https://r-xla.github.io/anvl/reference/apply_promotion.md)
  : Bring a Primitive's Operands to One Data Type
- [`promotion_common()`](https://r-xla.github.io/anvl/reference/promotion_rule.md)
  [`promotion_like()`](https://r-xla.github.io/anvl/reference/promotion_rule.md)
  [`promotion_dtype()`](https://r-xla.github.io/anvl/reference/promotion_rule.md)
  [`promotion_rdata_common()`](https://r-xla.github.io/anvl/reference/promotion_rule.md)
  [`promotion_grouped()`](https://r-xla.github.io/anvl/reference/promotion_rule.md)
  [`promotion_rule()`](https://r-xla.github.io/anvl/reference/promotion_rule.md)
  : Promotion Rules

## Transformations

Code transformations

- [`jit()`](https://r-xla.github.io/anvl/reference/jit.md) : JIT Compile
  a Function
- [`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)
  [`value_and_gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)
  : Gradients

## Backend

Backend-related functionality and implementations

- [`backend()`](https://r-xla.github.io/anvl/reference/backend.md) : Get
  Backend Name of an Array
- [`local_backend()`](https://r-xla.github.io/anvl/reference/local_backend.md)
  [`with_backend()`](https://r-xla.github.io/anvl/reference/local_backend.md)
  : Temporarily Set the Backend
- [`AnvlBackendPjrt()`](https://r-xla.github.io/anvl/reference/AnvlBackendPjrt.md)
  : PJRT Backend
- [`AnvlBackendQuickr()`](https://r-xla.github.io/anvl/reference/AnvlBackendQuickr.md)
  : Quickr Backend

## Device

Devices of the active backend and the default device

- [`nv_device()`](https://r-xla.github.io/anvl/reference/nv_device.md) :
  Create a Device
- [`default_device()`](https://r-xla.github.io/anvl/reference/default_device.md)
  : Get the Default Device
- [`local_default_device()`](https://r-xla.github.io/anvl/reference/local_default_device.md)
  [`with_default_device()`](https://r-xla.github.io/anvl/reference/local_default_device.md)
  : Temporarily Set the Default Device
- [`is_device()`](https://r-xla.github.io/anvl/reference/is_device.md) :
  Test Whether an Object Is a Device
- [`quickr_device()`](https://r-xla.github.io/anvl/reference/quickr_device.md)
  : Quickr Device

## Miscellaneous

General-purpose helpers

- [`arr()`](https://r-xla.github.io/anvl/reference/arr.md) : Create an R
  Array
- [`eq_type()`](https://r-xla.github.io/anvl/reference/eq_type.md)
  [`neq_type()`](https://r-xla.github.io/anvl/reference/eq_type.md) :
  Compare AbstractArray Types
- [`map_tree()`](https://r-xla.github.io/anvl/reference/map_tree.md)
  [`pmap_tree()`](https://r-xla.github.io/anvl/reference/map_tree.md) :
  Map a Function over Trees

## Internals

Internal data structures and functions

### Graph

- [`AnvlGraph()`](https://r-xla.github.io/anvl/reference/AnvlGraph.md) :
  Graph of Statements
- [`GraphDescriptor()`](https://r-xla.github.io/anvl/reference/GraphDescriptor.md)
  : Graph Descriptor
- [`GraphNode`](https://r-xla.github.io/anvl/reference/GraphNode.md) :
  Graph Node
- [`GraphValue()`](https://r-xla.github.io/anvl/reference/GraphValue.md)
  : Graph Value
- [`GraphBox()`](https://r-xla.github.io/anvl/reference/GraphBox.md) :
  Graph Box
- [`GraphLiteral()`](https://r-xla.github.io/anvl/reference/GraphLiteral.md)
  : Graph Literal
- [`GraphStatement()`](https://r-xla.github.io/anvl/reference/GraphStatement.md)
  : Graph Statement
- [`local_descriptor()`](https://r-xla.github.io/anvl/reference/local_descriptor.md)
  : Create a Graph
- [`current_descriptor()`](https://r-xla.github.io/anvl/reference/current_descriptor.md)
  : Get the Current Graph
- [`subgraphs()`](https://r-xla.github.io/anvl/reference/subgraphs.md) :
  Get Subgraphs from Higher-Order Primitive

### Abstract Types

- [`nv_aval()`](https://r-xla.github.io/anvl/reference/AbstractArray.md)
  [`AbstractArray()`](https://r-xla.github.io/anvl/reference/AbstractArray.md)
  : Abstract Array Class
- [`ConcreteArray()`](https://r-xla.github.io/anvl/reference/ConcreteArray.md)
  : Concrete Array Class
- [`LiteralArray()`](https://r-xla.github.io/anvl/reference/LiteralArray.md)
  : Literal Array Class
- [`RData()`](https://r-xla.github.io/anvl/reference/RData.md) : R Data
  Class
- [`IotaArray()`](https://r-xla.github.io/anvl/reference/IotaArray.md) :
  Iota Array Class
- [`to_abstract()`](https://r-xla.github.io/anvl/reference/to_abstract.md)
  : Convert to Abstract Array
- [`is_arrayish()`](https://r-xla.github.io/anvl/reference/arrayish.md)
  : Array-Like Objects

### Transformations

- [`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md) :
  Trace an R Function into a Graph
- [`transform_gradient()`](https://r-xla.github.io/anvl/reference/transform_gradient.md)
  : Transform a Graph to Its Gradient
- [`graph_to_quickr_r_function()`](https://r-xla.github.io/anvl/reference/graph_to_quickr_r_function.md)
  : Convert an AnvlGraph to a Plain R Function
- [`compile_pjrt()`](https://r-xla.github.io/anvl/reference/compile_pjrt.md)
  : Trace, Lower, and Compile a Function to an XLA Executable
- [`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md) :
  Lower a Graph to StableHLO
- [`current_platform()`](https://r-xla.github.io/anvl/reference/current_platform.md)
  [`local_platform()`](https://r-xla.github.io/anvl/reference/current_platform.md)
  : Current Lowering Target Platform

### Tree

Utilities for working with nested structures

- [`reexports`](https://r-xla.github.io/anvl/reference/reexports.md)
  [`flatten`](https://r-xla.github.io/anvl/reference/reexports.md)
  [`build_tree`](https://r-xla.github.io/anvl/reference/reexports.md)
  [`unflatten`](https://r-xla.github.io/anvl/reference/reexports.md)
  [`tree_size`](https://r-xla.github.io/anvl/reference/reexports.md)
  [`tree_path`](https://r-xla.github.io/anvl/reference/reexports.md) :
  Objects exported from other packages

### Miscellaneous

- [`AnvlBackend()`](https://r-xla.github.io/anvl/reference/AnvlBackend.md)
  : Create a Backend
- [`active_backend()`](https://r-xla.github.io/anvl/reference/active_backend.md)
  : Get Active Backend
- [`jit_cache_size()`](https://r-xla.github.io/anvl/reference/jit_cache_size.md)
  : Number of Cached Programs of a Jitted Function
- [`Shape()`](https://r-xla.github.io/anvl/reference/Shape-constructor.md)
  : Create a Shape Object

## Primitives

Low-level primitive operations (prim\_\* functions)

### Construction

- [`new_primitive()`](https://r-xla.github.io/anvl/reference/new_primitive.md)
  : Create a Primitive
- [`AnvlPrimitiveDef()`](https://r-xla.github.io/anvl/reference/AnvlPrimitiveDef.md)
  : Primitive Definition
- [`graph_desc_add()`](https://r-xla.github.io/anvl/reference/graph_desc_add.md)
  : Add a Statement to a Graph Descriptor
- [`rule_reverse()`](https://r-xla.github.io/anvl/reference/rule_reverse.md)
  : Reverse Rule

### Implemented Primitives

- [`prim_abs()`](https://r-xla.github.io/anvl/reference/prim_abs.md) :
  Primitive Absolute Value
- [`prim_acos()`](https://r-xla.github.io/anvl/reference/prim_acos.md) :
  Primitive Arc Cosine
- [`prim_acosh()`](https://r-xla.github.io/anvl/reference/prim_acosh.md)
  : Primitive Inverse Hyperbolic Cosine
- [`prim_add()`](https://r-xla.github.io/anvl/reference/prim_add.md) :
  Primitive Addition
- [`prim_all()`](https://r-xla.github.io/anvl/reference/prim_all.md) :
  Primitive All Reduction
- [`prim_and()`](https://r-xla.github.io/anvl/reference/prim_and.md) :
  Primitive Bitwise And
- [`prim_any()`](https://r-xla.github.io/anvl/reference/prim_any.md) :
  Primitive Any Reduction
- [`prim_asin()`](https://r-xla.github.io/anvl/reference/prim_asin.md) :
  Primitive Arc Sine
- [`prim_asinh()`](https://r-xla.github.io/anvl/reference/prim_asinh.md)
  : Primitive Inverse Hyperbolic Sine
- [`prim_atan()`](https://r-xla.github.io/anvl/reference/prim_atan.md) :
  Primitive Arc Tangent
- [`prim_atan2()`](https://r-xla.github.io/anvl/reference/prim_atan2.md)
  : Primitive Arctangent 2
- [`prim_atanh()`](https://r-xla.github.io/anvl/reference/prim_atanh.md)
  : Primitive Inverse Hyperbolic Tangent
- [`prim_bitcast_convert()`](https://r-xla.github.io/anvl/reference/prim_bitcast_convert.md)
  : Primitive Bitcast Conversion
- [`prim_broadcast_in_axes()`](https://r-xla.github.io/anvl/reference/prim_broadcast_in_axes.md)
  : Primitive Broadcast
- [`prim_cbrt()`](https://r-xla.github.io/anvl/reference/prim_cbrt.md) :
  Primitive Cube Root
- [`prim_ceiling()`](https://r-xla.github.io/anvl/reference/prim_ceiling.md)
  : Primitive Ceiling
- [`prim_chol()`](https://r-xla.github.io/anvl/reference/prim_chol.md) :
  Primitive Cholesky Decomposition
- [`prim_clamp()`](https://r-xla.github.io/anvl/reference/prim_clamp.md)
  : Primitive Clamp
- [`prim_concatenate()`](https://r-xla.github.io/anvl/reference/prim_concatenate.md)
  : Primitive Concatenate
- [`prim_convert()`](https://r-xla.github.io/anvl/reference/prim_convert.md)
  : Primitive Convert Data Type
- [`prim_convolution()`](https://r-xla.github.io/anvl/reference/prim_convolution.md)
  : Primitive Convolution
- [`prim_cos()`](https://r-xla.github.io/anvl/reference/prim_cos.md) :
  Primitive Cosine
- [`prim_cosh()`](https://r-xla.github.io/anvl/reference/prim_cosh.md) :
  Primitive Hyperbolic Cosine
- [`prim_cummax()`](https://r-xla.github.io/anvl/reference/prim_cummax.md)
  : Primitive Cumulative Maximum
- [`prim_cummin()`](https://r-xla.github.io/anvl/reference/prim_cummin.md)
  : Primitive Cumulative Minimum
- [`prim_cumprod()`](https://r-xla.github.io/anvl/reference/prim_cumprod.md)
  : Primitive Cumulative Product
- [`prim_cumsum()`](https://r-xla.github.io/anvl/reference/prim_cumsum.md)
  : Primitive Cumulative Sum
- [`prim_digamma()`](https://r-xla.github.io/anvl/reference/prim_digamma.md)
  : Primitive Digamma
- [`prim_div()`](https://r-xla.github.io/anvl/reference/prim_div.md) :
  Primitive Division
- [`prim_dot_general()`](https://r-xla.github.io/anvl/reference/prim_dot_general.md)
  : Primitive Dot General
- [`prim_dynamic_slice()`](https://r-xla.github.io/anvl/reference/prim_dynamic_slice.md)
  : Primitive Dynamic Slice
- [`prim_dynamic_update_slice()`](https://r-xla.github.io/anvl/reference/prim_dynamic_update_slice.md)
  : Primitive Dynamic Update Slice
- [`prim_eigh()`](https://r-xla.github.io/anvl/reference/prim_eigh.md) :
  Primitive Symmetric Eigendecomposition
- [`prim_eq()`](https://r-xla.github.io/anvl/reference/prim_eq.md) :
  Primitive Equal
- [`prim_erf()`](https://r-xla.github.io/anvl/reference/prim_erf.md) :
  Primitive Error Function
- [`prim_erf_inv()`](https://r-xla.github.io/anvl/reference/prim_erf_inv.md)
  : Primitive Inverse Error Function
- [`prim_erfc()`](https://r-xla.github.io/anvl/reference/prim_erfc.md) :
  Primitive Complementary Error Function
- [`prim_exp()`](https://r-xla.github.io/anvl/reference/prim_exp.md) :
  Primitive Exponential
- [`prim_expm1()`](https://r-xla.github.io/anvl/reference/prim_expm1.md)
  : Primitive Exponential Minus One
- [`prim_fill()`](https://r-xla.github.io/anvl/reference/prim_fill.md) :
  Primitive Fill
- [`prim_floor()`](https://r-xla.github.io/anvl/reference/prim_floor.md)
  : Primitive Floor
- [`prim_gather()`](https://r-xla.github.io/anvl/reference/prim_gather.md)
  : Primitive Gather
- [`prim_ge()`](https://r-xla.github.io/anvl/reference/prim_ge.md) :
  Primitive Greater Than or Equal
- [`prim_gt()`](https://r-xla.github.io/anvl/reference/prim_gt.md) :
  Primitive Greater Than
- [`prim_if()`](https://r-xla.github.io/anvl/reference/prim_if.md) :
  Primitive If
- [`prim_ifelse()`](https://r-xla.github.io/anvl/reference/prim_ifelse.md)
  : Primitive Ifelse
- [`prim_iota()`](https://r-xla.github.io/anvl/reference/prim_iota.md) :
  Primitive Iota
- [`prim_is_finite()`](https://r-xla.github.io/anvl/reference/prim_is_finite.md)
  : Primitive Is Finite
- [`prim_le()`](https://r-xla.github.io/anvl/reference/prim_le.md) :
  Primitive Less Than or Equal
- [`prim_lgamma()`](https://r-xla.github.io/anvl/reference/prim_lgamma.md)
  : Primitive Log-Gamma
- [`prim_log()`](https://r-xla.github.io/anvl/reference/prim_log.md) :
  Primitive Logarithm
- [`prim_log1p()`](https://r-xla.github.io/anvl/reference/prim_log1p.md)
  : Primitive Log Plus One
- [`prim_lt()`](https://r-xla.github.io/anvl/reference/prim_lt.md) :
  Primitive Less Than
- [`prim_lu()`](https://r-xla.github.io/anvl/reference/prim_lu.md) :
  Primitive LU Decomposition
- [`prim_max()`](https://r-xla.github.io/anvl/reference/prim_max.md) :
  Primitive Max Reduction
- [`prim_min()`](https://r-xla.github.io/anvl/reference/prim_min.md) :
  Primitive Min Reduction
- [`prim_mul()`](https://r-xla.github.io/anvl/reference/prim_mul.md) :
  Primitive Multiplication
- [`prim_ne()`](https://r-xla.github.io/anvl/reference/prim_ne.md) :
  Primitive Not Equal
- [`prim_negate()`](https://r-xla.github.io/anvl/reference/prim_negate.md)
  : Primitive Negation
- [`prim_not()`](https://r-xla.github.io/anvl/reference/prim_not.md) :
  Primitive Bitwise Not
- [`prim_or()`](https://r-xla.github.io/anvl/reference/prim_or.md) :
  Primitive Bitwise Or
- [`prim_pad()`](https://r-xla.github.io/anvl/reference/prim_pad.md) :
  Primitive Pad
- [`prim_plogis()`](https://r-xla.github.io/anvl/reference/prim_plogis.md)
  : Primitive Logistic (Sigmoid)
- [`prim_pmax()`](https://r-xla.github.io/anvl/reference/prim_pmax.md) :
  Primitive Parallel Maximum
- [`prim_pmin()`](https://r-xla.github.io/anvl/reference/prim_pmin.md) :
  Primitive Parallel Minimum
- [`prim_popcnt()`](https://r-xla.github.io/anvl/reference/prim_popcnt.md)
  : Primitive Population Count
- [`prim_pow()`](https://r-xla.github.io/anvl/reference/prim_pow.md) :
  Primitive Power
- [`prim_print()`](https://r-xla.github.io/anvl/reference/prim_print.md)
  : Primitive Print
- [`prim_prod()`](https://r-xla.github.io/anvl/reference/prim_prod.md) :
  Primitive Product Reduction
- [`prim_psigamma()`](https://r-xla.github.io/anvl/reference/prim_psigamma.md)
  : Primitive Psigamma
- [`prim_qr()`](https://r-xla.github.io/anvl/reference/prim_qr.md) :
  Primitive QR Decomposition
- [`prim_reduce()`](https://r-xla.github.io/anvl/reference/prim_reduce.md)
  : Primitive Generic Reduce
- [`prim_remainder()`](https://r-xla.github.io/anvl/reference/prim_remainder.md)
  : Primitive Remainder
- [`prim_reshape()`](https://r-xla.github.io/anvl/reference/prim_reshape.md)
  : Primitive Reshape
- [`prim_rev()`](https://r-xla.github.io/anvl/reference/prim_rev.md) :
  Primitive Reverse
- [`prim_rng_bit_generator()`](https://r-xla.github.io/anvl/reference/prim_rng_bit_generator.md)
  : Primitive RNG Bit Generator
- [`prim_round()`](https://r-xla.github.io/anvl/reference/prim_round.md)
  : Primitive Round
- [`prim_rsqrt()`](https://r-xla.github.io/anvl/reference/prim_rsqrt.md)
  : Primitive Reciprocal Square Root
- [`prim_scan()`](https://r-xla.github.io/anvl/reference/prim_scan.md) :
  Primitive Scan
- [`prim_scatter()`](https://r-xla.github.io/anvl/reference/prim_scatter.md)
  : Primitive Scatter
- [`prim_shift_left()`](https://r-xla.github.io/anvl/reference/prim_shift_left.md)
  : Primitive Shift Left
- [`prim_shift_right_arithmetic()`](https://r-xla.github.io/anvl/reference/prim_shift_right_arithmetic.md)
  : Primitive Arithmetic Shift Right
- [`prim_shift_right_logical()`](https://r-xla.github.io/anvl/reference/prim_shift_right_logical.md)
  : Primitive Logical Shift Right
- [`prim_sign()`](https://r-xla.github.io/anvl/reference/prim_sign.md) :
  Primitive Sign
- [`prim_sin()`](https://r-xla.github.io/anvl/reference/prim_sin.md) :
  Primitive Sine
- [`prim_sinh()`](https://r-xla.github.io/anvl/reference/prim_sinh.md) :
  Primitive Hyperbolic Sine
- [`prim_sort()`](https://r-xla.github.io/anvl/reference/prim_sort.md) :
  Primitive Sort
- [`prim_sqrt()`](https://r-xla.github.io/anvl/reference/prim_sqrt.md) :
  Primitive Square Root
- [`prim_static_slice()`](https://r-xla.github.io/anvl/reference/prim_static_slice.md)
  : Primitive Static Slice
- [`prim_sub()`](https://r-xla.github.io/anvl/reference/prim_sub.md) :
  Primitive Subtraction
- [`prim_sum()`](https://r-xla.github.io/anvl/reference/prim_sum.md) :
  Primitive Sum Reduction
- [`prim_svd()`](https://r-xla.github.io/anvl/reference/prim_svd.md) :
  Primitive Singular Value Decomposition
- [`prim_tan()`](https://r-xla.github.io/anvl/reference/prim_tan.md) :
  Primitive Tangent
- [`prim_tanh()`](https://r-xla.github.io/anvl/reference/prim_tanh.md) :
  Primitive Hyperbolic Tangent
- [`prim_top_k()`](https://r-xla.github.io/anvl/reference/prim_top_k.md)
  : Primitive Top-K
- [`prim_transpose()`](https://r-xla.github.io/anvl/reference/prim_transpose.md)
  : Primitive Transpose
- [`prim_triangular_solve()`](https://r-xla.github.io/anvl/reference/prim_triangular_solve.md)
  : Primitive Triangular Solve
- [`prim_which_max()`](https://r-xla.github.io/anvl/reference/prim_which_max.md)
  : Primitive Index of the Maximum
- [`prim_which_min()`](https://r-xla.github.io/anvl/reference/prim_which_min.md)
  : Primitive Index of the Minimum
- [`prim_while()`](https://r-xla.github.io/anvl/reference/prim_while.md)
  : Primitive While Loop
- [`prim_xor()`](https://r-xla.github.io/anvl/reference/prim_xor.md) :
  Primitive Bitwise Xor

## Package

- [`anvl`](https://r-xla.github.io/anvl/reference/anvl-package.md)
  [`anvl-package`](https://r-xla.github.io/anvl/reference/anvl-package.md)
  : anvl: Accelerated Array Computing and Automatic Differentiation
- [`install_anvl()`](https://r-xla.github.io/anvl/reference/install_anvl.md)
  : Install What a Backend Needs to Run
