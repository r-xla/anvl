# Promotion Rules

Functions for materializing R values to arrays and promoting inputs.
Most commonly used via the `.promote` argument of
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md).

`promote_common()` brings every input to their common data type
([`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md)).
R values always yield within the type category (such as float) and
otherwise contribute their default data type.

`promote_like()` brings the inputs to the data type of a selected input.
If the selected data type is an R value, it's default data type is used.

`promote_dtype()` brings the inputs to the specified data type.

`promote_rdata_common()` brings the *R values* to the common data type,
as long it is within their category (a `double` can e.g. *not* become a
float). `AnvlArray` inputs are left as they are and the function throws
an error if not all of them have exactly the same data type. This rule
is commonly used in primitives expecting homogenous inputs for one or
more argument subsets.

`promote_grouped()` applies several rules to disjoint subsets.

`promotion_rule()` creates a new promotion rule. It takes in
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
values and outputs a list of data types, with `NULL` indicating no
conversion.

## Usage

``` r
promote_common(on = NULL, fallback = NULL)

promote_like(arg, on = NULL, coerce = FALSE)

promote_dtype(dtype, on = NULL, coerce = FALSE)

promote_rdata_common(on = NULL)

promote_grouped(...)

promotion_rule(fn, kind, on = NULL, ...)
```

## Arguments

- on:

  (`NULL` \| [`character()`](https://rdrr.io/r/base/character.html) \|
  [`numeric()`](https://rdrr.io/r/base/numeric.html))  
  Subset of arguments to apply a rule to. Indicated either via position
  or argument name.

- fallback:

  (`NULL` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html)
  \| `character(1)`)  
  The data type to settle on when *every* input is a bare R value, in
  place of the default those would commit to on their own. `NULL`
  (default) leaves them their default.

- arg:

  (`character(1)` \| `numeric(1)`)  
  Which input to take the data type from: its name in the
  [`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
  call, or its position. Naming it needs the call's arguments to be
  named.

- coerce:

  (`logical(1)`)  
  Bring an input to the target even where that is not a promotion,
  instead of raising an error. Two things are refused without it: a
  float reaching an integer data type, which no category crosses to on
  its own (an R double at `i32`, or an `f32` array at `i32`), and
  narrowing a value the target cannot hold (an `f64` array at `f32`).
  The default is `FALSE`.

- dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html)
  \| `character(1)`)  
  The data type to bring the inputs to.

- ...:

  (`PromotionRule`)  
  The rules to apply to disjoint argument subsets.

- fn:

  (`function`)  
  The rule.

- kind:

  (`character(1)`)  
  What the rule is, for printing: it shows as `<promote_{kind}>`.

## Value

`function(args) -> list()` A function returning data types for those
inputs to be converted and `NULL` for those to be left unchanged.

## See also

[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md),
[`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md),
[`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md)

## Examples

``` r
promote_common()(list(pi, nv_scalar(2L, "i64")))
#> [[1]]
#> <f32>
#> 
#> [[2]]
#> <f32>
#> 
promote_common(fallback = "f64")(list(1, 2))
#> [[1]]
#> <f64>
#> 
#> [[2]]
#> <f64>
#> 
promote_common(c(1, 2))(list(-3, 4, 1))
#> [[1]]
#> <f32>
#> 
#> [[2]]
#> <f32>
#> 
#> [[3]]
#> NULL
#> 
promote_like("x", coerce = TRUE)(list(x = nv_scalar(1, "f32"), nv_scalar(1, "f64")))
#> [[1]]
#> <f32>
#> 
#> [[2]]
#> <f32>
#> 
# Without `coerce`, a target the input cannot hold is refused.
try(promote_like("x")(list(x = nv_scalar(1, "f32"), nv_scalar(1, "f64"))))
#> Error : Cannot bring argument 2 to data type "f32".
#> ✖ "f64" is not promotable to "f32".
#> ℹ Convert it explicitly with `nv_convert()`.
# Every input at the widest float in the call, and never below f32.
widest_float <- promotion_rule(
  function(args) {
    widths <- vapply(args, function(a) {
      dt <- peek_dtype(to_abstract(a))
      if (tengen::is_dtype_float(dt)) tengen::dtype_width(dt) else 0L
    }, integer(1))
    rep(list(as_dtype(paste0("f", max(c(32L, widths))))), length(args))
  },
  "widest_float"
)
widest_float
#> <promote_widest_float> 
as_anvl_arrays(nv_array(1L), 2.5, nv_array(1, dtype = "f64"), .promote = widest_float)
#> [[1]]
#> AnvlArray
#>  1
#> [ CPUf64{1} ] 
#> 
#> [[2]]
#> AnvlArray
#>  2.5000
#> [ CPUf64{} ] 
#> 
#> [[3]]
#> AnvlArray
#>  1
#> [ CPUf64{1} ] 
#> 
```
