# Promotion Rules

Functions for materializing R values as arrays and promoting inputs.
Most commonly used via the `.promote` argument of
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md).

`promotion_common()` brings every input to their common data type
([`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md)).
An R value takes the data type the arrays meet at when that is in its
own or a higher category, and otherwise contributes its default data
type.

`promotion_like()` brings the inputs to the data type of a selected
input. If the selected input is an R value, its default data type is
used.

`promotion_dtype()` brings the inputs to the specified data type.

`promotion_rdata_common()` brings the *R values* to the data type of the
arrays among the inputs, which must all have the same one. An R value
must be in that data type's category (a `double` can e.g. *not* become
an integer). When all inputs are R values, they settle on their shared
default data type; R values of different storage types are an error.
This rule is commonly used in primitives expecting homogeneous inputs
for one or more argument subsets.

`promotion_grouped()` applies several rules to disjoint subsets. The
rules must all refer to arguments by name, or all by position.

`promotion_rule()` creates a new promotion rule. It takes in
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
values and outputs a list of data types, with `NULL` indicating no
conversion.

## Usage

``` r
promotion_common(on = NULL, fallback = NULL)

promotion_like(arg, on = NULL, coerce = FALSE)

promotion_dtype(dtype, on = NULL, coerce = FALSE)

promotion_rdata_common(on = NULL)

promotion_grouped(...)

promotion_rule(fn, kind, on = NULL, ...)
```

## Arguments

- on:

  (`NULL` \| [`character()`](https://rdrr.io/r/base/character.html) \|
  [`numeric()`](https://rdrr.io/r/base/numeric.html))  
  Subset of arguments to apply a rule to. Indicated either via position
  or argument name. For `promotion_rule()`, `on` only declares which
  arguments the rule covers (which `promotion_grouped()` needs to check
  that its rules are disjoint); `fn` itself must restrict itself to
  them.

- fallback:

  (`NULL` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html)
  \| `character(1)`)  
  The data type to settle on when *every* input is a bare R value, in
  place of the default those would materialize at on their own. `NULL`
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

  For `promotion_grouped()`: (`PromotionRule`)  
  The rules to apply to disjoint argument subsets.

  For `promotion_rule()`: (any)  
  Further fields stored in the rule's `spec` attribute next to `on`,
  e.g. for [`format()`](https://rdrr.io/r/base/format.html) to show.

- fn:

  (`function`)  
  The rule.

- kind:

  (`character(1)`)  
  What the rule is, for printing: it shows as `<{kind}>`, so give it the
  name of the function that builds it.

## Value

(`PromotionRule`)

## See also

[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md),
[`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md),
[`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md)

## Examples

``` r
promotion_common()(list(pi, nv_scalar(2L, "i64")))
#> [[1]]
#> <f32>
#> 
#> [[2]]
#> <f32>
#> 
promotion_common(fallback = "f64")(list(1, 2))
#> [[1]]
#> <f64>
#> 
#> [[2]]
#> <f64>
#> 
promotion_common(c(1, 2))(list(-3, 4, 1))
#> [[1]]
#> <f32>
#> 
#> [[2]]
#> <f32>
#> 
#> [[3]]
#> NULL
#> 
promotion_like("x", coerce = TRUE)(list(x = nv_scalar(1, "f32"), nv_scalar(1, "f64")))
#> [[1]]
#> <f32>
#> 
#> [[2]]
#> <f32>
#> 
# without `coerce`, a target the input cannot hold is refused
try(promotion_like("x")(list(x = nv_scalar(1, "f32"), nv_scalar(1, "f64"))))
#> Error : Cannot bring `..2` to data type "f32".
#> ✖ "f64" is not promotable to "f32".
#> ℹ Convert it explicitly with `nv_convert()`.
promotion_dtype("f64")(list(1, nv_scalar(2, "f32")))
#> [[1]]
#> <f64>
#> 
#> [[2]]
#> <f64>
#> 
promotion_rdata_common()(list(nv_scalar(1, "f64"), 2))
#> [[1]]
#> <f64>
#> 
#> [[2]]
#> <f64>
#> 
try(promotion_rdata_common()(list(nv_scalar(1L, "i32"), 2.5)))
#> Error : `..2` is an R double, which cannot be used at the "i32" data type here.
#> ℹ A literal is only ever built at a data type of its own category: a double
#>   becomes a float, an integer an integer, a logical a "bool".
#> ℹ Use an operation that promotes across categories, or convert explicitly with
#>   `nv_convert()`.
rule <- promotion_grouped(
  promotion_dtype("f64", on = "x"),
  promotion_like("x", on = "y")
)
rule
#> <promotion_grouped(<promotion_dtype(f64) on "x">, <promotion_like("x") on "y">) on "x", "y"> 
rule(list(x = 1, y = nv_scalar(2L, "i32"), z = 3L))
#> [[1]]
#> <f64>
#> 
#> [[2]]
#> <f32>
#> 
#> [[3]]
#> NULL
#> 
# every input at the widest float in the call, and never below f32.
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
#> <widest_float> 
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
