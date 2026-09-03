# Bring a Primitive's Operands to One Data Type

Applies a [promotion
rule](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md) to
the operands of a primitive, inside the primitive's own body.

A primitive promotes nothing on its own: an R value among its operands
would commit to its own default, so whether a call worked would depend
on whether the array it met happened to be at that default. A primitive
whose operands must *agree* says so with this, on the same list it goes
on to hand
[`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md):

    function(lhs, rhs) {
      operands <- apply_promotion(list(lhs = lhs, rhs = rhs), promote_rdata_common())
      graph_desc_add(self, operands, infer_fn = infer_fn)[[1L]]
    }

[`promote_rdata_common()`](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md)
is the rule for it: an operand that has a data type keeps it, and an R
value takes the one the others have, within its own category. That is
what makes `prim_mul(x_f64, 2)` work whatever `x`'s data type is, while
keeping a primitive from widening the array it was handed – promotion
across categories is the `nv_*` layer's job.

## Usage

``` r
apply_promotion(operands, promote)
```

## Arguments

- operands:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  The operands, named as the primitive's
  [`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md)
  call names them.

- promote:

  (`function`)  
  The rule to apply; see
  [promotion_rule](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md).

## Value

([`list()`](https://rdrr.io/r/base/list.html))  
`operands`, each realized at the data type the rule named for it.

## Details

Pass only the operands that must agree, and name them as the
[`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md)
call names them.
[`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md)
promotes its two branches and leaves `pred` a `bool`;
[`prim_scatter()`](https://r-xla.github.io/anvl/dev/reference/prim_scatter.md)
promotes `x` and `update` and leaves the indices alone. A primitive with
one arrayish operand, or with deliberately heterogeneous ones
([`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)'s
payload,
[`prim_while()`](https://r-xla.github.io/anvl/dev/reference/prim_while.md)'s
loop state), calls this not at all.

Call it before the body uses the operands for anything else, so it sees
settled data types throughout:
[`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md)
reads `dtype(init)` to trace its reductor and
[`prim_scatter()`](https://r-xla.github.io/anvl/dev/reference/prim_scatter.md)
builds its update computation's parameter slots from
[`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md),
both before recording a call.

It is idempotent: once every operand is at the data type the rule names,
realizing them again changes nothing.

## See also

[promotion_rule](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md),
[`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md),
[`vignette("extending_primitive")`](https://r-xla.github.io/anvl/dev/articles/extending_primitive.md)

## Examples

``` r
# An R value takes the data type of the operand it meets.
operands <- apply_promotion(list(lhs = nv_scalar(1, "f64"), rhs = 2), promote_rdata_common())
dtype(operands$rhs)
#> <f64>
```
