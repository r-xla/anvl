# Review: data type / shape documentation pass

Written by a review agent against the state of the branch before the last round
of fixes, so a few items (the "floating-point" wording, the unnamed list
returns, `nv_array`'s spelled-out defaults) were already being addressed while
it ran. Every runtime claim was verified by executing in the dev container;
every spec anchor against `../stablehlo/SPEC.md`.

**Status:** everything below that was wrong, dead or missing has since been
fixed, and two later rounds of review (plus the merge of main's configurable
default data types) have moved on from it -- where this report says a page
states `i32`, the page now says "the default integer data type"; `LIST.md`'s "Review" section lists what changed and the two
observations that were deliberately left alone. The report is kept as the record
of what was checked.

## Summary

The pass is impressively consistent where it is templated. All 174 template call
sites draw their data-type phrase from the seven-word vocabulary and nothing
else; every `\value` on every `prim_*` page and all but one `nv_*` page opens
with a parenthesized type; the primitive layer never names a concrete default
data type *as a default*; no `try()` survives in the primitive examples; and I
found no wrong data-type *claim* among the ~50 primitives and wrappers I
executed in the dev container (*signed numeric* really rejects `ui32`,
*integerish* rejects `f32`, *numeric* rejects `bool`, the `i32` index outputs
and boolean-counted reductions are as documented, and a safetensors round-trip
preserves all eleven data types). What remains falls into three groups:
statements that are simply false (an example comment contradicting its own
parameter, a parameter describing the opposite function, three pages whose
return names a nonexistent argument, sixteen StableHLO links pointing at anchors
the spec does not have, seven references to a nonexistent `nv_shape()`); pages
the sweep never reached, now sitting beside the pages it did
(`prim_convolution`, thirteen `nv_*` pages still on bare `param_x`, `nv_matmul`,
`nv_det`/`nv_determinant`, `nv_fill`'s `dtype`); and small mechanical damage in
rewritten blocks (`nv_clamp`'s doubled sentence, `as_array`'s orphaned "Of
length 1.", `prim_while`'s dangling clause). Two stated conventions hold only in
a minority of pages: one comment per example stanza (98 of 275 pages have any
comment) and "the other parameters point at the primary operand" (1 of 5
`roxy_agree()` pages).

## Findings

**1. `prim_static_slice`'s example comments contradict its own parameters and
the code.** `R/primitives.R` examples → `man/prim_static_slice.Rd`:
`# 1-D: extract elements 2 through 4 (limit is exclusive)` and `# 2-D: extract a
submatrix (rows 1-2, columns 2-3)`. The limit is inclusive — the same page says
`End indices (inclusive), one per axis` and
`ceiling((limit_indices - start_indices + 1) / strides)`; running them returns
`2 3 4 5` and a **3x3** block (rows 1-3, cols 2-4). *Fix:* drop "(limit is
exclusive)", say "elements 2 through 5" / "rows 1-3, columns 2-4".

**2. `prim_argmin`'s `axis` describes `prim_argmax`.** `man/prim_argmin.Rd`
(from `@inheritParams prim_argmax`, `R/primitives.R:1215`): `Axis along which to
find the index of the maximum.` while the description says "minimum". *Fix:* own
`@param axis`.

**3. `nv_conv1d`/`2d`/`3d` returns name a nonexistent argument.** `R/api.R:3765`
+2, added by this pass: ``Has the data type `x` and `kernel` agreed on``. The
argument is `weight`. *Fix:* `x` and `weight`.

**4. Sixteen StableHLO links point at anchors the spec lacks.** `roxy_spec()`
renders "specified under [op](…/spec#op)". Absent from `../stablehlo/SPEC.md`
(128 `### op` sections): `acos acosh asin asinh atan atanh cosh sinh digamma
lgamma polygamma erf erfc erf_inv top_k` (all CHLO — `../stablehlo/R/op-acos.R:4`,
`dialect = "chlo"`) plus `prim_fill`'s `` `r roxy_spec("tensor")` `` → "Lowers to
`hlo_tensor()`, specified under [tensor](…/spec#tensor)"; `hlo_tensor()` is
stablehlo's constant builder (documented on the `hlo_constant` page, so that link
dangles too) and the spec op is `constant`. All 91 `hlo_*` names are correct —
only the anchors are wrong. *Fix:* `spec#constant` for fill; a CHLO helper using
`https://openxla.org/stablehlo/generated/chlo#chlo<op>_chlo<op>op` (cf.
`../stablehlo/man-roxygen/op_chlo.R`).

**5. Seven references to `nv_shape()`, which does not exist**
(`exists("nv_shape")` is `FALSE`): new this pass at `R/primitives.R:354`
(``shape `nv_shape(x)[permutation]` ``) and `:632`
(``nv_shape(update) <= nv_shape(x)``); pre-existing at
`R/primitives.R:497,568,572,3364`, `R/api-generics.R:176`. *Fix:* `shape(x)` —
the new text copied the old mistake.

**6. `nv_polygamma` claims integers are accepted; they are not.**
`R/api.R:1442-1450`: "so an integer or boolean input is converted rather than
refused." Verified: `nv_polygamma(nv_array(c(1L,1L)), nv_array(c(2L,3L)))` errors
``  `lhs` must have dtype float. x Got i32``; conversion happens only when the
operand meets a float. `prim_polygamma.Rd` says "Can be any float data type", so
the wrapper contradicts primitive and code. *Fix:* "…which must be a float — an
integer or boolean input is converted only where it meets a float operand."

**7. `nv_trace`'s return is wrong for boolean.** `R/api.R:3043`: ``A scalar with
the same data type as `x`.`` Verified `dtype(nv_trace(<bool 2x2>))` is `i32` (it
sums via `nv_reduce_sum()`). *Fix:* reuse "…except a boolean input, which is
counted at `i32`".

**8. `prim_dynamic_update_slice`'s inherited section names an argument it
lacks.** Via `@inheritSection prim_dynamic_slice Out Of Bounds Behavior`
(`R/primitives.R:636`): `clamp(1, start_indices, nv_shape(x) - slice_sizes + 1)`
— there is no `slice_sizes`; the bound is `shape(update)` (verified: start
clamped to 4, result `1 2 3 10 20`), and "before the slice is extracted" is
backwards for a write.

**9. `nv_array`'s `dtype` names the concrete defaults and disagrees with
`?dtypes`.** `R/array.R:43-48` (rewritten here): "`f32` for a double on
`"pjrt"`, `f64` for a double on `"quickr"`, `i32` for an integer and `bool` for a
logical" — the explicit prohibition, on the most-visited page; and it is the only
page making the default backend-dependent, where `man/dtypes.Rd` says flatly "a
`double` becomes `f32`…". Both are true (`R/backend-quickr.R:166`), so the pages
disagree. *Fix:* link `[default data type][default_dtypes]` here; put the backend
caveat on `?dtypes`.

**10. `prim_convolution` was missed entirely**: seven "1-based" mentions plus
"converted to StableHLO's 0-based layout internally",
`\value{(\code{\link{arrayish}})}` with no `\cr` and no content, no rules
section, no examples, and `kernel` with no data-type sentence.

**11. Thirteen `nv_*` pages still render "Input array."** (`@template param_x`):
`nv_pad`, `nv_reshape`, `nv_transpose`, `nv_reverse`, `nv_select`, `nv_squeeze`,
`nv_unsqueeze`, `nv_static_slice`, `nv_subset_assign`, `nv_tril`, `nv_triu`,
`nv_bitcast_convert`, `nv_broadcast_to` — several listed as covered in `LIST.md`.
Worst on `nv_pad` (return ``Has the same data type as `x`.`` also drops the shape
arithmetic `prim_pad` states). `.claude/skills/add-api-function/SKILL.md:141`
still recommends exactly this for `nv_clamp()`/`nv_pad()`.

**12. `nv_clamp`'s bound parameter has a doubled sentence.** `R/api.R:1534`:
"They take `x`'s data type -- see `x` -- and are brought to `x`'s data type: …" —
`prim_clamp` has the clean short form ("shares its data type -- see `x`."),
`nv_pad` the clean long form. *Fix:* start at "Brought to `x`'s data type: …".

**13. `as_array`'s return is a fragment.** `R/reexports.R:55`:
`(array | vector)\cr` / `Of length 1.` — "of length 1" belonged to the `vector`
branch only, so the page now claims every result has length 1.

**14. "floating-point" instead of *float***: `R/api.R:2023` and `:2053`
(`nv_det`, `nv_determinant`: "Square matrix of floating-point data type." —
siblings use "Can be any float data type."), `R/api.R:1659` (`nv_linspace`: "so
the result is floating-point"), and both `nan_rm` templates ("`NaN` values in
floating-point inputs", 14 pages).

**15. "1-based" on eight rendered pages, two newly written.** New this pass at
`R/primitives.R:3152` → `prim_gather`/`prim_scatter`: "The dimension numbers are
intricate; anvl states them 1-based and converts on the way down." (also
"dimension" for an anvl concept). Pre-existing: `prim_convolution` (7x),
`prim_lu`/`nv_lu`, `nv_top_k`, and `nv_cummax`/`nv_cummin` via finding 16.

**16. Framework name-drop in a shared template.**
`man-roxygen/param_nv_cum_with_indices.R:4-5`: "is the 1-based index of the last
occurrence … (dtype `i32`, matching torch)" — three violations in one clause; the
primitive template states the same `i32` without any. Pre-existing:
"Torch-style 1D/2D/3D convolution", "NumPy-style broadcasting rules", "(as it
would be for NumPy)".

**17. `prim_and`/`or`/`xor`/`not` and wrappers are described as logical but are
bitwise.** Eight pages say "Element-wise logical AND/…" while params now say
"integerish"; verified `nv_and(12L,10L)`=8, `nv_or`=14, `nv_xor`=6,
`nv_not(12L)`=-13.

**18. Text left ungrammatical by rewriting.** `prim_while` description ends
"Otherwise, no state is maintained between iterations." (dangling,
`R/primitives.R:2690`); `prim_bitcast_convert`'s `dtype`: "Of the same bit width
as the input's leaves the shape unchanged" (no subject); "Has `i32` data type
**whatever the input's is**" on `prim_argmax`/`prim_argmin`/`nv_argmax`/
`nv_argmin`/`nv_argsort` and "Has a float data type whatever the input's is" on
`nv_median`/`nv_quantile` (7 pages → "regardless of the input's");
`man/nv_flatten.Rd` `\title{Flatte}` (`R/api.R:267`), the only `nv_*` page with
unguarded `@examples`, description "N-dimensional … 1-dimensional";
`man/nv_bitcast_convert.Rd`'s example calls `prim_bitcast_convert()`; typos
"underluing stableHLO" and "from a array" (`prim_gather`), "divident"
(`prim_remainder`), "the axis and it's size" (`AnvlArray`), "if was conveted"
(`peek_dtype`, pre-existing but the page the new default text points at).

**19. `promotion_rule`'s central sentence is backwards.** `R/promotion.R:121`:
"as long it is within their category (a `double` can e.g. *not* become a float)"
— a double's category *is* float; it cannot become an *integer*; "as long it is"
also drops an "as".

**20. `nv_matmul`'s return says nothing.** `\value{(\code{\link{arrayish}})}` —
the only such `nv_*` page; its `lhs,rhs` names no accepted data types. Same gap
on `nv_chol`, `nv_conv1d/2d/3d`, `nv_crossprod`/`nv_tcrossprod`, `nv_outer`, and
on `man/AnvlArray.Rd`, `man/arrayish.Rd`, `man/common_dtype.Rd` (bare type, no
text) — while `nv_concatenate`, in the same file, does it properly.

**21. `nv_fill`'s `dtype` is undocumented and the constructors describe defaults
six ways.** `man/nv_fill.Rd`: `Data type.` only — no accepted types and nothing
about `NULL` (verified it follows the value: `f32`/`i32`/`bool`). Compare
`prim_fill` ("Can be any data type."), `nv_iota`, `nv_eye` ("defaults to
`"f32"`", and its type slot omits the `NULL` its text describes), `nv_seq` ("the
default (`NULL`) is `i32`"), `nv_linspace`, `nv_array` (finding 9).

**22. "Can be any numeric data type, boolean being the one exception" (4
sites)** — `prim_iota`, `prim_rng_bit_generator`, `nv_iota`, `nv_seq`: *numeric*
already excludes boolean per `section_dtype_words.R`, so the clause implies the
opposite of the vocabulary.

**23. The `roxy_agree()` "others point at it" rule holds on 1 of 5 pages.** Only
`prim_clamp` does ("shares its data type -- see `x`."); `prim_pad`,
`prim_scatter`, `prim_dynamic_update_slice`, `prim_convolution` siblings say
nothing about data types.

**24. Three phrasings for "must be boolean", two in the type slot.** "Must be a
boolean or an R logical." (reduce_any/all, 4 pages) vs `prim_if` "Must be a
scalar of the boolean data type, or an R logical." vs `nv_if` `(arrayish of
boolean type, scalar)`; and `(arrayish of integer type)` on
`prim_scatter`/`prim_dynamic_update_slice` where
`prim_gather`/`prim_dynamic_slice` keep the plain slot and say it in prose.
"boolean type"/"integer type" are outside the vocabulary.

**25. `nv_solve` vs `nv_triangular_solve`** state the same result two ways: "with
`b`'s shape and the data type `a` and `b` agreed on" vs "with the same shape and
dtype as `b`" (prose "dtype" + the discouraged "same as the input" form). "dtype"
as prose also on `prim_round`, `prim_eigh`, `nv_determinant`, `nv_lu`.

**26. The example convention is met on a minority.** 98 of 275 pages with
examples have any comment (39 of 99 `prim_*`), including pages this pass rewrote
(`nv_det`, `nv_clamp`, `nv_pad`, `nv_diag`, `prim_top_k`, `prim_fill`…). Five
sentence-case leftovers and one two-line comment (`prim_sort`).

**27. `LIST.md` claims "To do — None" while 22 exported `nv_*` pages are absent
from the ledger** (`nv_matmul`, `nv_det`, `nv_determinant`, `nv_chol`,
`nv_conv1d/2d/3d`, `nv_reshape`, `nv_transpose`, `nv_reverse`, `nv_select`,
`nv_squeeze`, `nv_unsqueeze`, `nv_static_slice`, `nv_subset_assign`, `nv_trace`,
`nv_tril`, `nv_triu`, `nv_concatenate`, `nv_bitcast_convert`, `nv_broadcast_to`,
`nv_normal`). Findings 3, 7, 11, 14, 20 all live in that set. The primitive
ledger is exact: all 100 pages listed, nothing listed without a page.

**28. Leftovers.** `man-roxygen/param_prim_x_any.R`, `section_shapes_unary.R`,
`section_shapes_binary.R`, `section_shapes_reduce.R` are referenced by nothing in
`R/`. `.claude/skills/add-primitive/SKILL.md` still tells authors to use
`@template param_prim_x_any` for `prim_clamp()`/`prim_pad()` and to hand-write
"Lowers to [stablehlo::hlo_<name>()]" instead of `roxy_spec()`.

**29. Cosmetic:** `roxy_agree()` and `dtype_out` substitutions land after roxygen
wraps, so 13 `.Rd` files carry one 208–268-character unwrapped line
(`prim_clamp`, `prim_pad`, `prim_ifelse`, `prim_reduce`, `prim_scatter`,
`prim_concatenate`, `prim_polygamma`, `prim_convolution`, `prim_dynamic_slice`,
`prim_dynamic_update_slice`, `prim_triangular_solve`, `nv_solve`,
`nv_triangular_solve`; plus `nv_reduce_sum`/`nv_reduce_prod`).

**30. Small wording inconsistencies for one pass:** `return_reduce` renders "Has
**boolean** data type." vs "Has **a float** data type."; `param_while_init` is
the only default-data-type sentence not linking `[default_dtypes]`; "dimension"
for anvl concepts in `prim_reduce` ("0-dimensional"), `prim_sort`
("1-dimensional slices"), `nv_reshape` example comment,
`section_nv_cum_relation`, `nv_flatten`; constraints repeated in `@description`
on `prim_polygamma`, `prim_chol`, `prim_triangular_solve`; the last `try()` in an
example on `man/promotion_rule.Rd`; `nv_conv3d` inherits `nv_conv2d`'s 4-axis
`x`/`weight` shapes while its own description and new return describe five axes;
`nv_select`'s `index` "Scalar or 1D arrayish input (integer)."; `prim_reduce`'s
"floating point math".

## Checked and found clean

- **Vocabulary**: all 174 `@templateVar dtypes`/`dtype_out` values are from the
  seven words (`any` 66, `float` 70, `integerish` 14, `numeric` 13, `signed
  numeric` 4, `integer` 2, plus 4 "a boolean or an R logical"). No
  "real"/"double"/ad-hoc group word; the only "floating-point" hits are findings
  14/30.
- **"Can be *of* any data type" (11 pages) is not a defect** — used consistently
  for plural/`...` parameters.
- **Parenthesized returns**: all 100 `prim_*` and every `nv_*` page except
  `man/nv_normal.Rd` (maintainer-owned). Two lack the `\cr` (`promotion_rule`,
  `prim_convolution`).
- **Never naming the default** holds across the whole primitive layer; every
  `f32`/`i32`/`bool` there is a fixed choice. Exceptions are the `nv_*`
  constructors (findings 9, 21).
- **Data-type claims vs the running package**: verified for ~35 primitives and 15
  wrappers — every group and every stated output data type matched, including
  primitives keeping `bool` in `reduce_sum`/`cumsum` while the wrappers count at
  `i32`.
- **`roxy_agree()` operand lists** match `apply_promotion()` in all ten
  primitives and both `nv_*` users.
- **All 91 `hlo_*` names** named by `roxy_spec()` exist and are exported by
  stablehlo.
- **`\usage` vs `\item`**: no undocumented argument and no orphan `\item` across
  all 275 pages.
- **"tensor"** in anvl-facing text: none (only stablehlo's own names and
  `safetensors`).
- **Serialization**: "Any data type … round-trip unchanged" verified for all
  eleven data types.
- **`nv_static_slice`'s "(inclusive)"** is correct; only `prim_static_slice`'s
  comments are wrong.
- **`nv_qr`/`nv_svd`/`nv_eigh`** use `@inherit prim_*`, so they cannot drift.

*Out of scope, noted only:* the maintainer-owned RNG/distribution pages deviate
similarly (`nv_normal` is the one page whose `\value` lacks a parenthesized type
and a `nv_qnorm()` clause, names `"f32"` as a default, says "real array", cites
Cephes "as used by JAX"; `nv_runif`/`nv_rbinom` render a bare "Data type." and
state no result data type or shape).

## Digest

30 findings. The three most important:

1. Six outright false statements introduced or left in place —
   `prim_static_slice`'s "limit is exclusive" example comments (the limit is
   inclusive; the calls return four elements and a 3x3 block), `prim_argmin`'s
   `axis` describing the *maximum*, `nv_conv1d/2d/3d` returns naming a `kernel`
   argument that does not exist (it is `weight`), `nv_polygamma` claiming integer
   inputs are converted rather than refused (they error), `nv_trace` claiming the
   input's data type for a boolean input (it is `i32`), and
   `prim_dynamic_update_slice` inheriting a clamp formula with a `slice_sizes`
   argument it does not have.
2. Sixteen `roxy_spec()` StableHLO links point at spec anchors that do not exist
   — the fifteen CHLO ops (`acos`, `erf`, `top_k`, …) plus `prim_fill`'s
   `spec#tensor` — and seven doc references call a nonexistent `nv_shape()`.
3. `nv_array`'s `dtype` spells out the concrete defaults (`f32`/`f64`/`i32`/
   `bool`), the one thing the conventions forbid, and in a backend-dependent form
   that contradicts `?dtypes`.
