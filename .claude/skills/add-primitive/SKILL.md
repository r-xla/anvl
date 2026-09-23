---
name: add-primitive
description: Add a new primitive operation to anvl (prim_* function with stablehlo, reverse rules, and tests)
user_invocable: true
---

# Add a New Primitive to anvl

Read `vignettes/extending_primitive.Rmd` first — it is the primary guide with a complete walkthrough (primitive creation via `new_primitive()`, stablehlo rule, reverse rule, nv\_\* API, file organization). This skill covers additional details not in the vignette.

## Before Starting: Check StableHLO Support

1. Check `../stablehlo/R/` for an `op-<name>.R` file (e.g. `op-add.R`).
2. Check that the operation can be expressed in StableHLO
   I.e., either it exists as `stablehlo::hlo_<name>()` or it can be expressed as a combination of existing StableHLO operations.
3. If the op doesn't exist in stablehlo, stop and tell the user — it must be added there first.
4. Read the StableHLO SPEC (`../stablehlo/SPEC.md`) for the operation's semantics and constraints.

## Argument Naming

The primary array argument of a `prim_*` (and its `nv_*` wrapper) is always named **`x`** — never `operand`, `input`, `a`, or anything else. This holds even when StableHLO's own spec calls it `operand` or `input`.

- Multiple arrays in the same role: `xs` (a list, as in `prim_sort(xs, ...)`).
- Two symmetric operands of a binary op: `lhs` / `rhs`. The one exception is `prim_atan2(y, x)`, which follows `base::atan2()`.
- Arguments naming a genuinely different role keep a descriptive name: `start_indices`, `update`, `weight`, `init`, `reducer`, `padding_value`, ...
- Axis arguments derived from `x` follow it, and are spelled _axes_, never _dims_: `x_batching_axes`, `scatter_axes_to_x_axes`, `offset_axes`, `index_vector_axis`.

When a StableHLO builder or `*DimensionNumbers()` constructor takes the spec name, map anvl's name back at the call site rather than renaming the anvl argument, e.g.

```r
stablehlo::GatherDimensionNumbers(
  operand_batching_dims = x_batching_axes - 1L,  # spec name on the left
  ...
)
```

## Integer Literals

Axis numbers, shape entries, indices and counts are integers, so write them as `1L`, not `1` --
including in the arithmetic that converts to StableHLO's 0-based indexing (`axes - 1L`,
`rep(1L, rank)`, `seq_len(rank) - 1L`).

A literal that meets an array keeps its `L` as well -- `prim_fill(1L, dtype = dtype(x), ...)`,
`hlo_scalar(0L, dtype = dtype(x), ...)` -- whatever category that array is in. An R integer widens
into any category, while a plain `1` is an R *double* that would pull an integer array into the
float category.

Drop the `L` only where the value is genuinely a real number that happens to be whole, such as a
coefficient built in R: `prim_fill(2 / sqrt(pi), dtype = dtype(x), ...)`.

## Roxygen Documentation

Use templates from `man-roxygen/` where applicable. You can check the folder for available ones.

- **Rules section:** `@templateVar primitive_id <name>` + `@template section_rules`
- Do NOT mention "1-based indexing" — it's the R default.
- Add `@export` to the roxygen block.

## Shortcuts for Simple Ops

The vignette shows the manual `graph_desc_add()` approach. For simple ops without extra parameters, pass a body produced by `make_unary_op()` / `make_binary_op()` to `new_primitive()`. Both helpers take an inference rule from `R/rules-inference.R` and rely on the lexically-bound `self` installed by `new_primitive()`. An elementwise op usually reuses a shared rule rather than getting its own:

```r
# Simple unary (e.g. prim_abs, prim_negate):
prim_<name> <- new_primitive("<name>", make_unary_op(infer_numeric_uni))

# Simple binary (e.g. prim_add, prim_mul):
prim_<name> <- new_primitive("<name>", make_binary_op(infer_generic_biv))
```

The shared rules are `infer_{generic,numeric,float,integer,integerish}_{uni,biv}()`; the prefix names the dtype group the op accepts (see `?dtypes`).

## Inference Rule

Every primitive has an inference rule `infer_<name>()` in `R/rules-inference.R`, passed to `graph_desc_add(..., infer_fn = infer_<name>)`. It maps the incoming `AbstractArray`s plus the static params to the outgoing `AbstractArray`s, and it is **the** place where the primitive's arguments are checked. Read the header comment of `R/rules-inference.R` before writing one; the rules below summarize it.

### Shape of the function

- **Formals are the primitive's own**: every operand and _every_ static param, under the names the `prim_*()` takes, including params the rule never reads (`precision`, `descending`). `graph_desc_add()` calls it as `do.call(infer_fn, c(avals_in, params))`.
- **Return** a plain `list()` of `AbstractArray(dtype = ..., shape = Shape(...))`, named when the primitive has named outputs (`list(values = ..., indices = ...)`).
- **Implement the StableHLO spec constraints**, in order, tagging each check with its constraint number from `../stablehlo/SPEC.md` as a comment (`# (C2)`). Translate to anvl's vocabulary: arrays not tensors, axes not dimensions, `x` not `operand`, and 1-based axis numbers.
- An output that met nothing takes `default_int()` / `default_float()`, never a hardcoded `"i32"` / `"f32"`.

### Where a check goes

- Anything decidable from the avals and params goes **in the rule, not in the `prim_*()` body**. A check in both places gives one mistake two wordings. (Either place reports `prim_<name>()` as the call: a primitive names itself on the way in, and `trace_fn()` rewrites the call of anything raised under it.)
- The wrapper keeps only what a rule cannot do: normalizing (`resolve_axis()` / `resolve_axes()` turn a negative axis into a concrete one, so the rule only sees the result), coercing a param before it is stored (then use the same `assert_*_param()` helper the rule uses, so the wording stays the same), checking sub-graph functions (traced before `graph_desc_add()`), and a guard the wrapper's own next line depends on.
- Validate every whole-number param with `assert_int_param()` / `assert_size_param()` **before** indexing or comparing with it. Otherwise an `NA` reaches an `if ()` and the caller sees R's raw `missing value where TRUE/FALSE needed`.
- That check is per entry, so it does not survive arithmetic: params each inside the integer range still overflow when a rule adds or multiplies them. Compute a result shape in double (`as.double()`) and hand it to `assert_result_shape()`, as `infer_convolution()` does.
- Guard against inputs that would crash *later*, deeper in the stack, with a worse message: a stride of 0, an axis past the end of a lower-rank operand, a rank-0 input to an axis-taking op.

### Error messages

Every refusal must say **which argument** it is about and **what that argument was given**. A primitive takes up to a dozen params, so a message that names none of them leaves the caller guessing.

Use the `cli_abort()` form with a headline stating the requirement and an `x` bullet reporting the actual value:

```r
cli_abort(c(
  "{.arg strides} must be positive.",
  x = "Got {value_repr(stride)}."
))
```

- **Headline**: `{.arg <name>} must ...`: the requirement, phrased positively, naming the argument as the primitive spells it. Mention relevant context in parentheses (e.g. `one entry per axis of {.arg x} ({rank})`).
- **`x` bullet**: `Got <value>.`. Print the value with the helper that fits:
  - `value_repr()` for a caller's value of any type. It spells the value the way the caller would type it (`3`, `"afz"`, `c(1, 3)`, `integer(0)`, `NULL`), using `format_param()` from the graph printer for the entries, so show the vector itself, not its length: `Got c(3, 4).`, not `Got 2 (c(3, 4)).` Unlike `format_param()`, which only prints params that inference already accepted, it copes with anything: a matrix or array reads as the call that builds it (`matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)`), a list prints as `<list> of length n`, and anything else with a class as `<class>` (`<factor>`, `<function>`). Interpolate it plain, not inside `{.val}`, which would quote the string it returns.
  - `params_repr(list(a = ..., b = ...))` when the check involves several params at once (`` `start_indices` = 1, `strides` = c(1, 2) ``).
  - `shape_repr()` for shapes (`(2x3)`), `repr()` for a whole array type, `{.val {as.character(dtype(x))}}` for a data type.
- **Never print a caller's value with a bare `{x}` / `{.val {x}}`** unless its length is already checked to be 1. A caller can pass anything (`prim_fill(1:1000, ...)`, a 5000-character string, a list), and the helpers above are what keep the message short and quick to build: they show at most 8 entries (`c(1, 2, 3, 4, 5, 6, 7, 8, ...) of length 1000`) and cut strings at 30 characters; `shape_repr()` stops after 8 axes. Don't paste a vector into a message yourself (`paste0(x, collapse = ", ")`).
- **Refuse input that would fail later with a worse message.** Examples: a shape whose element count overflows int64 (`assert_shapevec()` checks this), and a whole number outside the integer range, which `as.integer()` would silently turn into `NA` (`assert_int_param()` checks this). Some of these abort the process rather than raising: XLA `CHECK`-fails on a negative window bound, so `infer_convolution()` refuses a negative padded input itself.
- **Point at the offending entries** when only some are wrong: `Got {value_repr(start[bad])} at {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)}.`
- **An `i` bullet** adds context that isn't the offending value: the full set of params on a clash (`i = "Got {params_repr(parts)}."`), or where to look (`i = "See {.fn tengen::as_dtype} ..."`).
- **Operands without their own formal** are named by the argument the caller used: `..2` for operands passed through `...` (`prim_concatenate()`), `xs[[2]]` for ones collected in a list (`prim_sort()`). `assert_arrays(..., .arg = "xs")` does this for you. When operands disagree, name the first one *and* the one that disagrees: `` `..1` has shape (2x3), `..2` has shape (2x4). ``
- Pluralize with `cli::qty()` (`{cli::qty(n)}ax{?is/es}`, `entr{?y/ies}`). Don't write "axis(es)".
- Don't describe a range the caller could satisfy but that doesn't exist: at rank 0 say there is no axis to select, not "between 1 and 0" (see `assert_axes_in_range()`).

### Reuse the checking helpers

Reach for the shared helpers at the top of `R/rules-inference.R` before writing a message by hand. They already follow the rules above:

| Helper | Checks |
| --- | --- |
| `assert_array()` / `assert_arrays()` | operand is an `AbstractArray` |
| `assert_array_dtype(x, "float", "int", shape =, naxes =)` | dtype category, shape, rank |
| `assert_same_type()` / `assert_same_dtype()` | two operands agree |
| `assert_int_param(x, arg, len =, min_len =)` | whole-number param, no `NA`, length |
| `assert_size_param()` | as above, plus non-negative |
| `assert_flag_param()` | `TRUE` / `FALSE` |
| `assert_choice_param(x, arg, choices)` | one of a fixed set of strings |
| `assert_dtype_param()` | names a data type |
| `assert_axes_in_range()` / `assert_axes_unique()` / `assert_axes_sorted()` | axis vectors |
| `assert_axis_layout()` | several params that together name every axis once |
| `assert_result_shape(shape, what, parts)` | a shape the rule *computed*: refuses an axis past the integer range |

If a new check will be needed by more than one rule, add a helper next to these, in the same style.

### Testing the rule

Snapshot every refusal through the **primitive itself** in `tests/testthat/test-rules-inference.R` (`test_that("prim_<name>", { expect_snapshot(error = TRUE, prim_<name>(...)) })`), so the snapshot holds the message a caller actually sees, including the rewritten call. Cover each distinct `cli_abort()` of the rule, and read the generated `_snaps/rules-inference.md` entries: every one should name the argument and show the value.

## Reverse Rule: Additional Guidance

Beyond what the vignette covers:

- Build gradient expressions using `prim_*` primitives — never use R arithmetic directly.
- For non-differentiable points (e.g. `abs` at 0, `floor` everywhere), follow PyTorch conventions (subgradients, zero gradients, etc.). Read existing rules in `R/rules-reverse.R` for examples.

## Optional: Quickr Rule

If the primitive should also run under `local_backend("quickr")`, add a `quickr` lowering in `R/rules-quickr.R` via `quickr_register_prim_lowerer(prim_<name>, function(...) { ... })`. This emits plain R code for the quickr backend. If you skip it, the primitive still works on the pjrt backend; only the quickr one refuses it.

## API Wrapper (`nv_*`)

Follow the `/add-api-function` skill for this step — it covers design principles (R naming, semantics, generics), implementation, documentation, `_pkgdown.yml` placement, and testing.

`prim_*` primitives are auto-included under the "Primitives" section in `_pkgdown.yml` via `starts_with("prim_")`.

## Testing

The vignette covers file organization and the torch-vs-manual decision. This section adds concrete patterns.

### Decision: Torch comparison vs. manual R tests

- **Use torch comparison** (`inst/extra-tests/`) when the operation has a torch equivalent and the reverse rule is non-trivial.
- **Use manual R tests** (`tests/testthat/test-primitives-stablehlo.R` and `tests/testthat/test-primitives-reverse.R`) when no torch equivalent exists or the expected output can be stated analytically.

Choose one approach, not both.

### Test structure: property-based with edge cases

Use `describe()` / `it()` blocks. Cover:

- Different shapes (scalar, vector, matrix, 3D)
- Boundary values (depends on the specific operation)
- dtype variations where relevant
- Parameter variations (e.g. different `axes`, `permutation` values)
- Non-differentiable points: include those values in the test inputs and verify anvl's gradient matches torch's gradient at those points.

### Forward test example (torch comparison in `inst/extra-tests/test-primitives-stablehlo-torch.R`)

```r
describe("prim_foo", {
  gen_foo <- function(shp, dtype) {
    n <- if (!length(shp)) 1L else prod(shp)
    vals <- c(0, -1, 1, 0.5, -0.5, 100, -100, sample(rnorm(100), n - 7L))
    vals <- vals[seq_len(n)]
    if (!length(shp)) vals else array(vals, shp)
  }

  it("works for scalars", {
    expect_jit_torch_unary(prim_foo, torch::torch_foo, integer(), gen = gen_foo)
  })

  it("works for vectors", {
    expect_jit_torch_unary(prim_foo, torch::torch_foo, 10L, gen = gen_foo)
  })

  it("works for matrices", {
    expect_jit_torch_unary(prim_foo, torch::torch_foo, c(3, 4), gen = gen_foo)
  })
})
```

For binary ops, use `expect_jit_torch_binary` with `gen_x` / `gen_y`.

### Reverse test example (torch comparison in `inst/extra-tests/test-primitives-reverse-torch.R`)

```r
describe("prim_foo", {
  gen_foo <- function(shp, dtype) {
    n <- if (!length(shp)) 1L else prod(shp)
    vals <- c(0.5, -0.5, 1, -1, 2, -2, sample(rnorm(100), max(0, n - 6)))
    vals <- vals[seq_len(n)]
    if (!length(shp)) vals else array(vals, shp)
  }

  it("scalar gradient", {
    verify_grad_uni(prim_foo, torch::torch_foo, gen = gen_foo)
  })

  it("array gradient", {
    verify_grad_uni_array(prim_foo, torch::torch_foo, shape = c(3, 4), gen = gen_foo)
  })
})
```

For binary reverse tests, use `verify_grad_biv` / `verify_grad_biv_array` with `gen_lhs` / `gen_rhs`.

### Key testing helpers

| Helper                    | File                                               | Purpose                                  |
| ------------------------- | -------------------------------------------------- | ---------------------------------------- |
| `expect_jit_torch_unary`  | `inst/extra-tests/torch-helpers.R`                 | Compare unary forward with torch         |
| `expect_jit_torch_binary` | `inst/extra-tests/torch-helpers.R`                 | Compare binary forward with torch        |
| `verify_grad_uni`         | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare unary gradient (scalar + array)  |
| `verify_grad_uni_array`   | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare unary gradient (array only)      |
| `verify_grad_biv`         | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare binary gradient (scalar + array) |
| `verify_grad_biv_array`   | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare binary gradient (array only)     |
| `generate_test_data`      | `inst/extra-tests/torch-helpers.R`                 | Random input sampling by dtype           |

Custom generators (`gen`, `gen_x`, `gen_y`, `gen_lhs`, `gen_rhs`) have signature `function(shp, dtype)` and return an R array (or scalar for `integer()` shape).

### Meta-test coverage

`tests/testthat/test-primitives-meta.R` automatically checks that every `prim_*` primitive has corresponding stablehlo and reverse tests. Your new primitive will be flagged if tests are missing. Tests must use the full `prim_<name>` identifier as the `describe()` / `test_that()` label (e.g. `describe("prim_foo", { ... })`).

## Verify

```r
devtools::document()
devtools::load_all()
devtools::test()  # or run specific test files
```

## Checklist

- [ ] Can be expressed in StableHLO
- [ ] Primitive defined: `prim_<name> <- new_primitive("<name>", function(...) { graph_desc_add(self, ...) })` with roxygen docs and `@export` (auto-registered into the internal primitive registry). For simple shapes, use `make_unary_op()` / `make_binary_op()` / `make_reduce_op()` / `make_compare_op()` instead of writing the body by hand.
- [ ] Inference rule: `infer_<name>()` in `R/rules-inference.R` (or a shared one), with every refusal naming the argument and showing its value, snapshotted in `test-rules-inference.R`
- [ ] StableHLO rule: `prim_<name>[["stablehlo"]]` in `R/rules-stablehlo.R`
- [ ] Reverse rule: `prim_<name>[["reverse"]]` in `R/rules-reverse.R`
- [ ] Quickr rule (optional): `quickr_register_prim_lowerer(prim_<name>, ...)` in `R/rules-quickr.R`
- [ ] API wrapper: `nv_<name>` added via `/add-api-function` skill
- [ ] Tests: primitive forward + reverse, property-based with edge cases (use `describe("prim_<name>", { ... })`)
- [ ] `devtools::document()` run
- [ ] `devtools::test()` passes
