---
name: add-primitive
description: Add a new primitive operation to anvl (prim_* function with stablehlo, reverse rules, and tests)
user_invocable: true
---

# Add a New Primitive to anvl

Read `vignettes/extending_primitive.Rmd` first — it is the primary guide with a complete walkthrough (primitive creation via `new_primitive()`, stablehlo rule, reverse rule, nv_* API, file organization). This skill covers additional details not in the vignette.

## Before Starting: Check StableHLO Support

1. Check `../stablehlo/R/` for an `op-<name>.R` file (e.g. `op-add.R`).
2. Check that the operation can be expressed in StableHLO
   I.e., either it exists as `stablehlo::hlo_<name>()` or it can be expressed as a combination of existing StableHLO operations.
3. If the op doesn't exist in stablehlo, stop and tell the user — it must be added there first.
4. Read the StableHLO SPEC (`../stablehlo/SPEC.md`) for the operation's semantics and constraints.

## Argument Naming

The primary array argument of a `prim_*` (and its `nv_*` wrapper) is always named **`x`** — never `operand`, `input`, `a`, or anything else. This holds even when StableHLO's own spec calls it `operand` or `input`.

- Multiple arrays in the same role: `xs` (a list, as in `prim_sort(xs, ...)`).
- Two symmetric operands of a binary op: `lhs` / `rhs`.
- Arguments naming a genuinely different role keep a descriptive name: `start_indices`, `update`, `weight`, `init`, `reductor`, `padding_value`, ...
- Dimension-number arguments derived from `x` follow it: `x_batching_dims`, `scatter_dims_to_x_dims`.

When a StableHLO builder or `*DimensionNumbers()` constructor takes the spec name, map anvl's name back at the call site rather than renaming the anvl argument, e.g.

```r
stablehlo::GatherDimensionNumbers(
  operand_batching_dims = x_batching_dims - 1L,  # spec name on the left
  ...
)
```

## Roxygen Documentation

Use templates from `man-roxygen/` where applicable:

- **Unary ops:** `@template param_prim_x_any` (or `_float`, `_signed_numeric`)
- **Binary ops:** `@templateVar dtypes <phrase>` + `@template params_prim_lhs_rhs`, where the
  phrase completes "Arrayish values of ..." (e.g. `any data type`, `data type floating-point`)
- **Return:** `@template return_prim_unary`, `return_prim_binary`, `return_prim_compare`, `return_prim_reduce`
- **Rules section:** `@templateVar primitive_id <name>` + `@template section_rules`
- **StableHLO link:** `@section StableHLO:\n Lowers to [stablehlo::hlo_<name>()].`
- Do NOT mention "1-based indexing" — it's the R default.
- Add `@export` to the roxygen block.

## Shortcuts for Simple Ops

The vignette shows the manual `graph_desc_add()` approach. For simple ops without extra parameters, pass a body produced by `make_unary_op()` / `make_binary_op()` to `new_primitive()`. Both helpers take a stablehlo type-inference function and rely on the lexically-bound `self` installed by `new_primitive()`:

```r
# Simple unary (e.g. prim_abs, prim_negate):
prim_<name> <- new_primitive("<name>", make_unary_op(stablehlo::infer_types_<name>))

# Simple binary (e.g. prim_add, prim_mul):
prim_<name> <- new_primitive("<name>", make_binary_op(stablehlo::infer_types_<name>))
```

## Reverse Rule: Additional Guidance

Beyond what the vignette covers:

- Build gradient expressions using `prim_*` primitives — never use R arithmetic directly.
- For non-differentiable points (e.g. `abs` at 0, `floor` everywhere), follow PyTorch conventions (subgradients, zero gradients, etc.). Read existing rules in `R/rules-reverse.R` for examples.

## Optional: Quickr Rule

If the primitive should also run under `local_backend("quickr")`, add a `quickr` lowering in `R/rules-quickr.R` via `quickr_register_prim_lowerer(prim_<name>, function(...) { ... })`. This emits plain R code for the quickr backend. If you skip it, the primitive still works on the pjrt backend — the quickr meta test simply excludes it from coverage.

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

  it("tensor gradient", {
    verify_grad_uni_tensor(prim_foo, torch::torch_foo, shape = c(3, 4), gen = gen_foo)
  })
})
```

For binary reverse tests, use `verify_grad_biv` / `verify_grad_biv_tensor` with `gen_lhs` / `gen_rhs`.

### Key testing helpers

| Helper | File | Purpose |
|--------|------|---------|
| `expect_jit_torch_unary` | `inst/extra-tests/torch-helpers.R` | Compare unary forward with torch |
| `expect_jit_torch_binary` | `inst/extra-tests/torch-helpers.R` | Compare binary forward with torch |
| `verify_grad_uni` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare unary gradient (scalar + tensor) |
| `verify_grad_uni_tensor` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare unary gradient (tensor only) |
| `verify_grad_biv` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare binary gradient (scalar + tensor) |
| `verify_grad_biv_tensor` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare binary gradient (tensor only) |
| `generate_test_data` | `inst/extra-tests/torch-helpers.R` | Random input sampling by dtype |

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
- [ ] StableHLO rule: `prim_<name>[["stablehlo"]]` in `R/rules-stablehlo.R`
- [ ] Reverse rule: `prim_<name>[["reverse"]]` in `R/rules-reverse.R`
- [ ] Quickr rule (optional): `quickr_register_prim_lowerer(prim_<name>, ...)` in `R/rules-quickr.R`
- [ ] API wrapper: `nv_<name>` added via `/add-api-function` skill
- [ ] Tests: primitive forward + reverse, property-based with edge cases (use `describe("prim_<name>", { ... })`)
- [ ] `devtools::document()` run
- [ ] `devtools::test()` passes
