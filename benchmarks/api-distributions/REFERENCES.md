# Reference validation

The sweep compares values with base R and gradients with analytic formulas.
Both references can have numerical errors. This guide explains how the harness
checks them and when a disagreement can be attributed to base R.

For commands, result selection and storage, see the [harness README](README.md).

## Contents

- [Candidate disputes](#when-base-r-is-the-weaker-side-candidate-disputes)
- [Validation procedure](#validate-refs--checking-the-references-against-high-precision)
- [Gradient references](#the-gradient-references-are-checked-too)

## When base R is the weaker side: candidate disputes

`punif(1e-100, 0, 1, lower.tail = FALSE, log.p = TRUE)` is 0; the answer is
`log1p(-1e-100)` = −1e−100, which anvl returns. The sweep keeps its
comparisons against base R. A spec may additionally declare a **stable reference** (`ref_stable`), an accurate evaluation of the
same function with the same parameters, and each sample is then tested
against it. A sample is a **candidate base R dispute** only when all three
hold, with *s* the stable value and *B* its declared error bound in absolute
units (`ref_stable_bound_ulp64` double ulps at *s*):

| condition | test |
|---|---|
| base R is off | \|g − s\| > 4 ulp_f64(s) + B |
| anvl is accurate | \|f − s\| ≤ 2 ulp_dtype(s) + B |
| anvl is no further | \|f − s\| ≤ \|g − s\| |

Where no ulp comparison is meaningful — *s* is ±0 or ±∞, or beyond the f32
range in an f32 cell — anvl must equal *s* at the cell's precision, sign of
zero included, and base R must not. A NaN on any side is never a dispute. The
multipliers are provisional thresholds for exclusion, not accuracy criteria:
failing them only leaves a sample counted against base R. The same test
records the opposite case too — anvl and base R returning the same value,
both beyond anvl's tolerance (`n_ref_shared`) — which a comparison against
base R alone can never see.

### Candidate records and reference identity

The sweep records candidate disputes alongside the original comparisons:

- candidate counts and errors with candidates omitted (`worst_rel_err_excl`
  and `worst_out_normal_excl`, per band, input class and result);
- a histogram column and a `ref_candidate` flag on regions and exact points;
- a `disputes` table retaining the samples furthest beyond tolerance in each
  binade, with all three values, both distances and both thresholds.

Readers use the adjusted figures only after the stable reference passes
validation under a matching identity. Until then, the candidates remain
included in the assessment.

Each result's `ref_stable_id` hashes the stable reference, dispute classifier
and spacing function. The hash follows their non-base dependencies, including
functions, captured values and package versions. It also includes the declared
bound, exact reference parameters, flags, dtype, parameter policy and R build.
Changing a helper such as `same_value()` or a constant such as `DISPUTE_K`
therefore changes the identity.

The `ref_params` field stores the reference's parameters as hex doubles.
Validation uses those values, so it can reproduce the parameters even if the
spec's named parameter set has since changed.

`nv_punif`'s stable reference uses the same small-tail algorithm as anvl.
Agreement between them in f64 becomes independent evidence only through the
MPFR check.

Currently, only `nv_punif`'s log-scale value cells declare a stable reference.
Evaluating it adds work to those cells; measure that cost on the target system.

## `validate-refs` — checking the references against high precision

```bash
Rscript run.R validate-refs                      # selected latest/deepest results
Rscript run.R validate-refs --filter spec=nv_punif --prec 512
Rscript run.R validate-refs --shard 3 --shards 32   # one share, for a cluster array
```

The command needs Rmpfr and validates the selected current results. It includes
all backends in the store unless `--backends` restricts them. See the README's
[result-selection rules](README.md#which-results-are-selected), including the
limits of `--run` and stores containing multiple platforms.

Work is grouped by reference kind, identity and output. An anvl cell and its
JAX twin can therefore share one validation, using retained inputs from both.
Validation shards must read the same fixed sweep data. Each invocation writes
separate records to the shared store; no second merge is needed.

### Identity checks

Validation first derives the reference identity from the current code and the
stored `ref_params`. If it differs from the sweep's identity, validation fails
before evaluating the reference.

The code must also be identifiable. Dynamic calls such as `get()`, `do.call()`
and `eval()`, including qualified forms such as `base::get()`, cannot be
followed by the identity calculation. The same applies to function names passed
as strings to `sapply()` and similar functions. A reference or MPFR truth using
an unsupported form cannot pass validation.

### Sampling

Each reference unit draws inputs from its selected results:

- exact points and retained disputes;
- the two worst retained inputs per sign and binade;
- retained earlier failures for the same cell and output, including failures
  from older reference and truth versions;
- `--per-binade` random patterns per exponent field and sign (default 1);
- `--focus-per-binade` additional patterns (default 8) per sign in binades
  containing stable-reference disputes or shared errors, or gradient errors
  within a factor of 1,000 of the worst finite sweep error;
- `--random` additional patterns per sign (default 512).

Random patterns are drawn with the sign bit clear, then negated to cover both
signs. Thus `--random 512` adds 1,024 inputs before deduplication. The seed comes
from the cell ID and output; a shared unit uses its first member in sorted
cell/output order. Duplicate input bit patterns are removed.

### Comparison and retained samples

The reference is compared with the spec's `ref_stable_mpfr` or `ref_grad_mpfr`
at `--prec` bits (default 256). Error is measured in f64 ulps at the truth.
The subtraction, division by the spacing and comparison with the bound all
happen in MPFR. Rounding the difference to double first can hide part of a
subnormal ulp.

An exactly zero truth requires a zero reference, allowing either sign. A NaN
or infinite truth requires the same value. A truth that rounds beyond the
double range requires the corresponding infinity. Validation passes only if
every sample meets its rule or falls within the declared bound.

The `validations` table records the identities, precision, seed, sampling
settings, sample count, worst error, pass/fail result and reason. The
`validation_samples` table retains the union of:

- the 20 samples with the largest errors;
- the first 200 failing samples.

Later checks replay the retained failures. Failures beyond this retention limit
may not be revisited. Both tables are included in exports for the selected
reference identities.

### Truth and method identities

Each record identifies the MPFR truth, its dependencies and the selected
output. It also identifies the validation method: the sampler, comparator and
procedure. Records from a different method are ignored when determining
current validation status.

An MPFR truth follows the reference's endpoint and out-of-domain conventions,
using the requested precision to evaluate the mathematics. It still needs
stable formulas. At 256 bits, for example, `1 - 4e-78` rounds to 1. The truths
use small-tail expressions such as `log1p` and `expm1` to avoid this loss. The
normal family's deep tails use a continued fraction and log-space evaluation
because MPFR's `exp()` underflows at about |z| = 4e6.

### Validation status and exclusions

Status is determined separately for each reference identity and gradient
output, using records from the current validation method and the most recent
MPFR truth identity used in those records:

| status | meaning |
|---|---|
| `validated` | at least one matching validation passed and none failed |
| `failed` | at least one validation failed; a later pass does not undo it |
| `not validated` | no applicable validation record |
| `no identity` | the reference uses a code form the identity calculation cannot follow |

When a validation uses a new truth identity, records for older truths remain
in the store but no longer determine status. Editing the truth alone does not
update stored status; run validation with the changed truth.

A candidate dispute with a validated stable reference becomes a
`reference_limitation` when results are read. Readers show adjusted error
summaries beside the original comparisons against base R. Other candidates
remain included. Gradient-reference status is reported but does not change
how gradient errors are scored.

Passing sampled validation provides evidence for the sampled inputs. It does
not prove a global error bound.

## The gradient references are checked too

A gradient's reference is an analytic formula written here, not base R, and
being analytic does not make its floating-point evaluation trustworthy. The
normal family's are built to avoid the ways the obvious evaluation fails:

- `z = (x - mean)/sd` is carried as a double-double (`std_z()`), because a
  rounded z costs `exp(-z^2/2)` about z² ulp — 85 ulp at z ≈ 8 on the shifted
  parameters — and anvl and base R round z the same way, so a reference that
  did too would share their error instead of measuring it;
- `m · φ(z)` and `m / φ(z)` go through `phi_times()` / `phi_recip()`, which
  split z so its square is exact and scale so nothing under- or overflows
  before the answer does: φ becomes subnormal around |z| = 37.6 and rounds
  to zero by z = 38.6, while `(z² − 1)φ(z)` is still about 1.7e−321 there;
  for huge z the bracket can overflow even though the answer is 0;
- `(z² − 1)/sd` as `(z − 1) · ((z + 1)/sd)`, finite where z² overflows;
- the inverse Mills ratio as a direct ratio above z = −20 and a continued
  fraction below, not a difference of two logs of ~−z²/2.

The normal-family specs currently declare a **16 f64 ulp** bound for gradient
reference validation. Run `validate-refs` and inspect the records for the
reference identity in use. `selftest` also checks selected
regression cases against recorded MPFR values.

