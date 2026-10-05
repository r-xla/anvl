# `api-distributions` — bit-pattern sweeps of the distribution API

This harness checks all nine distribution functions in `R/api-distributions.R`,
in f32 and f64. It compares values with base R and first derivatives with
analytic reference formulas, using bit-pattern sweeps and separate exact-point
checks.

The benchmark measures accuracy; interpreting an error depends on the function,
precision and application. Algorithm-comparison reports live separately in
[`../nv_pnorm/`](../nv_pnorm/) and [`../nv_qnorm/`](../nv_qnorm/).

Results are read on the [anvl-bench site](https://github.com/r-xla/anvl-bench/blob/main/README.md),
which renders an `export` from this harness. The harness has three uses:

- **Producing the site's data:** `run`, `validate-refs`, `status`, then
  `export`. Run it on one machine ([Quick start](#quick-start); the site's
  [local development](https://github.com/r-xla/anvl-bench/blob/main/README.md#local-development) section
  shows how to view the export), or across a cluster ([HPC.md](HPC.md)).
- **Comparing a change while working on an implementation:** rerun the sweep
  after reinstalling anvl, then `diff` against the previous run
  ([Comparing a change](#comparing-a-change)).
- **Taking failing inputs into tests:** `query.R` reads the store directly and
  returns the worst inputs with their bit patterns, ready for a regression test
  ([Reading the store directly](#reading-the-store-directly)).

## Contents

- [Quick start](#quick-start)
- [Coverage and result selection](#coverage-and-result-selection)
- [Inspecting results](#inspecting-results)
- [Reading what a disagreement means](#reading-what-a-disagreement-means)
- [Storage and export](#storage-and-export)
- [Adding a function](#adding-a-function)
- [Reference validation](REFERENCES.md)
- [Distributed execution](HPC.md)

## Quick start

Run the examples from `benchmarks/api-distributions`. Install anvl and its
sibling dependencies first, plus `nanoparquet`, `cli` and `Rmpfr` for the
workflow below. JAX cells also need `reticulate` and Python with `jax`.
`sw_sql()` needs `duckdb` and `DBI`.

The harness calls the **installed `anvl` package**. Reinstall after changing
anvl. Provenance records the checkout's SHA separately; it does not verify
that the installed package matches the checkout or capture uncommitted changes.

Choose new store and export directories outside the repository:

```bash
export NV_SWEEP_STORE=/absolute/path/to/sweep-store
Rscript run.R run --depth smoke
Rscript run.R validate-refs
Rscript run.R status
# Inspect coverage, errors and reference statuses before exporting.
Rscript run.R export --out /absolute/path/to/sweep-export
```

The anvl-bench site reads the export. To view it locally, export into the
site's `data/` directory and serve it, as its
[local development](https://github.com/r-xla/anvl-bench/blob/main/README.md#local-development)
section describes.

To publish it, zip the **contents** of the `--out` directory, not the directory
itself, so `manifest.json` and the Parquet files sit at the root of the zip.
Name the zip after the platform the sweep ran on, which is also the
`platforms` entry in `manifest.json`:

```bash
cd /absolute/path/to/sweep-export
zip -q -r ../darwin-arm64-cpu.zip . -x '.*'
```

Then attach the zip to an anvl-bench release and deploy it, as
[How the results get here](https://github.com/r-xla/anvl-bench/blob/main/README.md#how-the-results-get-here)
describes; its
[release asset layout](https://github.com/r-xla/anvl-bench/blob/main/README.md#release-asset-layout)
section describes the layout the site expects.

### Comparing a change

After editing and reinstalling anvl, rerun the same selection at the same depth
in the same store:

```bash
Rscript run.R run --depth smoke
Rscript run.R diff
```

Use a fresh store for publication of a particular version. An accumulating
store can combine results from several versions or depths.

### Checking the harness

After changing the harness's `R/` files, run its synthetic checks in a separate
store:

```bash
Rscript run.R selftest --store /absolute/path/to/selftest-store
```

JAX helpers use the interpreter named by `RETICULATE_PYTHON` when set.
Otherwise they try the sibling `py-benchmarks/.venv` if it exists, then leave
interpreter selection to reticulate.

### Commands and options

| command | selection and useful options | writes |
|---|---|---|
| `list` | current grid; anvl by default; `--filter`, `--backends` | none |
| `run` | current grid; anvl by default; `--depth`, `--filter`, `--backends`, `--jobs`, `--topk`, `--shard`/`--shards`; `--dry-run` previews it | sweep store, except with `--dry-run` |
| `status` | current results and attempts; all stored backends by default; `--filter`, `--backends` | none |
| `diff` | newest matching run against previous attempts; `--from`, `--to`, `--filter`, `--backends` | none |
| `validate-refs` | current results; `--filter`, `--backends`, `--run`, `--prec`, sampling options, `--shard`/`--shards` | validation records and retained samples |
| `export` | current results and coverage; `--filter`, `--backends`; requires `--out` | export directory |
| `merge` | all table files from `--from` | destination store |
| `selftest` | synthetic cells and harness assertions | selected store |

`--store` selects the store for these commands; `list` does not read it.
[Result selection](#which-results-are-selected) defines “current” and explains
how depth, platform and errors affect each reader.

### Terms

- **Cell:** one function, backend, precision, value/gradient mode, parameter set
  and flag combination; the unit of work.
- **Result:** one output of a cell. A value cell produces one result; a gradient
  cell produces one per differentiated argument.
- **Binade:** a range of magnitudes between successive powers of two. The
  harness groups by sign and exponent field, with special handling for zeros,
  subnormals, infinities and NaNs.
- **Ulp:** a unit in the last place—the spacing at a value in a given precision.
  Sweep error uses the result's precision; reference validation uses f64 ulps.

The current anvl grid has **280 cells and 500 results**. `run.R list` reports
the grid generated by the current specs.

### Files

```text
run.R          list, run, status, diff, export, merge, selftest,
               validate-refs
query.R        read the store from R or the shell
R/util.R       bit patterns, ulp spacing, formatting
R/engine.R     enumeration, scoring, reducers, region classification
R/cells.R      spec definitions, grid, cell IDs, filtering
R/store.R      Parquet store and result selection
R/provenance.R environment fingerprint
R/validate.R   reference validation against MPFR
sweeps/        function specs and shared helpers
REFERENCES.md  reference validation and analytic formulas
HPC.md         cluster execution
```

## Coverage and result selection

### The three depths

A sweep walks a pattern-index space of 2³¹ indices per sign. Depth sets the
stride: coarser sweeps sample across the range but can miss localized failures.
In f64, random low bits mean that exact zeros and infinities are generally
absent from the sweep. Separate exact-point checks cover them at every depth.

| depth | stride | samples per cell | intended use |
|---|---|---|---|
| `smoke` | 8,192 | 524,288 | default; after changes |
| `quick` | 128 | 33,554,432 | broader checks |
| `full` | 1 | 4,294,967,296 | exhaustive f32 or stratified f64 coverage |

These counts exclude exact-point checks. Time representative cells on your
setup before scheduling a full run.

In `f32`, `full` enumerates all 2³² bit-pattern indices once. Finite values,
signed zeros and infinities are delivered exactly. Widening to double quiets
signalling NaNs, and comparisons ignore NaN payloads and signs.

In `f64`, `full` takes one
sample from each of the 2³² contiguous blocks of 2³² — every (sign, exponent,
top-20-mantissa-bit) combination, with the low 32 bits drawn from a fixed seed
so another machine reproduces the same inputs.

`--filter spec=nv_qnorm` selects **32 anvl cells and 64 results**. Its runtime
depends on the precision, outputs and machine. See [HPC.md](HPC.md) for distributing larger runs.

### Selecting what to run

```bash
Rscript run.R list                                    # what cells exist
Rscript run.R run --filter spec=nv_qnorm --depth full
Rscript run.R run --filter 'dtype=f64,kind=grad'
Rscript run.R run --filter 'spec=nv_punif|nv_qunif' --depth quick --jobs 8
Rscript run.R run --backends anvl,jax                 # JAX side too
Rscript run.R run --dry-run --depth full              # what would run
```

`--filter` takes comma-separated `key=value` terms; `|` inside a value means
“any of”. For `run` and `list`, use `spec`, `family`, `backend`, `dtype`, `kind`,
`param_set`, `flags` or `cell_id`. `flags` and `cell_id` also allow substring
regular-expression matches.

`status`, `diff`, `export` and `validate-refs` additionally accept `output` to select
an individual result. `run` does not: a gradient cell computes all derivatives
together. Use `run --dry-run` or `list` to check a selection. Unknown leading
keys error, but an unrecognized term after a valid key can be interpreted as
part of that key's value because cell IDs themselves contain commas.

`--jobs N` uses forked workers on supported systems. Each cell writes separate
files. `run` and `list` default to anvl; include `--backends anvl,jax` to
select JAX too. `status`, `diff`, `export` and `validate-refs` default to all
backends represented in the store.

### Which results are selected?

| reader | selection |
|---|---|
| `sw_results()`, `sw_worst()` | latest per cell, output, platform and depth |
| `status`, `export`, `validate-refs` | newest attempt per cell, platform and depth; of those that succeeded, the deepest per cell/output (`current_results()`) |
| `diff` | newest matching run versus each cell's previous attempt at the same platform and depth, succeeded or errored; `--from`/`--to` select runs |
| `sw_detail()`, `sw_hist()`, `sw_bands()`, `sw_ranges()`, `sw_points()` | matching historical rows, not just the latest run |

An older full sweep can therefore take precedence over a newer smoke sweep in
exports. `validate-refs` applies `--run` after latest/deepest
selection; it does not retrieve arbitrary historical results by run ID.

**Current platform limitation:** `deepest_per_cell()` groups by cell/output
without platform. In a store containing multiple platforms it retains only
one platform's result for each cell/output. Keep separate stores per platform
for these readers. Selection can also combine runs and anvl versions; use a
fresh store for a version-specific publication. These are selection behaviours,
not guarantees that the selected data forms a coherent snapshot.

## Inspecting results

Results themselves are read on the site (see the top of this README). The
harness's own commands check the store and compare runs.

### `status` — is the store complete?

```bash
Rscript run.R status
Rscript run.R status --filter spec=nv_qnorm --backends anvl,jax
```

`status` reports three things, for every backend the store holds unless
`--backends` names some:

- **Coverage.** For each platform and depth, how many of the declared cells
  were swept, and which cells have no successful result at any depth. A cell
  counts as swept at a depth when its newest attempt there succeeded.
- **Errors.** The count of cells whose newest attempt errored, with up to 20
  entries showing platform, depth, run and the first line of the error. An error superseded by a later
  successful run of the same cell, platform and depth is not listed.
- **References.** How many stable and gradient references are validated,
  failed, not validated or have no identity, and which results use a failed
  reference or one with no identity.

Terminal lists are truncated: at most 10 cells missing a successful sweep,
20 errored attempts, and 20 results with failed or unidentified references.
Omitted entries are counted. For the full list of current errors, read the
store from R:

```r
source("query.R")
attempts <- latest_attempts(latest_results(sw_store()))
attempts[!is.na(attempts$error), ]
```

`run` records a cell's error and carries on, so its exit status does not show
whether a sweep completed; `status` is where that is checked before an export.
`export` applies the same rules and carries them to the site, so a cell that
errored or was never run is shown there too.

### Retaining worst inputs per binade

The store keeps up to **10 inputs with positive finite relative error per sign
and binade**, with zeros handled separately. `--topk N` changes this limit.
Inputs are ranked by relative error, so this is not a separate shortlist of
the largest ulp errors. Non-finite disagreements are retained as regions.

Per-binade retention preserves examples across input magnitudes instead of
letting one small interval fill the entire shortlist. Returned detail rows are
ranked globally and carry `binade` and `sign` to join onto the behaviour bands.

### `diff` — did the change help?

```bash
Rscript run.R diff                       # newest run vs each cell's previous attempt
Rscript run.R diff --from <run> --to <run>
Rscript run.R diff --filter spec=nv_qnorm
```

```
  selftest  d/dscale f64  clean/broken
    worst rel err        1.00e-06 -> 0.001   (1e+03x worse)
    worst ulp            4.86e+09 -> 4.89e+12

SUMMARY
     4 regressed
     0 improved
    20 unchanged (bit-identical to the earlier run)
     0 had no earlier result to compare against
```

**Input sampling is deterministic** for a fixed harness, seed, dtype and depth.
Numerical results also depend on the installed packages, runtime and settings.
`diff` compares the largest relative error across sweep and exact points,
region and point counts for failures, boundaries, backend limitations and
undefined-domain conventions, and the signed-zero sample count.

The printed phrase “bit-identical to the earlier run” means only that those
metrics match exactly. Individual outputs, histograms, retained inputs and
reference-limitation counts can differ. A change only in worst ulp error can
also be reported as unchanged. A reported change needs investigation; package
versions, runtime settings and reference validation can affect the comparison.

Results pair by cell, output, platform **and depth**. A smoke result and a full
result sample different inputs, so pairing across depths would report the extra
coverage as a regression; an unpaired result is reported as having nothing to
compare against instead.

Each cell is compared with its **previous attempt** at that platform and depth:
the most recent earlier run that swept it, whether it succeeded or errored. A
cell that errors where its previous attempt succeeded is a regression, listed
first; a cell that succeeds where its previous attempt errored is an
improvement, never compared with an older result from before the error. A cell
that errored both times is listed as still erroring, with the earlier message
where it differs, and one that errors on its first attempt is listed apart. As
with `status`, an output filter narrows the successful results only, since an
errored cell has no output to select.

## Reading what a disagreement means

**A disagreement does not mean anvl is wrong.** base R is the reference, not the
truth, and it is sometimes the weaker implementation. At `x = 5.551e-17`:

```r
punif(x, 0, 1, lower.tail = FALSE, log.p = TRUE)         # base R:  0
as.double(nv_punif(nv_array(x, dtype = "f64"), 0, 1,
                   lower_tail = FALSE, log_p = TRUE))    # anvl: -5.551e-17
log1p(-x)                                                # correct: -5.551e-17
```

base R forms `1 - x` first, which rounds to exactly 1, and loses the value
entirely; anvl goes through `log1p` and keeps it. The sweep correctly reports a
disagreement, and the right response is to leave `nv_punif` alone.

### How disagreements are classified

A disagreement with no finite relative error is routed to the failure-region
tracker, unless it matches the reference rounded to the result's precision.
The tracker groups adjacent failing samples and assigns a cause using the
tests below. `unidentified` means none of those tests explains the disagreement.

| cause | tested how | category |
|---|---|---|
| `nan_input` | the input is NaN | failure |
| `input_flushing` | a subnormal input whose result is **bit-identical** to the function's own result at the same-signed zero, **and** that zero result is **checked against the zero reference**: equal to it, including the sign of zero, or equal to it rounded to the result's precision | backend limitation |
| `flush_inherits_zero_error` | as above, but the zero result fails that check — *any* error at zero, finite or not: the subnormals inherit it | failure |
| `domain_boundary` | an endpoint of the valid input domain (e.g. p = 0, p = 1), or a subnormal inheriting the behaviour of a zero that is one | boundary |
| `outside_domain` | wholly outside the valid input domain | failure — or, for a gradient, see below |
| `zero_input` | ±0 that is not a domain endpoint | failure |
| `inf_input` | ±∞ inside the domain | failure |
| `unidentified` | none of the above | failure |

This zero-reference check is separate from MPFR reference validation.
It compares the implementation and reference at zero during the sweep.

The initial categories are listed below. Validated reference disputes can
also become `reference_limitation`, described under [Reference validation](#reference-validation):

- **failure** — an unexplained disagreement, including an incorrect value
  outside the valid domain where NaN is required.
- **backend limitation** — the platform, not the function: input flushing.
- **boundary** — behaviour at a domain endpoint, which needs an explicit
  convention or limiting value; the reference supplies the limiting value.
- **undefined domain** — a *gradient* outside the valid input domain where the
  forward values on both sides are NaN, so no derivative is defined and the two
  sides differ only in convention (e.g. `qnorm` for p > 1: anvl's d/dp is NaN,
  the reference's is 0). A gradient cell never computes forward values, so this
  is **established** from the matching value cell, which swept the identical
  inputs: only where it found both values NaN for every out-of-domain sample in
  every binade the region touches. Anything less stays a failure. This
  category is set aside as an undefined-domain convention; doing so does not
  validate the derivative.

**The evidence for a convention must come from the same run.** The value cell
is looked up by the gradient region's own `run_id`, which fixes the platform,
the anvl build, the harness, the depth and the seed together, so "the identical
inputs" holds by construction. Evidence from any other run is never used: a
gradient cell re-run on its own stays a failure, and its region's `evidence`
column says why. Exact points follow the same rule, matched by bit pattern.
Resolution always reads the **whole store** — `status`, `diff`, `query.R`
and `export` all go through the same `resolved_ranges()` /
`resolved_points()` — so a filter can narrow what is *selected* but never what
is used as evidence: a gradient-only export classifies exactly as a full one
does.

Zero is never a subnormal: it is the value a subnormal is flushed *to*, and it
is checked, not exempted. The same holds in the aggregates: each sign's swept
zero is a band row of its own (`zero = TRUE`), binade 0 holds only the
subnormals, and the categories give zero its own input class. A value
disagreement outside the domain, where NaN is required, is a failure.

Alongside the cause, each region records **what the two sides returned**, as a
tally of (value kind, reference kind) pairs over the kinds NaN, +∞, −∞, +0, −0,
subnormal and normal, plus a representative input with its bits and both
values. For f64 the region's bounds are those of the 2³²-pattern blocks its
failing samples fell in and are **not** evidence that the endpoints were
evaluated; the first and last *sampled* failing inputs are carried separately
(`sampled_from`, `sampled_to`) and are the ones to cite.

The same kind pairs are tallied for every binade in the `kinds` table, in and
out of the domain separately, and two further things are counted per binade
without being failures: **signed-zero disagreements** (`n_zero_sign`: both
sides zero, opposite signs, which `==` cannot see) and **differences caused by
input flushing**, split by whether the result at zero passes the
zero-reference check: `n_flushed` when it passes, and
`n_flushed_zero_error` when it fails and the subnormals also carry the error at
zero. Their error magnitudes stay in the histogram and worst inputs.

This is measurement, not judgement: it says what the numbers are and why.

### Reference validation

`validate-refs` checks stable value references and analytic gradient references
against MPFR at 256 bits by default:

```bash
Rscript run.R validate-refs
Rscript run.R validate-refs --filter spec=nv_punif --prec 512
```

A stable reference supplies additional evidence where base R may be inaccurate.
Currently `nv_punif`'s log-scale value cells and the lower-tail log-scale value
cells of `nv_pexp` and `nv_qexp` declare one. A candidate dispute
becomes a `reference_limitation` only after that reference passes validation
under a matching identity. Comparisons against base R remain available beside
the adjusted figures. Gradient-reference validation reports whether the
reference passed; it does not change how gradient errors are scored.

Validation samples exact points, retained worst inputs and disputes, retained
counterexamples, and random inputs. Passing provides evidence on those samples;
it does not prove a global error bound. The normal-family gradient references
currently declare a bound of 16 f64 ulps, the exponential family's 8.

See [Reference validation](REFERENCES.md) for the dispute thresholds, sampling,
retention limits, reference identities, and analytic formulas.

### Exact points, beside every sweep

The f64 sweep draws its low 32 bits at random, so it essentially never lands on
±0, ±∞, or an exact point such as p = 1 — a boundary bug there is invisible
without a separate check. Every cell therefore also evaluates a fixed set of
**exact points**, kept in their own `points` table and never added to the
sweep's counts, so nothing is counted twice:

- ±0, ±∞, NaN, the smallest and largest subnormal, the smallest normal, the
  largest finite value, ±0.5, ±1;
- each spec's valid-domain endpoints and distribution-support edges;
- anvl's **branch points**, where its implementation switches algorithm (anvl
  cells only);

every point **at the cell's precision** — a boundary of an f32 cell is the
boundary after conversion to f32 — and every finite one, universal points
included, with its two representable neighbours. The ±0 results are recorded
explicitly, so "zero passes" is a recorded comparison, not an absence of
regions.

The points take part in every assessment without entering the sweep's counts.
Each results row carries their summary (`n_points`, `n_points_identical`,
`n_points_failure`, `n_points_boundary`, `n_points_backend`,
`n_points_domain`, `worst_point_rel_err` and where), a failing point makes its
result failing, the site lists every point that is not bit-identical, and
`Rscript query.R points --category failure` lists them.

### What a result's state counts

`result_state()` (`R/store.R`) turns a result's counts into flags, and the
site's `js/model.js` ports it. Every category counts separately: **failures**,
**domain boundary behaviour** and **backend limitations** each count regions
and exact points, and the largest finite error is taken over the sweep and the
points together. A result is **bit-identical** only if every sampled input and
every exact point matches its reference down to the sign of zero
(NaN payloads and signs are ignored); a result that
differs *only* by undefined-domain conventions is flagged as such, as set
aside. `diff` compares the subset of metrics listed in its
[command description](#diff--did-the-change-help).
Unvalidated candidate disputes remain included. Once validated, verified
reference limitations have separate counts and adjusted error summaries beside
the unchanged comparisons against base R.

### The reference sees the parameters the implementation sees

anvl converts a bare `min = -pi` to f32 for an f32 argument, so an f32 cell's
reference is evaluated with its parameters rounded to f32 too. Otherwise the
two sides compute different functions: f32(−π) lies *below* the double −π, and
at x = f32(−π) anvl is inside the support while a double-parameter reference is
outside — a failure by construction. Domain, support and branch points are
taken from the same rounded parameters.

### Correctly rounded is not the same as zero error

The reference is evaluated in double precision and is never rounded before
scoring, so the
relative error measures numerical error against it. Separately, each sample
records whether the result is that reference **correctly rounded to the
result's precision** (`n_rounded` on the summary and bands, `rounded` on each
worst input). The two agree almost everywhere and part company at the edges of
the f32 range:

| base R's value | correctly rounded f32 | relative error | routed as |
|---|---|---|---|
| magnitude ≥ f32 overflow threshold (2^128 − 2^103) | ±∞ | infinite | a match, not a failure |
| below half the smallest subnormal | ±0 | exactly 1 | a finite error, marked rounded |
| representable, result ±∞ | — | infinite | a failure (spurious overflow) |

Both statements are kept because both are true: the implementation can be
perfectly rounded while its error against the reference is still 1. For f64 the
reference is already a double, so rounded and identical coincide and
`n_rounded` is always 0. The rounding itself relies on the platform's
double-to-float conversion; `run.R selftest` checks it at both edges on the
machine that runs the sweep.

### A known platform property: subnormal flush-to-zero

Earlier PJRT/XLA CPU observations found arithmetic and comparisons flushing
subnormals to zero in both precisions, while storage preserved them. Those
observations did not record a runtime version here, so they are not a guarantee
for every operation or platform. The following examples illustrate the observed
behaviour; check them against the installed runtime:

```r
as.vector(nv_array(-1e-39, dtype = "f32") >= 0)             # TRUE  (R says FALSE)
as.double(nv_dunif(nv_array(-1e-39, dtype = "f32"), 0, 1))  # 1     (R says 0)
```

The harness tests input flushing per sample: a failure is attributed to input
flushing only if the result is bit-identical to the function's own result at
the same-signed zero and that zero result passes the zero-reference check
(see the cause table above). A flush that changes the answer still counts, however large the
change: a subnormal whose result is its signed zero's and differs from base R
is counted in `n_flushed` when that zero result passes the check, and in
`n_flushed_zero_error` when it fails. Either way its error stays in the histogram and worst inputs.
`Rscript query.R ranges --cause input_flushing` lists the attributed regions.
A runtime change can change that classification; agreement still depends on
the function result.

**Never write the literal `-0` inside a function in this harness.** R's
byte-code compiler (R 4.6.1, JIT level 3 — verified) folds it to `+0` in some
call shapes: `c(0, -0)` in a compiled function returns two positive zeros. Use
`NEG_ZERO` (`R/util.R`), built from its bit pattern; `selftest` checks that it
survives.

## Storage and export

### `export` — publishing selected results

Export uses the latest/deepest selection described above. It does **not**
enforce one version or platform, complete function coverage, or a single run.
For a coherent publication, sweep into a fresh store for the intended installed
version and platform, validate its references, then inspect coverage with
`status` before exporting. Use a fresh output directory so old optional tables
cannot remain from a previous export.

```bash
Rscript run.R export --out ../anvl-bench-darwin-arm64-cpu
Rscript run.R export --out <dir> --backends anvl     # restrict, if you mean to
```

Unlike `run`, `export` does not default to anvl alone: it exports whatever
backends the store holds. It exports errored and never-run cells as such in
`coverage.parquet`, and warns when there are any; it refuses only when there is
no successful result at all.

```
manifest.json     index: schema version, platform, specs, depths, coverage counts,
                  row counts
runs.parquet      environment fingerprints for runs contributing selected
                  successful results
summary.parquet   selected successful results
coverage.parquet  every declared cell: the depth of its deepest successful
                  sweep and the latest error at its deepest errored depth;
                  includes errored and never-run cells
detail.parquet    the worst inputs, per binade
bands.parquet     profile per result, sign and exponent group, with separate
                  rows for swept zeros
hist.parquet      the error distribution
ranges.parquet    the no-finite-error regions, resolved (cause, category, evidence)
kinds.parquet     what each side returned, per binade
points.parquet    the exact points, resolved
categories.parquet  per-result figures by input class (normal, zero, subnormal,
                  outside the domain, ±∞ & NaN); each cell's domain and support
                  are on summary
disputes.parquet  retained candidate disputes and shared reference errors
validations.parquet  reference-validation records
validation_samples.parquet  retained validation samples and failures
```

A coverage error can refer to a run absent from `runs.parquet`; look up that
run in the source store for its provenance. The error shown is selected by
depth first, so an older full-depth error can take precedence over a newer
smoke-depth error.

Schema version is currently 8. Optional tables are written only when data is
available; consult `manifest.json` for the actual file inventory.

One file per **table**, not per function. The overview page summarises every
function at once, so splitting by function would mean fetching all the pieces
anyway in more requests, and would turn "look at another platform" into many
downloads instead of one artifact.

`manifest.json` is deliberately JSON and carries no measurements: it is the
index, readable without a Parquet reader. The measurements stay in Parquet
to preserve NaN, infinities, signed zero and subnormals. Standard JSON has no
numeric representation for NaN or infinities, and signed-zero preservation
depends on the serializer and consumer.

### Row groups are aligned to cells, so drilling in is cheap

Export sorts `detail`, `bands`, `hist`, `ranges`, `kinds`, `points` and
`disputes` by cell/output and starts a row group at each cell boundary. Readers
that support selective row-group access can fetch a cell without reading the
whole table. Transfer size depends on the data and reader.

### The store

Results are **Parquet, written once, never mutated** — one file per cell per
run, in a partitioned directory.

Each cell writes separate files within each table and run partition. Workers
can write independently, and `merge` copies table files between stores.
Parquet preserves the numeric values needed here, including NaN, infinities,
signed zero and subnormals. Bit patterns are stored as hex strings so consumers
do not need to preserve unsigned 64-bit integers.

```bash
export NV_SWEEP_STORE=/path/to/store
Rscript run.R merge --from /other/machine/store
```

The default is `file.path(tools::R_user_dir("anvl-sweeps", "cache"), "store")`.
`NV_SWEEP_STORE` or `run.R --store` can override it; choose a location outside
the repository. Runs accumulate without replacing earlier results. Reader
selection differs by command, as described above.

Every run records host, CPU, OS, architecture, device label, R version,
`default_dtypes()`, package versions and checkout SHAs. SHAs are read directly
from `.git`, without invoking Git. The anvl SHA comes from the checkout
containing this harness; it does not verify the installed package's source.
`NV_SWEEP_DEVICE` supplies the device label (default `cpu`); set it to match the
runtime actually used. It labels the run and does not select a device.

### Reading the store directly

```bash
Rscript query.R worst --spec nv_qnorm --n 20
Rscript query.R ranges --category failure
Rscript query.R points --category failure
Rscript query.R detail --cell 'nv_qnorm/anvl/f64/value/standard/lower_tail=TRUE,log_p=FALSE'
Rscript query.R runs
```

or in R:

```r
source("query.R")
sw_worst(spec = "nv_pnorm", metric = "ulp")
sw_detail("nv_punif/anvl/f64/grad/wide/lower_tail=TRUE,log_p=TRUE", output = "min")
sw_hist("nv_pnorm/anvl/f64/value/standard/lower_tail=TRUE,log_p=FALSE")
sw_compare("darwin-arm64-cpu", "linux-x86_64-cuda")
sw_sql("select spec, dtype, max(worst_rel_err) from results group by 1, 2")
```

`sw_detail()`, `sw_hist()` and the other history readers can return rows from
multiple runs or platforms, and some omit run identifiers in their output. For
an unambiguous historical query, use `sw_sql()` or read a raw table and filter
by `run_id`, `cell_id` and `output`.

`sw_detail()` gives the worst individual inputs **with their bit patterns** —
paste a `bits` value straight into a regression test in
`tests/testthat/test-api-distributions.R`.

## Adding a function

Start from an existing spec such as [`sweeps/dnorm.R`](sweeps/dnorm.R).
Write one file in `sweeps/`. It returns a `sweep_spec()` and is discovered
automatically; nothing is registered anywhere. Files beginning with `_` are
shared helpers, except `_selftest.R`, which is loaded only when requested.
The following sketch shows the fields; replace `...` before running it.

```r
source(file.path(here(), "sweeps", "_normal.R"), local = TRUE)

sweep_spec(
  name      = "nv_dnorm",
  family    = "normal",
  params    = list(standard = list(mean = 0, sd = 1)),
  flags     = list(log = c(FALSE, TRUE)),
  domain    = function(p, f) c(-Inf, Inf),         # valid input domain
  support   = function(p, f) c(-Inf, Inf),         # optional; reporting only
  branch_points = function(p, f, dtype) c(...),    # optional; anvl's own

  value     = function(x, dtype, p, f) as.double(anvl::nv_dnorm(...)),
  ref_value = function(x, p, f) dnorm(x, p$mean, p$sd, log = f$log),
  ref_stable = function(x, p, f) ...,                # optional; see REFERENCES.md
  ref_stable_bound_ulp64 = 8,                        #   its error bound, double ulps
  ref_stable_note = "why base R is weaker here",     #   required with ref_stable
  ref_stable_covers = function(f) isTRUE(f$log),     #   optional; which flags

  grad_wrt  = c("x", "mean", "sd"),
  grad      = function(x, dtype, p, f) list(x = ..., mean = ..., sd = ...),
  ref_grad  = function(x, p, f)       list(x = ..., mean = ..., sd = ...),
  ref_grad_mpfr = function(x, p, f) ...,       # optional MPFR truth
  ref_grad_bound_ulp64 = 16,                  # bound checked by validation
  ref_stable_mpfr = function(x, p, f) ...,     # needed to validate ref_stable

  jax_value = function(x, dtype, p, f) ...,          # optional
  jax_grad  = function(x, dtype, p, f) ...,          # optional
  jax_covers = function(f, kind) TRUE    # where JAX has no twin
)
```

The grid is `params × flags × dtypes × {value, grad} × backends`; an unknown
field name is an error, so a typo in a config that takes hours to run cannot
silently do nothing.

**`grad` returns all declared first derivatives from one call.** A reverse pass
computes them together, so the harness scores all outputs during the same sweep.

Three things worth knowing before you write a reference:

1. **Evaluate references stably and validate them.** Analytic formulas can
   still lose accuracy through cancellation, overflow or underflow. See
   `inv_mills()` in `sweeps/_normal.R` and supply MPFR truths and explicit
   bounds for reference validation.

2. **Declare the domain honestly, and keep it apart from the support.** The
   *domain* is where the function is defined at all (p in [0, 1] for a
   quantile); outside it the value is NaN by specification. The *support* is
   where the distribution lives, and excuses nothing: a CDF below its support
   has a perfectly good value. `branch_points` are read from anvl's source and
   go stale when it changes — keep the pointer to the source beside them.
3. **Prose belongs in the spec.** The derivation of an analytic derivative and
   the reason for a parameter set are the most valuable things in these files.
   Keep them next to the code they justify.

Then check the engine still behaves:

```bash
Rscript run.R selftest
```

`sweeps/_selftest.R` is a synthetic family with errors injected at known places
— a one-ulp nudge on a known interval, a zeroed tail, a 1e-6 gradient error —
and `selftest` asserts the engine finds each at the right magnitude. It exists
because a sweep that silently sweeps nothing looks exactly like a sweep that
found nothing, and at full depth that is an expensive way to discover a
misspelled filter.
