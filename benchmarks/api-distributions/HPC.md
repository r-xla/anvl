# Distributed sweep execution

This guide describes the harness's requirements and behaviour when work is split
across processes or machines. For image builds, Slurm submission, calibration,
monitoring and release ZIP creation, use the
[anvl-bench cluster guide](https://github.com/r-xla/anvl-bench/blob/main/hpc/README.md).
The [harness README](README.md) describes scoring, result selection and the
export schema; [Reference validation](REFERENCES.md) describes the
MPFR checks.

## Keep value and gradient cells together

Use one sweep invocation per function, including both value and gradient
cells. Each invocation can use `--jobs` to distribute its cells across local
workers. Submit the six functions as separate tasks when using several nodes.
The [example below](#scheduler-independent-example) follows this arrangement.

The harness needs the matching value cell in the same run to classify some
gradient disagreements as undefined-domain conventions. Ordinary cell sharding
can separate the pair. See [Run identity and value/gradient evidence](#run-identity-and-valuegradient-evidence)
for the consequences and requirements for custom task groupings.

## Assigning work

`run.R list --backends anvl,jax` reports the current grid. As checked on 2026-10-01, it
contains 160 anvl cells and 96 JAX cells, producing 512 results. JAX covers only
a subset of variants; derive task counts from the image or checkout you run.

For a sweep, the harness sorts the grid by `cell_id`, applies the filter, then
assigns row position `k` to shard `((k - 1) %% n) + 1`. Shard numbers start at
1. Every worker must use the same specs, backends, filter and shard count.
Changing any of these can change the assignment. More shards than cells leaves
empty tasks. Related cells, including anvl/JAX twins, need not share a task or
node.

Validation uses a different partition: sorted reference units, identified by
reference kind, identity and output. Anvl and JAX results sharing a unit
contribute inputs to one validation. All validation workers must see the same
merged sweep data and use the same code and selection. Keep sweep data fixed
until validation finishes. Validation records can be written concurrently to
that store; they do not need a second merge.

## Requirements across workers

- Use matching harness code, installed ecosystem packages, runtime settings and
  sampling seed. The harness calls installed `anvl`; checkout SHAs alone do not
  establish which code is installed.
- Use the same depth for directly comparable inputs. The f64 low bits use the
  harness's fixed `SWEEP_SEED`; do not replace it with a per-task seed.
- Give every process a writable store and sufficient temporary/cache space.
  Containers and scheduler-specific paths belong in the deployment guide.
- Set `NV_SWEEP_DEVICE` to the device actually used. This is a provenance label,
  not a device selector. Keep publication stores separate by platform and
  intended software version.
- Run `selftest` in a separate store on the target compute environment before a
  long sweep. It checks selected scoring and platform behaviours; it does not
  establish full sweep coverage.

## Run identity and value/gradient evidence

**Current limitation:** every `run` invocation creates a separate `run_id`.
Workers created by one invocation's `--jobs` share that ID, but separate shard
invocations do not. Merging preserves the IDs.

Classifying an out-of-domain gradient disagreement as `undefined_domain`
requires evidence from the matching value cell in the **same run**. If the two
cells land in different shards, merging their stores cannot supply that
same-run evidence: the disagreement remains a failure, with an evidence note.
This affects classification, not the recorded numerical comparisons.

The ordinary row-based shard assignment does not keep these pairs together.
For complete convention classification with the current implementation, use
one invocation per selected group containing both value and gradient cells
(for example, one per function, without a `kind` filter), optionally with
`--jobs`. Any custom grouping must preserve those pairs. A shared-store path
alone does not create a shared run identity.

## Stores, merging and completeness

Each invocation writes separate files by table, run and cell. Workers can use
one shared store or separate stores. `merge` copies table files into the
destination; it does not turn their runs into one run or verify completeness.
Partial stores can be merged for inspection or recovery.

**Process success is not a completeness check.** `run` catches cell errors,
records them and prints `done: N ok, M error`, but does not return a failing
process exit merely because cells failed. `validate-refs` likewise records
failed comparisons without requiring a nonzero process exit. A scheduler's
successful dependency therefore establishes neither successful coverage nor
passing reference validation.

Before publication, run `status` on the merged store with the campaign's
filter and backends. It compares each cell's newest attempt with the declared
grid per platform and depth and tallies errors and reference statuses. Terminal
lists stop at 20 errored attempts, 10 missing cells and 20 results with failed
or unidentified references, with counts of omitted entries. The README shows
[how to read the complete error list](README.md#status--is-the-store-complete).
`export` carries coverage for every declared cell on the represented platforms.
Existing results in an accumulating store can still mask missing work: a cell
a new submission never reached counts as swept from an earlier run at the same
depth. Prefer a fresh store for each publication version/platform; consult the
README's result-selection rules when reusing one.

Validation distinguishes failed comparisons from interrupted jobs. A completed
check can record `failed`; an interrupted invocation may write no new records.
Earlier records remain and can still determine status. Retry an interrupted
validation against unchanged inputs and code. If a reference changes identity,
rerun the affected sweeps before validating: old results record the old
reference identity. [Reference validation](REFERENCES.md) describes truth and
method identity changes.

## Scheduler-independent example

Run these commands from this directory. Choose new, absolute writable paths
for each campaign and measure the workload before choosing a depth and worker
count. This example uses one task per function so value and gradient evidence
share a run ID.

```bash
parts=/scratch/my-sweep/parts
analysis=/scratch/my-sweep/analysis
jobs=4

Rscript run.R list --backends anvl,jax

# Submit one task for each: nv_dnorm nv_pnorm nv_qnorm nv_dunif nv_punif nv_qunif.
spec=nv_dnorm
Rscript run.R run --dry-run --backends anvl,jax --filter "spec=$spec"
Rscript run.R run --store "$parts/$spec" --depth full \
  --backends anvl,jax --filter "spec=$spec" --jobs "$jobs" --quiet
```

After all six tasks finish, inspect their logs and merge the stores:

```bash
for spec in nv_dnorm nv_pnorm nv_qnorm nv_dunif nv_punif nv_qunif; do
  Rscript run.R merge --store "$analysis" --from "$parts/$spec"
done
Rscript run.R status --store "$analysis" --backends anvl,jax
```

Then validate against the fixed merged data. Execute the command once for each
`j` from 1 through `m`; validation shards may run concurrently in this store.

```bash
m=4
j=1
Rscript run.R validate-refs --store "$analysis" --backends anvl,jax \
  --shard "$j" --shards "$m"
```

After all validation tasks finish, inspect the statuses and export:

```bash
Rscript run.R status --store "$analysis" --backends anvl,jax
Rscript run.R export --store "$analysis" --out /scratch/my-sweep/export
```

### Cell sharding

When dividing individual cells is necessary, use `--shard i --shards n`.
Execute each shard with the same backends and filter. This gives smaller tasks
but can separate the value/gradient pairs described above. Merging the results
does not restore their shared run identity.

Sweep processing uses chunks, but that does not bound every job's memory:
retained regions, compilation, worker count and analysis-table reads also
matter. Measure representative value and gradient cells, and measure merge,
validation and export separately. Cluster resource settings and measurements
belong in the deployment guide.
