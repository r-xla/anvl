# Distributed sweep execution

This guide describes the harness's requirements and behaviour when work is split
across processes or machines. For image builds, Slurm submission, calibration,
monitoring and release ZIP creation, use the
[anvl-bench cluster guide](https://github.com/r-xla/anvl-bench/blob/main/hpc/README.md).
The [harness README](README.md) describes scoring, result selection and the
export schema; [Reference validation](REFERENCES.md) describes the
MPFR checks.

There are two ways to divide a sweep:

- **A work queue** (`plan`, `work`, `finalise`, `queue`), for large grids and
  uneven cells. Cells are cut into parts, a pool of workers takes parts until
  none are left, and every result shares one run ID. Use this on a cluster.
- **Whole invocations** (`run`, optionally `--shard`), one cell at a time per
  process. Simpler, and fine on one machine; across several it runs into the
  [run-identity limitation](#run-identity-and-valuegradient-evidence), and a
  task lasts as long as its slowest cell.

## The work queue

Cells differ in cost by orders of magnitude -- a binomial quantile cell takes
about a hundred times a normal one -- so a cell is a poor unit of work: a task
per cell finishes when its slowest cell does. The queue cuts each cell into
*parts*, each a contiguous stretch of the cell's chunks (both signs), and lets
any number of workers take parts until none are left. At full depth a chunk is
one f64 binade or one eighth of an f32 binade, and a cell has 2048 of them.

```bash
q=/scratch/my-sweep/queue
parts=/scratch/my-sweep/parts

# once: cut the grid into parts, sized by what each cell cost before
Rscript run.R plan --queue "$q" --depth full --backends anvl,jax \
  --costs /path/to/last-release.zip --unit-minutes 15

# on every node, as many as you like, all at once
Rscript run.R work --queue "$q" --store "$parts" --hours 23.5

# when the workers have stopped
Rscript run.R finalise --queue "$q" --store "$parts"
Rscript run.R queue --queue "$q"
```

**Exactness.** A cell assembled from parts is identical to the same cell swept
in one piece -- every table, every row. Each chunk draws its f64 low words
from its own seed, so a part needs nothing from the chunks before it, and each
reducer absorbs a later part's state exactly as it would have added those
samples. The selftest holds this to account, with seams cut through a failure
region.

**One run.** The plan names one run ID and every worker writes under it, so a
gradient cell and its value twin are always in the same run, wherever each was
swept. The [run-identity limitation](#run-identity-and-valuegradient-evidence)
does not arise. The run's provenance row is written once per store, by
whichever worker gets there first; each part also keeps its own host and CPU,
and assembling a cell whose parts ran on different CPU models says so.

**Sizing parts.** Costs decide only how finely each cell is cut, never what
is measured: a cell assembled from 3 parts is identical to one assembled from
64. A missing, stale or wrong cost costs efficiency, not correctness.

`plan --costs` reads what cells cost earlier: any mix of stores, export
directories, `summary.parquet` files and release ZIPs, separated by commas
**in order of priority** -- a later source overrides every earlier one for the
cells it measured, whatever the depth. A measurement at another depth than the
plan's is scaled by sample count. Each cell is cut into parts of about
`--unit-minutes` (default 15), at most `--max-parts` (default 64).

You do not need a previous campaign. A smoke sweep is 1/8192 of a full one, so
measuring the whole grid at smoke depth is cheap, and its store makes a
complete cost source:

```bash
Rscript run.R plan --queue "$calib" --depth smoke --backends anvl,jax --max-parts 1
Rscript run.R work --queue "$calib" --store "$calib_store"    # on a few workers
Rscript run.R plan --queue "$q" --depth full --backends anvl,jax --costs "$calib_store"
```

Scaled up, a smoke time overstates the cost -- compiling is a larger share of
a short sweep -- and an overstated cell is merely cut finer. The same serves
the other cases:

- **A new function** is unmeasured in any earlier campaign. Measure just its
  cells at smoke depth and list that store after the others.
- **A function whose code changed** has a stale cost in an earlier campaign,
  which the plan cannot detect: costs are matched by cell, not by code. Measure
  it again at smoke depth and list that store last, so it overrides the old
  measurement.
- **A cell no source measured** is cut into `--max-parts` and queued first.
  Workers do not know what its first part will cost, and assume twice the
  planned part size; after that they use the rate its finished parts measured.

A stale cost that understates a cell makes its parts long. A worker may then
be killed at its time limit mid-part. The part is not lost: its claim goes
stale and is swept again (see *Failures*). `plan --dry-run` prints the plan --
how many cells each source measured, how many parts, which cells nothing
measured -- without writing it.

**Workers.** A worker keeps taking parts: the next part of the cell it last
swept while any is left (its functions are already compiled), otherwise the
first open part in the plan, biggest first. With `--until <epoch seconds>` or
`--hours <h>` it takes only parts it expects to finish in time -- from what
the cell's finished parts cost per chunk, or the plan's estimate -- and stops
when none fits, rather than being killed mid-part. The worker that completes a
cell's last part assembles the cell into its store. Workers share the queue
directory and, normally, one store; every file is written once under a unique
name.

**Memory.** A worker's memory creeps up over its life -- a few MB a cell,
from about 0.6 GB after its first part, measured on a cluster in 2026-10 --
and no cause has been found: anvl's and JAX's compilation caches and glibc's
allocator arenas were each ruled out. So `work --max-memory <MB>` bounds it
instead: past that resident size, checked after each part, the worker stops at
the part boundary and exits with status **3**, and whatever started it starts
a fresh process, which begins clean. Nothing is lost but a few seconds of
start-up. Size the memory limit as the threshold plus a part's transient peak
(about 0.4 GB) plus what the scheduler counts beyond the worker itself.

**Failures.** A claim carries a heartbeat: a background loop touches it every
minute for as long as its worker's process is alive, however long one chunk
takes. A claim silent for longer than `--stale-minutes` (default 10) belongs
to a dead worker -- killed for running out of memory, say -- and the next
worker takes the part over. A heartbeat that only ticked between chunks once
made live workers on a slow cell look dead, and four of them swept one part
side by side; that cannot happen now. Keep the threshold a few minutes above
the one-minute beat, since node clocks can differ.

When no worker is running -- say, every one of them hit its time limit --
`queue --release` drops the claims of unfinished parts at once, so that new
workers start on them immediately. A part that
errors is recorded, and its cell is recorded as errored, as `run` would. To
retry, delete the part's files from `done/` and its cell's directory from
`final/`.

**Requirements.** The queue is coordinated by `mkdir` alone, which is atomic
on the shared filesystems clusters use for scratch (Lustre, GPFS, NFS). Every
worker must run the harness the plan was made with: `work` refuses a plan made
with a different anvl, chunk size or sampling scheme, or one whose cells its
grid lacks. `finalise` assembles any cell whose parts are all saved but which
no worker finished assembling; it is safe to repeat, but run it only once the
workers have stopped.

The queue directory holds:

```text
plan.rds               the cells, their parts, the run ID and depth
claims/<part>.<n>/     attempt n at a part; its mtime is the heartbeat, its
                       owner file the worker (host, pid, Slurm task)
done/<part>.rds        the part's reducer state, or its error
done/<part>~<s>~<e>    an empty marker: its cost in seconds, whether it erred
final/<cell key>/done  the cell is in the store
```

A part's state is small -- kilobytes to a few hundred, mostly per-binade
tallies -- so even tens of thousands of parts fit comfortably on scratch.
Delete the queue once its cells are merged and checked.

## Keep value and gradient cells together

Outside the queue, use one sweep invocation per function, including both value
and gradient cells. Each invocation can use `--jobs` to distribute its cells
across local workers. Submit the six functions as separate tasks when using
several nodes. The [example below](#scheduler-independent-example) follows
this arrangement.

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
- Use the same depth for directly comparable inputs. The f64 low bits are
  drawn from seeds fixed by the harness's `SWEEP_SEED` and each chunk's number
  and sign; do not replace them with a per-task seed. Runs record the scheme
  as `sweep_sampling`: results under another scheme sample other f64 inputs.
- Give every process a writable store and sufficient temporary/cache space.
  Containers and scheduler-specific paths belong in the deployment guide.
- Set `NV_SWEEP_DEVICE` to the device actually used. This is a provenance label,
  not a device selector. Keep publication stores separate by platform and
  intended software version.
- Run `selftest` in a separate store on the target compute environment before a
  long sweep. It checks selected scoring and platform behaviours; it does not
  establish full sweep coverage.

## Run identity and value/gradient evidence

**Current limitation, outside the work queue:** every `run` invocation creates
a separate `run_id`.
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
