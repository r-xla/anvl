# `rng-testu01` — statistical test batteries for anvl's RNG

This harness runs anvl's random stream through the batteries that Salmon et
al. (2011), *Parallel Random Numbers: As Easy as 1, 2, 3* (SC '11), used to
validate ThreeFry and Philox: TestU01's **SmallCrush**, **Crush** and
**BigCrush**. It can also run [PractRand](https://pracrand.sourceforge.net/),
which reads the same bit stream.

anvl draws its bits with `prim_sample_bits()`: ThreeFry-2x32 with 20 rounds
(the generator and round count of the paper's `threefry2x32` and of JAX). The
global random state is a key k, the seed passed to `nv_set_seed()`, and a
counter c, which starts at 0. Every call of `prim_sample_bits()`, a *sampling
call*, splits the random state: it draws from the fresh key
k' = ThreeFry_k(c), hashing it with the counters 0, 1, 2, ..., and advances c
by one. A `ui32` draw takes the low and then the high word of each hash. See
the "Random Numbers" vignette for the details.

So the stream a user sees is made of many keys, each used for the few values
of one sampling call. `gen.R` writes exactly that stream: the values of
consecutive sampling calls, `--per-call` values each.

## Files

| File       | What it does                                                                 |
|------------|------------------------------------------------------------------------------|
| `build.sh` | Downloads and builds TestU01 (and PractRand with `--practrand`) into `_deps/`, compiles the harness `testu01` |
| `testu01.c`| Reads raw 32-bit words (or doubles, `-d`) from stdin and runs one battery    |
| `gen.R`    | Writes anvl's stream of consecutive sampling calls to stdout, from one seed or several interleaved |
| `run.sh`   | Pipes `gen.R` into a battery and files the report under `results/`          |

`_deps/`, the binaries and `results/` are gitignored.

## Quick start

Run everything where R, anvl's dependencies and a pjrt build that this anvl
checkout works with are installed. `gen.R` loads
anvl with `pkgload::load_all()` from the checkout it lives in (set
`ANVL_INSTALLED=1` to use the installed package).

```bash
cd benchmarks/rng-testu01
./build.sh                       # TestU01 + harness, ~20 s
./build.sh --practrand           # also PractRand's RNG_test, ~1 min

./run.sh smallcrush              # seed 42, ~20 s
./run.sh crush --seed 7          # hours, see below
./run.sh bigcrush                # many hours
```

`run.sh` writes `results/<battery>-<options>-<time>.txt` (the battery's
report), `.tsv` (every statistic's p-value) and `.log` (generator throughput,
the number of words the battery consumed, wall time and exit status). The exit
status is 1 if a p-value is outside [1e-10, 1 - 1e-10], TestU01's threshold for
a clear failure.

The pieces also work on their own:

```bash
Rscript gen.R --seed 42 | ./testu01 -n "anvl seed 42" smallcrush
Rscript gen.R --seed 42 --mode unif | ./testu01 -d smallcrush
Rscript gen.R --seed 42 | ./RNG_test stdin32 -tlmax 1GB
```

To run Crush or BigCrush past the end of a session, detach it from the shell,
in the environment that has R and anvl:

```bash
nohup setsid ./run.sh bigcrush --seed 42 </dev/null >/dev/null 2>&1 &
```

and check on it with `cat results/bigcrush-*.log`. The `.txt`
report is written when the battery ends; with `TESTU01_OPTS=-v` it instead
fills in test by test with TestU01's detailed output, which shows how far the
battery has got.

## What to run

### Options

`gen.R` (passed through by `run.sh`):

| Option         | Default      | Meaning                                                                     |
|----------------|--------------|-----------------------------------------------------------------------------|
| `--seed S`     | 42           | Seed for `nv_set_seed()`                                                    |
| `--dtype D`    | `ui32`       | `ui32` or `ui64` draws                                                      |
| `--mode M`     | `bits`       | `bits`: raw words. `unif`: `nv_runif(dtype = "f64")` as doubles (`testu01 -d`) |
| `--per-call P` | 1            | Values drawn per sampling call                                              |
| `--streams K`  | 1            | Interleave the streams seeded S, S + 1, ..., S + K - 1                      |
| `--chunk C`    | 1            | Numbers taken from one stream before moving on to the next                  |
| `--calls M`    | 2^20 / K     | Sampling calls per jitted call (an `nv_scan()`), per stream                 |
| `--n N`        | unlimited    | Stop after N numbers                                                        |

`testu01` options, passed by `run.sh` through `TESTU01_OPTS`: `-r` reverses
the bits of every word, `-v` prints TestU01's detailed output.
`PRACTRAND_MAX` sets PractRand's `-tlmax` (default `1GB`).

### Variants worth running

- **Values per sampling call** (`--per-call 1`, `2`, `4`, ...). With `ui32`
  and one value per call, each key contributes only the low word of its first
  hash; with two, both words of it; with more, consecutive hashes of one key.
- **Several seeds.** One battery run is one sample: rerun with other `--seed`s.
- **Reversed bits** (`TESTU01_OPTS=-r`). Most TestU01 tests turn each word
  into a uniform in [0, 1) and look at its leading bits, so the low bits of a
  word are tested less. Reversing the bits tests them as strongly.
- **Interleaved streams** (`--streams K --chunk 1`). The stream that TestU01 sees
  takes one word from each of K generators seeded with consecutive integers in
  turn. Consecutive seeds are keys that differ in a few low bits.
- **Uniform doubles** (`--mode unif`). Tests `nv_runif()` rather than the bits
  under it. TestU01 reads the leading 32 bits of each double, so this mostly
  checks the conversion to `[0, 1)`.

Not covered yet: deep chains of forks (`nv_while()` loops nested in each
other), which need a way to draw from a random state other than the global one.

## The batteries

The word counts are those the harness reported for SmallCrush, and those of
the TestU01 user guide (L'Ecuyer & Simard, 2007) for Crush and BigCrush. Times
are for this harness on one core of the dev machine for SmallCrush; for Crush
and BigCrush they are rough: TestU01's own computation dominates them, and it
runs concurrently with the generator.

| Battery    | Tests / statistics | 32-bit words consumed   | Time            |
|------------|--------------------|-------------------------|-----------------|
| SmallCrush | 10 / 15            | 2.27e8 (~2^27.8, ~0.9 GB) | ~20 s         |
| Crush      | 96 / 144           | ~2^35 (~140 GB)         | ~1 h            |
| BigCrush   | 106 / 160          | ~2^38 (~1.1 TB)         | ~4-8 h          |
| PractRand  | grows by doubling  | `-tlmax` bytes          | ~12 s per GB    |

- **SmallCrush** is a quick screen: birthday spacings, collisions, gaps,
  poker, coupon collector, max-of-t, weight distribution, binary matrix rank,
  Hamming independence and random walks. Weak generators (e.g. many LCGs)
  already fail it.
- **Crush** adds many more parameterisations of these tests and others
  (serial tests on overlapping tuples, close pairs, Fourier, linear complexity,
  autocorrelation, ...). It separates generators with long-range or linear
  structure (Mersenne Twister fails its linear-complexity tests).
- **BigCrush** runs the tests of Crush on longer sequences and more
  parameterisations; it is the most stringent of the three.
- **PractRand** is a later battery (not in the paper) that keeps testing ever
  longer prefixes of the stream (256 MB, 512 MB, 1 GB, ...), so it is a cheap
  way to look further into one stream than BigCrush does.

anvl's generator throughput (draw plus read-back into R) is about 250 MB/s, so
it feeds Crush in about 10 minutes and BigCrush in about 75 minutes, less than
TestU01 itself needs.

### Reading a report

TestU01 lists only the statistics whose p-value lies outside
[0.001, 0.999]; "All tests were passed" means none did. With 144-160
statistics, a p-value of 1e-4 or so turns up by chance in about one run in
ten: the user guide calls p in [1e-10, 1e-4] *suspicious* and asks for a
rerun, and p outside [1e-10, 1 - 1e-10] a *clear failure*. A suspicious value
that does not recur with another seed is noise; one that recurs, or
a clear failure, is a defect. The same reading applies to PractRand's
"unusual" (noise) and "suspicious" / "FAIL" evaluations.

## How the paper used them

Salmon et al. call a generator **Crush-resistant** if it passes SmallCrush,
Crush and BigCrush, and for each of their counter-based generators report the
fewest rounds at which it is. For ThreeFry-2x32 that is 13 rounds; the
recommended 20 rounds, which anvl uses, leave a safety margin of 7. They also
ran the batteries on more than the plain incrementing-counter stream: keys and
counters varied in unusual patterns (e.g. only some words or bits changing),
and many keys. This harness covers the parts reachable through anvl's API:

- many keys derived from consecutive counters of one key, each used for a
  few values: the default, with `--per-call`;
- many related root keys: `--streams`, consecutive seeds being keys that
  differ in their low bits;
- the low bits of each word: `TESTU01_OPTS=-r`.

The paper's unusual counter patterns would need a C reimplementation of the
hash, which tests the algorithm rather than anvl.

## Results

Results for the stream described above, on the dev container
(`feat/rng-split`). Earlier runs of this harness tested an earlier scheme, in
which every draw continued the counter of one key, and are not repeated here.

On 2026-10-15, every SmallCrush run passed:

| Run                                       | Smallest / largest p-value |
|-------------------------------------------|----------------------------|
| `--per-call 1`                            | 0.024 / 0.98               |
| `--per-call 1 --seed 1`                   | 0.14 / 0.78                |
| `--per-call 1`, `-r`                      | 9.8e-5 / 0.99 (see below)  |
| `--per-call 2`                            | 0.029 / 0.98               |
| `--per-call 2`, `-r`                      | 0.031 / 0.90               |
| `--per-call 4`                            | 0.048 / 0.97               |
| `--per-call 3 --dtype ui64`               | 0.0023 / 0.92              |
| `--per-call 1 --streams 64`               | 0.085 / 0.92               |
| `--per-call 1 --mode unif`                | 0.0076 / 0.85              |

The one value outside [0.001, 0.999], RandomWalk1 M at p = 9.8e-5 with
`--per-call 1` and reversed bits, is in TestU01's suspicious range, so the same
configuration was rerun with the seeds 1, 2, 3, 7 and 100: all five passed every
test, so it was noise.

Crush passed all 144 statistics on the stream of one value per sampling call,
the case in which each key is used for the fewest bits:

| Run                    | Smallest / largest p-value | Time   |
|------------------------|----------------------------|--------|
| `--per-call 1`         | 0.0049 / 0.998             | 2.6 h  |
| `--per-call 1`, `-r`   | 0.019 / 0.996              | 2.1 h  |

BigCrush and PractRand have not been run on this stream yet.
