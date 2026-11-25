# Random numbers inside compiled programs

Script: `random_numbers.R` (run with `Rscript random_numbers.R`; anvl from source via
`pkgload::load_all()`). Ran on 2026-10-22, macOS arm64, CPU only. Versions: torch 0.17.0,
tensorflow (R) 2.20.0 / TF 2.15, greta 0.5.1, quickr 0.3.0.9000, ast2ast 1.0.2, odin2 0.3.46 /
dust2 0.3.28, cmdstanr 0.9.0 / CmdStan 2.39.0, TMB 1.9.19, RTMB 1.8, nimble 1.3.0. The script
ran to completion (exit 0).

What was checked per package: (1) a draw inside the compiled program, and whether it is redone
on every call rather than frozen at compile or tape time; (2) which distributions; (3) generator
design and seeding; (4) GPU (from the docs only); (5) independent streams.

## Suggested table row "Random numbers"

| Package | Cell |
|---|---|
| anvl | yes; explicit functional state (XLA Threefry); runif/rnorm/rbinom/sample; GPU |
| torch | yes; global per-device generator (+ Generator objects); many distributions; GPU |
| tensorflow | yes; stateless seeds or stateful `Generator`, `split()` for streams; GPU |
| greta | yes, via `calculate(nsim=)` / `simulate()`; seed from R's RNG; GPU |
| quickr | `runif()` only; R's RNG |
| ast2ast | no |
| odin2/dust2 | yes; ~16 distributions; own RNG, one stream per particle |
| Stan | `_rng` in generated quantities / transformed data only; per-chain streams |
| TMB | in `SIMULATE` blocks only; R's RNG |
| RTMB | frozen in the tape; simulation runs as interpreted R |
| nimble | yes; R's RNG |

If the table needs one word: anvl, torch, tensorflow, greta, odin2, nimble: yes; Stan, TMB:
restricted (only in some blocks); quickr: uniform only; RTMB: no (frozen at tape time); ast2ast:
no.

## Results per package

**anvl.** `jit(function(state) nv_rnorm(state, shape = 3L))` returns `list(state, values)`.
The same state gives the same draws, and the returned state gives new draws without recompiling.
`nv_rng_state()` also works on a traced seed inside `jit()`. Distributions: `nv_runif`,
`nv_rnorm`, `nv_rbinom`, `nv_sample_int`, `nv_sample` (the last two sample uniformly with
replacement). There is no global state. The program lowers to
`stablehlo.rng_bit_generator` (THREE_FRY), so the XLA CUDA backend should support it (from the
design; not run here).
*Streams:* there is no `split()` / `fold_in()`, so the only way to get a stream per chain or
device is a separate seed for each one.
*Real limitation found (bug):* `nv_rng_state(seed)` bitcasts the i32 seed into two ui16 values
and widens them to `ui64[2] = (low 16 bits, high 16 bits)`. XLA's THREE_FRY reads the state as
(key, counter). So:
- there are only 65536 distinct keys;
- a seed's high 16 bits become the *starting counter*, so seeds that share their low 16 bits are
  the same stream at different offsets. Verified: `runif` with seed 65536 returns exactly
  elements 3..8 of `runif` with seed 0, because `state(0)` after 4 f32 draws is `(0, 2)`.
  `nv_rng_state(1L)` = `(1, 0)` and `nv_rng_state(65536L)` = `(0, 1)`.

Seeds 0..65535, e.g. `1:n_chains`, give distinct keys and therefore non-overlapping Threefry
streams. Large or random seeds can collide. Possible fix: put the whole seed in the key word
(`(as ui64(seed), 0)`) and add an `nv_rng_split()`.

**torch.** `jit_trace(function(x) x + torch_randn(3), ...)` records the random op, so two calls
give different draws. `torch_manual_seed()` makes the draws reproducible, and the TorchScript
path (`jit_compile`) also redraws. The design is a global stateful generator per device, plus
`torch_generator()` objects that `torch_normal(..., generator =)` and similar accept (passing
`generator =` to `torch_randn()` failed in R: "Not compatible with requested type", which is
probably an R-binding gap). There is no split or jump, only separate generators with separate
seeds. GPU: yes.

**tensorflow.** Both designs work inside `tf_function(jit_compile = TRUE)`:
- `tf$random$stateless_normal(shape, seed)` is a pure function of a seed tensor: the same seed
  gives the same draws and a new seed gives new draws.
- `tf$random$Generator$from_seed(1)` is a stateful variable that advances on every call; a
  second generator from the same seed reproduces the draws.
- `Generator$split(n)` gives independent generators, and `stateless_split()` derives new seeds.

GPU: yes.

**greta.** `calculate(y, nsim = 2)` draws again on every call. `seed =` and also `set.seed()`
make the draws reproducible, because greta's `get_seed()` is `sample.int(2^30, 1)`, so the TF
seed comes from R's RNG. All greta distributions can be simulated. There are no explicit
streams. GPU via `compute_options = gpu_only()`; greta's own message warns that the seed "may
not always be respected" on GPU.

**quickr.** Only `runif()` (including `min` / `max`); `rnorm()` fails with "Unsupported function:
rnorm". Each call draws again. Under `set.seed()` the draws are bit-identical to R's `runif()`,
so quickr uses R's global RNG. No streams, no GPU.

**ast2ast.** No random functions in its function registry; `runif()` fails with "Invalid
function runif". (`pso()` uses R's `unif_rand()` internally, but that is not exposed to user
code.)

**odin2/dust2.** `update(x) <- x + Normal(0, 1)` draws again on each step and each run, and the
RNG state advances. `seed =` in `dust_system_create()` reproduces the draws; `seed = NULL` takes
the seed from R's RNG. Each particle has its own xoshiro256 stream (the RNG state is 32 bytes per
particle), and the streams are separated by jumps. The first particles' streams stay the same
when `n_particles` grows (verified: the first 3 of 5 particles match the 3-particle run). 16
distributions: Normal, TruncatedNormal, Uniform, Exponential, Gamma, Beta, Binomial,
BetaBinomial, Poisson, NegativeBinomial, Hypergeometric, Cauchy, LogNormal, Weibull, ZIPoisson,
ZINegativeBinomial. GPU: no; dust2 dropped dust1's GPU support (dust2 "migrating" vignette:
"we no longer support any GPU models", with a rewrite planned).

**Stan.**
- `normal_rng` in the model block is rejected at compile time: "Random number generators are
  only allowed in transformed data block, generated quantities block or user-defined functions
  with names ending in _rng."
- In generated quantities the draw is redone every iteration. In transformed data it is drawn
  once per run (verified: the same value in all 3 iterations).
- `seed =` reproduces the draws, and the two chains differ (per-chain streams derived from seed
  and chain id).
- There is an `_rng` for almost every distribution.
- GPU: no; OpenCL speeds up only selected likelihood functions.

**TMB.** `rnorm` / `rpois` inside `SIMULATE {}` draw again on each `obj$simulate()` call. Under
`set.seed()` the draws are bit-identical to R's `rnorm()`, so TMB uses R's global RNG.
`SIMULATE` runs the template with plain `double`, outside the AD tape. Distributions: the R r*
set (`rnorm`, `rpois`, `rbinom`, `rgamma`, `rnbinom`, ...) plus `$simulate()` on the density
classes (MVNORM, GMRF, AR1, ...). No streams, no GPU.

**RTMB.**
- *Real limitation, verified:* in `MakeTape(function(x) x + rnorm(3), numeric(3))` the tape
  returns the same draws on every call, because `rnorm()` runs once while the tape is recorded
  and its values become constants.
- `obj$simulate()` and `obj$report()` rerun the user's R function (`func(p)`) in interpreted R,
  not the tape. So simulation (`y <- OBS(y); y %~% dnorm(...)`) works and follows `set.seed()`,
  but it does not run inside a compiled program.
- No streams, no GPU.

**nimble.** `rnorm()` in compiled `run` code draws again on each call. Under `set.seed()` the
draws are bit-identical to R's `rnorm()`, so nimble uses R's global RNG through Rmath. It has the
full r* set plus multivariate samplers (`rmnorm_chol`, `rwish`, `rdirch`, `rmulti`, `rcat`, ...).
There are no independent streams: one global R stream, and MCMC chains run one after another.
No GPU.

## Test-design notes (not limitations)

- torch `torch_randn(3, generator = g)` failed in R, but `torch_normal(0, 1, size = 3, generator
  = g)` works, so Generator objects are usable.
- The first TMB attempt with `rpois(vector<Type>(rep(...)))` did not compile; that was my code.
  A scalar `rpois(Type(3))` works.
- RTMB `%~%` simulation needs `OBS()` on the data; without it `simulate()` returns `NULL`.
- A draw in anvl or stateless TF is "frozen" in the sense that the same state gives the same
  numbers. That is the design (a pure function of the state), not a freeze at compile time: a
  new state gives new numbers from the same compiled executable.

## Uncertain points

- GPU cells come from the docs or the design only (the machine is CPU only); anvl on CUDA was not
  run here.
- Stan's per-chain mechanism (an offset of the Boost generator by chain id) is from memory of the
  Stan docs; the script verifies only that chains differ and that the seed reproduces them.
- Whether `nv_rng_state()`'s (key, counter) split is intended. The overlap is verified
  empirically, but the reading that XLA uses state[0] as key and state[1] as counter is inferred
  from that behaviour.
- greta: the distribution coverage of `calculate(nsim=)` was checked only for normal and poisson.
- The torch "many distributions" cell includes the `distr_*` classes, which are plain R calls
  around torch ops and are traceable.

## Summary

- Draws inside the compiled program, redone per call: anvl, torch, tensorflow, greta, odin2,
  nimble, quickr (`runif` only); Stan only in generated quantities / transformed data (rejected
  in the model block); TMB only in `SIMULATE` blocks.
- RTMB freezes `rnorm()` into the tape (verified); its `simulate()` reruns interpreted R.
- ast2ast has no RNG functions.
- Global R RNG (`set.seed()`, bit-identical to base R): quickr, TMB, nimble, RTMB; greta seeds
  TF from R's RNG.
- Own generators: torch (global stateful per device), Stan (per chain), dust2 (per particle),
  TF (stateless seeds or a stateful Generator).
- Only anvl and TF's stateless ops use explicit functional state.
- Independent streams: dust2 (per particle, jump-separated), Stan (per chain), TF
  (`split`/`stateless_split`). anvl has no split; its streams come from separate seeds only.
- GPU (docs): anvl, torch, tensorflow, greta.
- anvl bug: `nv_rng_state()` puts the seed's high 16 bits into the Threefry counter, so seeds
  k and k + 65536 overlap (seed 65536 = seed 0 shifted by 2 blocks) and only 65536 keys exist.
  Worth fixing (whole seed into the key) and adding `nv_rng_split()` before the paper claims
  parallel streams.
