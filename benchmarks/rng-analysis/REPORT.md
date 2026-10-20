# Statistical analysis of anvl's random number generation

**Code analysed:** branch `feat/rng-split`, HEAD `0325747e` plus the uncommitted working tree as of 2026-10-18, with the JAX layout of the random bits. The scripts, C tools and raw outputs this report refers to are not part of the repository; they were written to `devcontainer/scratch/rng-analysis/` of the r-xla workspace, and file names below are relative to it.

The analysis keeps two layers apart throughout:
- **The bit layer** is `prim_sample_bits()`: ThreeFry-2x32-20 laid out as in JAX 0.11.2. Its properties come from ThreeFry and from JAX's layout.
- **The sampler layer** is anvl's own code: `nv_unif_rand()`, through which every sampler goes, and `nv_runif`, `nv_rnorm`, `nv_rbinom`, `nv_sample_int` and `nv_sample`. Almost every finding that matters to R users is here.

## 0. Findings ranked by importance for R / statistics users

### 1. The default float is `f32`, and an `f32` uniform has only 2^23 values
`default_dtypes()` on pjrt is `f32` / `i32`. `nv_unif_rand(, "f32")` is exactly uniform on the grid {k·2^-23}, which has 8,388,608 points. Base R's `runif()` uses a 2^-32 grid (checked); anvl `f64` uses 2^-52.

**Ties (E3).** Duplicated values at 1e6 draws:
- `f32` uniforms: 57,257 (theory 57,305), i.e. 5.7 %; at 1e7 draws, 42 %.
- `f32` normals: 6,923.
- base R: 115.
- anvl `f64`: 0.

**Normal tail (E5).** An `f32` `nv_rnorm()` can never exceed |z| = sqrt(-2 log 2^-23) = 5.6467. Under N(0,1), P(|Z| > 5.6467) = 1.6e-8. The tail below the cut-off is also too thin:

| \|z\| > | `f32` mass relative to N(0,1) |
|---|---|
| 5 | 0.963 |
| 5.25 | 0.877 |
| 5.45 (genome-wide 5e-8) | 0.643 |
| 5.5 | 0.637 |
| 5.65 | 0 |

In 8.6e9 `f32` normals, the observed counts match the exact discretised model. Against N(0,1): p = 6e-5 at |z| > 5.25, and 0 observed vs 138 expected beyond 5.65. The `f64` tails matched N(0,1) in all bins (2.1e9 draws; the `f64` cut-off is 8.49).

**Scaled uniforms collapse (E4).** At `f32`, `nv_runif(n, 1e6, 1e6+1)` has 17 possible values, and it returns `min` 3.1 % and `max` 3.1 % of the time.

**Remedy:** use `dtype = "f64"` or set `anvl.default_dtypes` to `float = f64`. The integer samplers always use `f64` uniforms internally and are not affected.

### 2. The bit layer passed everything
- anvl was bit-identical to an independent C implementation and to a pure-Python copy of JAX's code (E1).
- Batteries (E11):
  - TestU01 SmallCrush: 20/20 bit and `f64` configurations passed. The two `f32`-uniform runs fail, as expected from their 23-bit resolution.
  - TestU01 Crush: 7/7 regular configurations passed.
  - PractRand: no anomalies at any length up to 256 GB in 4 configurations.
  - BigCrush was not finished: 15 of 160 statistics completed, all p in [0.02, 0.92].
- Targeted tests (E8-E10), none showing a defect: correlation tests, strict-avalanche tests on seed, counter, position and key bits, and birthday tests on values and keys. Every p-value below 0.02 was rerun on new data and did not recur.
- The results table in `benchmarks/rng-testu01/README.md` tested an **earlier** layout (a `ui32` value was one hash word). Its text is out of date as well.

### 3. Nested forks form a chain that cycles after about 2^32 levels
In nested forks, k_{d+1} = TF_{k_d}(0, c) iterates a random map on 2^64 keys, so the chain must eventually repeat. Brent cycle detection for seed 42 (c = 1) found a tail of 5.40e9 and a cycle of 8.30e8. From value 6.23e9 on (24.9 GB of `ui32`), the stream repeats.

The batteries see exactly this:
- PractRand on one value per fork level was clean to 16 GB and failed massively at 32 GB (p down to 1e-451).
- Crush failed its two Gap tests on the same layout.

Real `nv_while()` nesting depths are tiny, so this does not matter in practice. But **any "continue with the child" loop** walks such a chain: `prim_fork_random_state()` + `nv_set_random_state()` per iteration, or a future JAX-style `key, sub = split(key)` API. anvl's counter-based global state does not.

### 4. Key collisions: none within a random state, possible across random states
Within one random state, the 2^64 keys are provably distinct. Keys of different random states collide with probability about n²/2^65 for n keys in total:
- 2.7e-4 at 1e8 keys
- 2.7 % at 1e9
- 39 % at 2^32
- 50 % at 5.1e9

A collision means two sampling calls draw identical bits. Statistically this is one duplicated block in about 1e9, but it is avoidable: one seed with many calls never collides, while reseeding per replicate can. JAX has the same property.

### 5. `nv_runif()`'s "open interval (a, b)" holds only for (0, 1)
`U*(max-min)+min` is rounded, so the endpoints can be returned (finding 1, E4). Base R documents this caveat in `?runif`; anvl does not.

### 6. `nv_sample_int()` / `nv_sample()` are almost exactly uniform
I computed the exact distribution (all 2^52 inputs, IEEE rounding included) for n ∈ {2, 3, 6, 10, 100, 1000003, 2^20+1, 2^24+1, 2^30+1, 1717986918, 2^31−1}.
- Every index has probability q/2^52 or (q+1)/2^52, with q = floor(2^52/n).
- Relative deviation is at most n/2^52 ≤ 4.8e-7; total variation ≤ 9.3e-10.
- Detecting it would take at least about 1.6e14 draws, even with an oracle test.

For comparison, R < 3.6 (32-bit `floor(u n)`) had total variation 0.1 at n = 1717986918, reproduced here as P(odd) = 0.399. anvl gives 0.5001; base R ≥ 3.6 (rejection sampling) is exact.

### 7. `nv_unif_rand(, "f64")` shifts by 11, not 12
`bits >> 11` puts a 53rd bit onto the lowest exponent bit. That bit is already 1 in the bit pattern of 1.0, so the OR does nothing and the uniform is still exact on the 2^52 grid. But:
- the top bit of y0 is silently discarded;
- the comment "exponent bits are all 0" is wrong for `f64`;
- the values differ from JAX's convention (`>> 12`).

### 8. Semantics R users must know (verified)
- `set.seed()` reseeds anvl only before its first draw. That first draw consumes two base-R uniforms: `set.seed(1); nv_runif(1); runif(1)` returns R's third uniform.
- Draws are prefix-stable in their size: `nv_runif(5)[1:3] == nv_runif(3)` from the same state. `nv_rnorm()` is not.
- A matrix draw is filled row-major.
- `jit()` reproduces eager draws exactly.
- `mclapply` children segfault at compile time, or deadlock if the sampler was compiled in the parent. Forked parallelism is unavailable. PSOCK workers derive their state from their own base-R RNG, or can be given children made with `prim_fork_random_state()` in the parent.

### 9. Minor
- The `pmin()` in `sample_indices()` is dead code: the product provably cannot round up to n.
- `nv_rbinom()`:
  - costs O(size) and requires `size*n < 2^31`;
  - takes `prob` as a static scalar, so each value compiles a new program;
  - has Bernoulli bias < 2^-52.

## 1. What is implemented (from the code; verified in E1)

### The bit layer
- **Random state:** (k, c), with k = `random_key` (`ui32[2]`, JAX key order) and c = `random_counter` (`ui64`).
  - `nv_set_seed(s)` gives k = (0, s mod 2^32), c = 0.
  - An unset state takes k = `floor(runif(2)*2^32)` at its first draw.
- **Sampling call:**
  - The call key is k' = TF_k(hi32(c), lo32(c)), which is JAX's `fold_in`.
  - The value at row-major position t comes from (y0, y1) = TF_{k'}(0, t):
    - `ui64` = y0·2^32 + y1
    - `ui32` = y0 ⊕ y1
    - `ui16` / `ui8` = the low bits of the XOR
  - The state becomes (k, c+1 mod 2^64).
- **Fork:** consumes counter c and returns the child state (TF_k(c), 0).
- **Under `jit()`:** the j-th sampling call uses counter c+j; scan step i uses c + i·N_body + j; `nv_if` advances by the larger branch count.

### `nv_unif_rand()`
- **`f32`:** U = (b >> 9)/2^23, with b the `ui32` value. The mantissa is bits 9..31 of y0⊕y1; 9 bits are discarded.
- **`f64`:** U = ((b >> 11) mod 2^52)/2^52, with b the `ui64` value. That is y0 bits 0..30 and y1 bits 11..31; y0 bit 31 and y1 bits 0..10 are discarded. Checked on 1e5 values from anvl; the JAX convention would give different values.
- `V − 1` is exact, so the uniforms are bit-identical on every IEEE platform.

### The samplers

| sampler | sampling calls | construction |
|---|---|---|
| `nv_runif` | 1 | 0 → 2^-24 / 2^-53, then `U*(max-min)+min` |
| `nv_rnorm` | 2 | Box-Muller, R = sqrt(-2 log(1−U1)), θ = 2πU2; output is [sin half, cos half]; odd n drops the last value |
| `nv_rbinom` | 1 | `size·n` `f64` uniforms; success is U < p; each sample sums a consecutive group of `size` |
| `nv_sample_int` / `nv_sample` | 1 | `f64` U; index = min(floor(U·n)+1, n) |

There is no exponential or other sampler on this branch.

## 2. Theory of the bit layer
Tags used below:
- **[exact]** follows from the construction alone.
- **[PRF]** assumes ThreeFry-2x32-20 behaves like a random function family, including for the related, low-entropy keys and inputs anvl feeds it (seed keys (0, s); counters and positions 0, 1, 2, …). ThreeFry has no security proof. Salmon et al. (2011) designed it as non-cryptographic and "Crush-resistant": 13 rounds pass BigCrush in their experiments, and 20 is the recommended margin.

### 2.1 Bijections [exact]
**Lemma.** TF_k is a permutation of {0,1}^64 for every k.
**Proof.** Each mix step (x0, x1) → (x0+x1, rotl(x1, d) ⊕ (x0+x1)) inverts as x1 = rotr(b⊕a, d), x0 = a − x1. Each key injection adds constants. A composition of bijections is a bijection.

Consequences:
- **Keys within a state are distinct; the key sequence has period exactly 2^64.** The counter wraps mod 2^64 (wrap checked in E1). A state yields fewer than 2^95 values before repeating.
- **The `ui64` values of one draw are distinct.** A draw is a sample without replacement of 64-bit blocks.
- **Narrower outputs are many-to-one.** `ui32` = y0⊕y1 is 2^32-to-1, and every w-bit projection is 2^(64−w)-to-1. Therefore:
  - If (y0, y1) is uniform, then y0⊕y1 is exactly uniform: Σ_{y0} P(y0, y0⊕v) = 2^32·2^-64. So is any fixed set of w bits, including the 23 and 52 mantissa bits.
  - Within one draw, two positions agree in a w-bit projection with probability (2^(64−w)−1)/(2^64−1) instead of 2^-w. The relative deficit is about 2^-(64−w): 2^-32 for `ui32`, 2^-41 for `f32`, 2^-12 for `f64`.
- **Keys of different states are not related by the bijection** and can collide.
- **Keys and values come from the same function.** A child key at counter c equals the call key a sampling call at c would use, and a `ui64` value equals the key of a derived call. This matters only if a user restores a saved state and then runs different code from it.

### 2.2 Cross-state collisions [PRF + birthday]
For n keys split into states of n_i keys each:

p ≈ 1 − exp(−(n² − Σ n_i²)/2^65) ≤ n²/2^65

| n keys | P(some collision) |
|---|---|
| 2^20 | 3.0e-8 |
| 6.1e6 | 1e-6 |
| 1e7 | 2.7e-6 |
| 1e8 | 2.7e-4 |
| 1.9e8 | 1e-3 |
| 1e9 | 0.027 |
| 2^32 | 0.39 |
| 5.1e9 | 0.50 |
| 1e10 | 0.93 |

A collision gives bit-identical values at every common position. Iterating keys (fork chains) hits the same 2^32 limit through the rho structure of a random map: expected tail ≈ cycle ≈ sqrt(πN/8) = 2.7e9 (finding 3).

### 2.3 PRP vs PRF within a draw
By the PRP/PRF switching lemma, the advantage of distinguishing one draw from iid values is at most n(n−1)/2^65. That is ≤ 1/8 at the draw limit of 2^31−1 values. The bound is essentially attained by "is any `ui64` value repeated?", which an iid sample of that size answers yes with probability 11.75 % and an anvl draw never does.

For w-bit outputs, a collision count gives a z-score after one maximal draw of:

z = sqrt(C(n,2))·2^(w/2−64)

| output | z per maximal draw | maximal draws needed for z = 3 |
|---|---|---|
| `f64` mantissas (w = 52) | 0.0055 | about 3e5 (about 6e14 values) |
| `ui32` (w = 32) | 2^-17.5 | about 1e11 |
| `f32` (w = 23) | 2^-20 | hopeless |

The pair indicators are pairwise independent, so the collision count has variance C(n,2)·p(1−p). The effect is confined to one call, since calls use different keys. Outside raw `ui64` values it cannot be observed in practice.

### 2.4 Guarantees
- **[exact]:**
  - distinct keys within a state;
  - period 2^64;
  - distinct `ui64` values within a draw;
  - identical bits on every platform.
- **[PRF]:** everything about randomness. For a fixed key, a value is a fixed number; its randomness comes only from the key. This is tested in E8–E11.

## 3. Theory of `nv_unif_rand()` and the samplers (assuming ideal bits)

### 3.1 `nv_unif_rand()`

| | `f32` | `f64` | base R `runif` (Mersenne-Twister) |
|---|---|---|---|
| support | {k·2^-23, 0 ≤ k < 2^23} | {k·2^-52} | {k·2^-32, k ≥ 1}, plus about 2^-33 for k = 0 |
| P(U = 0) | 2^-23 | 2^-52 | 0 |
| maximum | 1 − 2^-23 | 1 − 2^-52 | 1 − 2^-32 |
| bits used | 23 of 32 | 52 of 64 | 32 |
| E[U] | 1/2 − 2^-24 | 1/2 − 2^-53 | about 1/2 |
| Var[U] | (1 − 2^-46)/12 | (1 − 2^-104)/12 | about 1/12 |
| KS distance to U(0,1) | 2^-23 | 2^-52 | 2^-32 |

- These are exact discrete uniforms.
- The bias is undetectable: detecting the `f32` mean bias at z = 1 needs 2.3e13 draws.
- What matters is the **resolution**:
  - **Ties:** expected duplicates ≈ n − N(1 − (1 − 1/N)^n) ≈ n²/(2N).
  - **Endpoints.**
  - **Transforms near 0 or 1:** logs, `-log U`, tail quantiles.

### 3.2 `nv_runif`
- On (0, 1) the support is {2^-24} ∪ {k·2^-23, k ≥ 1}, which mirrors base R's `fixup()`. The open interval (0, 1) holds.
- For general (a, b), Y = fl(fl(U·fl(b−a)) + a), so the endpoints can be returned. The "open interval" in the documentation is false in general.

### 3.3 `nv_rnorm`
V = 1 − U1 is exact. R takes the values r_k = sqrt(−2 log(k·2^-M)) with probability 2^-M each, and P(R ≥ r_k) = exp(−r_k²/2). So the radius is the exact Rayleigh distribution at a grid of quantiles, and θ is uniform on 2^M angles.

- **Cut-off** at sqrt(2M ln 2): 5.6467 for `f32`, 8.4904 for `f64`.
- **Coarse tail:** for `f32` only k = 1..31 give radii above 5.

| x | P(\|Z\| > x), N(0,1) | P(\|Z\| > x), `f32` Box-Muller | ratio |
|---|---|---|---|
| 4 | 6.33e-5 | 6.33e-5 | 0.9995 |
| 4.5 | 6.80e-6 | 6.77e-6 | 0.996 |
| 5 | 5.73e-7 | 5.52e-7 | 0.963 |
| 5.25 | 1.52e-7 | 1.33e-7 | 0.877 |
| 5.45 | 5.04e-8 | 3.24e-8 | 0.643 |
| 5.5 | 3.80e-8 | 2.42e-8 | 0.637 |
| 5.6 | 2.14e-8 | 9.8e-9 | 0.456 |
| 5.65 | 1.60e-8 | 0 | 0 |

- The Neave effect (pairs on spirals) needs a linear generator such as an LCG, so it does not apply here.
- Monte Carlo of extremes, rare events or tests at α ≈ 1e-7 must use `f64`.
- Base R's default `rnorm` is inversion with two uniforms combined, so its tails reach about 8–9. This is from memory of R's `snorm.c` and was not re-verified.

### 3.4 `nv_rbinom`
- P(U < p) = ceil(p·2^52)/2^52, so the Bernoulli bias is < 2^-52; p = 0 and p = 1 are exact.
- The result is therefore an exact Binomial up to that.
- Limits are in finding 9.

### 3.5 `nv_sample_int` / `nv_sample`
**Proposition.** Let b_k be the boundaries of fl(J·n/2^52) and e_k = ceil(k·2^52/n) the exact ones. Then b_k ∈ {e_k − 1, e_k}, so |p_k − 1/n| < 2^-51.

**Proof.**
- Rounding is monotone and k is representable, so a boundary can only move down.
- The points x_J = J·n/2^52 are n·2^-52 apart, and ulp(k)/2 ≤ k·2^-53 < n·2^-53. So at most one J can be affected.
- The `pmin()` is never active: (1 − 2^-52)·n is more than one ulp below n.

**Exhaustive computation** (`sampleint.out`): in every n tested, only counts q and q+1 occur, and exactly r = 2^52 mod n cells get q+1. Rounding nevertheless moves up to 17 % of the boundaries.

| n | max relative deviation | total variation |
|---|---|---|
| 3 | 4.4e-16 | 1.5e-16 |
| 1000003 | 1.4e-10 | 5.3e-11 |
| 2^30+1 | 2.4e-7 | 9.3e-10 |
| 1717986918 | 3.8e-7 | 2.3e-10 |
| 2^31−1 | 4.8e-7 | 4.7e-10 |

With 32-bit uniforms (R < 3.6): n = 1717986918 has counts 2 or 3, relative deviation 0.2 and total variation 0.1.

The most powerful test needs N = 9·2^104/(r(n−r)) draws to reach z = 3. That is at least 1.6e14, 4e16 for n = 2^31−1, and about 1e17 for n = 1717986918.

### 3.6 `f32` default
Combining 3.1–3.3: 23-bit uniforms, ties from about 1e4 draws on, normals bounded at 5.65, and coarse grids for intervals away from 0. All arithmetic downstream is also single precision.

## 4. Reproducibility semantics
- **Platform independence:**
  - Bits and uniforms are identical everywhere.
  - `nv_sample_int` and `nv_rbinom` are exact IEEE operations and therefore bit-reproducible; they need `f64` support.
  - `nv_runif` with general bounds (possible FMA fusion) and `nv_rnorm` (transcendentals) may differ in the last bits. This was not tested; only CPU was available.
- **`set.seed`:** see finding 8.
- **Prefix stability:** `nv_runif` is prefix-stable in its size. `nv_rnorm` is not: `nv_rnorm(6)[3:4] != nv_rnorm(4)[3:4]`.
- **Calls:** consecutive calls use different keys [PRF].
- **`jit()` and `nv_while()`:** see E1.
- **Parallel workers:** see finding 8. The parent-side fork pattern (children from `prim_fork_random_state()`) was checked to give distinct streams.

## 5. Experiments
**Tools:**
- `tf.h` and `anvlgen.c` (stream generator). Layouts: percall / streams / seeds / forkchain / counters. Formats: u32 / u64 / u8 / f64 / f32.
- `pairs.c`, `sac.c`, `collide.c`, `sampleint.c`, `keydraw.c`, `rho.c`.
- `jax_ref.py`.
- R drivers: `validate.R`, `samplers.R`, `structure.R`, `followup*.R`, `misc*.R`.
- Battery runners: `battery.sh`, `launch.sh`, `smallcrush_set.sh`, with results in `batteries/`.

Large-scale tests use the C generator, after E1 showed it bit-identical to anvl. Every test of sampler arithmetic used real anvl calls.

### E1. Validation
- **C vs Python reference:** `check_c_vs_ref.py` has 10 checks, all OK. Seed 1 gives `ui32` values 3017709682, 1008607346, 3512701570, exactly as anvl printed them.
- **anvl vs C:** `validate.R` → `validate.out`, **35/35 identical**:
  - `ui32` and `ui64` for seeds 1, 42, −7, 0, ±(2^31−1);
  - `ui8` and `ui16`;
  - row-major fill;
  - a 2^26-value draw;
  - counter 2^32 (high word first);
  - the 2^64 wrap and the repetition after it;
  - fork key, parent advance and child draws;
  - `jit(f)() == f()` for 5 sampling calls;
  - `nv_while` drawing from (fold_in(k, c), j);
  - `nv_runif` `f32`/`f64` exact;
  - `nv_rnorm` `f64` identical to R's double Box-Muller (0 difference), `f32` within 9.5e-7;
  - odd n;
  - `nv_sample_int` for n = 3, 6, 1000003, 2^31−1;
  - `nv_rbinom` grouping;
  - the unset state.

### E2. Uniforms (2^26 draws, `samplers-part1.out`)

| | `f32` | `f64` | base R |
|---|---|---|---|
| on the expected grid | yes | yes | yes (2^-32) |
| min / max | 0 (4 zeros, 8 expected) / 1 − 2^-23 | 9.5e-9 / 1 − 6e-9 | 2.6e-8 / 1 − 6.3e-9 |
| distinct values | 8,385,794 | 67,108,863 | 66,587,807 |
| mean | 0.500052 | 0.499962 | |
| 12·Var | 1.000092 | 0.999916 | |
| χ² on 1024 bins | p = 0.45 | p = 0.18 | |
| KS on 1e6 | p = 0.17 | p = 0.051 | |
| lowest 8 mantissa bits | p = 0.28 | p = 0.55 | |

The standard error of the mean is 3.5e-5. TestU01 on `f32` uniforms fails 12 of 15 SmallCrush statistics at p < 1e-300. TestU01 reads 32 bits per double and the low 9 are always zero, so this is the resolution. The `f64` uniforms pass SmallCrush and Crush.

### E3. Ties: duplicates observed (expected)

| n | anvl `f32` `nv_runif` | anvl `f64` | base R `runif` | anvl `f32` `nv_rnorm` |
|---|---|---|---|---|
| 1e4 | 5 (6.0) | 0 | 0 | 0 |
| 1e5 | 601 (593.7) | 0 | 1 (1.2) | 62 |
| 1e6 | 57,257 (57,305) | 0 | 115 (116.4) | 6,923 |
| 1e7 | 4,157,527 (4,158,049) | 0 (0.011 expected) | 11,719 (11,633) | 646,241 |

### E4. `nv_runif` endpoints (2^24 draws each)
- `f32` on (0, 1), (−1, 1) and (0, 1e-30): no endpoint hits.
- `f32` on (1000, 1001): 539 at `min` and 503 at `max` (512 expected each); 16,385 distinct values.
- `f32` on (1e6, 1e6+1): 524,505 at `min` and 523,934 at `max`; 17 distinct values.
- `f64` on all intervals: no endpoint hits.
- Base R on (1000, 1001): no endpoint hits.

### E5. `nv_rnorm` tails (`samplers-part2.out`)
8.59e9 `f32` draws and 2.15e9 `f64` draws.

| \|z\| > | `f32` observed | N(0,1) expected | p | `f32` model expected | `f64` observed | N(0,1) expected |
|---|---|---|---|---|---|---|
| 4.5 | 57,775 | 58,372 | 0.014 | 58,139 | 14,532 | 14,593 |
| 5 | 4,756 | 4,925 | 0.016 | 4,740 | 1,251 | 1,231 |
| 5.25 | 1,164 | 1,307 | 6.4e-5 | 1,146 | 343 | 327 |
| 5.5 | 198 | 326 | 2.5e-14 | 208 | 79 | 82 |
| 5.6 | 89 | 184 | 9.9e-15 | 84 | 36 | 46 |
| 5.65 | 0 | 138 | 2.8e-60 | 0 | 30 | 35 |
| 6 | 0 | 17 | 8.7e-8 | 0 | 3 | 4 |

- The observed `f32` maximum is 5.646660, the theoretical bound.
- Means and variances agree with N(0,1).

### E6. `nv_sample_int`
- Exact tables: see 3.5.
- With real anvl, 2.68e8 draws each, χ² p-values:

| n | p |
|---|---|
| 2 | 0.021 |
| 3 | 0.78 |
| 6 | 0.069 |
| 7 | 0.19 |
| 1025 | 0.38 |
| 2^31−1 (1000 bins) | 0.26 |

- Parity at n = 1717986918, restricted to draws ≤ 1e8:
  - base R with `sample.kind = "Rounding"`: P(odd) = **0.3993**
  - base R with `"Rejection"`: 0.5005
  - anvl: 0.5001 (1.56e7 draws, z = 0.89)

### E7. `nv_rbinom`
2^24 draws per configuration, χ² against `dbinom`:

| size | prob | p |
|---|---|---|
| 10 | 0.3 | 0.74 |
| 1 | 0.5 | 0.15 |
| 5 | 0.01 | 0.45 |
| 20 | 0.9 | 0.52 |

### E8. Independence (2^30 pairs per scheme)

| scheme (pairs compared) | z(corr) | p Hamming | p 2D | p bits |
|---|---|---|---|---|
| positions t, t+1 of one draw | 0.08 | 0.78 | 0.91 | 0.77 |
| calls c, c+1 | 0.66 | 0.92 | 0.38 | 0.26 |
| seeds s, s+1 | 0.31 | 0.42 | 0.26 | 0.54 |
| seeds 42 / 43 at call c | −1.41 | 0.099 | 0.22 | 0.61 |
| parent vs forked child | 1.02 | 0.84 | 0.18 | 0.41 |
| draw vs sibling fork's child | 0.73 | 0.16 | 0.018 | 0.44 |
| y0 vs y1 of one hash | 0.52 | 0.74 | 0.92 | 0.59 |
| `f64` mantissa vs the next call's `ui32` | −1.23 | 0.015 | 0.33 | 0.50 |

Replicates on disjoint seed ranges:
- sibling-fork 2D: p = 0.58 and 0.94;
- `f64`-vs-`ui32` Hamming: p = 0.60, 0.69 and 0.66.

### E9. Strict avalanche
2^24 base inputs; for each input bit flipped, count flips of each of the 64 output bits.

| input | base inputs | χ² p | max \|z\| (Bonferroni p) |
|---|---|---|---|
| seed | sequential | 0.43 | 3.76 (0.34) |
| seed | random | 0.60 | 3.82 (0.27) |
| counter | sequential | 0.060 | 4.40 (0.044) |
| counter | random | 0.0028 | 4.00 (0.26) |
| position | sequential | 0.22 | 3.43 (1) |
| position | random | 0.051 | 3.37 (1) |
| key | sequential | 0.41 | 3.88 (0.42) |
| key | random | 0.046 | 3.78 (0.63) |

The counter rows were replicated and nothing recurred: χ² p = 0.46, 0.78 and 0.38, and 0.21 with 2^26 inputs.

### E10. Birthday tests
- **`ui32` values,** 1024 blocks of 2^20 (128 colliding pairs expected per block):
  - within draws: 131,195 vs 131,072 (z = 0.34), variance 128.0;
  - one value per sampling call: 131,765 (z = 1.91), variance 124.8.
- **Top-b-bit pairs** among 2^28 values:
  - One `ui64` draw: 0 full collisions, as proved. Truncated counts match the permutation expectation, except a first-run z = −3.4 at b = 28 (78 statistics in total). Four replicates gave +1.7, −1.6, +1.9 and +0.3.
  - 2^28 keys of one state: 0 full collisions; truncated counts all p > 0.11.
  - 2^14 seeds × 2^14 counters, and a 2^26-level fork chain: consistent with iid.
  - `f64` mantissas: 11 colliding pairs vs 8 expected (p = 0.37).
  - `f32` mantissas (2^24 values): 16,778,414 vs 16,777,215 (p = 0.77).

### E11. Batteries
TestU01 1.2.3 and PractRand 0.95. The stream comes from `anvlgen`.

Layouts:
- **percall P** = seed 42's calls with P `ui32` values each.
- **seeds** = call 0 of seeds 0, 1, 2, … Seeds are 32-bit, so this stream repeats every 2^32 values. Crush (about 2^35 words) wraps around about 8 times.
- **streams K** = K consecutive seeds interleaved.
- **forkchain** = one value per level of a nested-fork chain.
- **counters** = calls 0, 2^32, 2·2^32, …
- **`-r`** = bit-reversed words.

**SmallCrush.** 20 of 22 runs passed with nothing outside [0.001, 0.999]:
- percall 1 (seeds 42 and 1), percall 1 `-r`, percall 2 and 3, percall 2^20 (and `-r`);
- `u64`, `u8` (and `-r`);
- seeds from 0 (and `-r`), seeds from −1e9;
- streams 2 and 1024;
- forkchain (and `-r`);
- counters;
- `f64` uniforms (percall and seeds).

The two failures were `f32` uniforms (percall 1 and 2^20), from their resolution.

**Crush** (144 statistics each, about 43 min CPU):

| run | result | min / max p |
|---|---|---|
| percall 1 | pass | 0.020 / 0.995 |
| percall 1, `-r` | pass | 0.0036 / 0.997 |
| percall 2^20 | pass | 0.0016 / 0.997 |
| `u64` | pass, one value outside [0.001, 0.999] | RandomWalk1 R p = 0.9992 |
| `f64` uniforms (`-d`) | pass | 0.028 / 0.998 |
| seeds | pass | 0.0036 / 0.995 |
| streams 2 | pass | 0.024 / 0.977 |
| forkchain | **fail**: Gap r=0 and r=22, p < 1e-300 | — |

**PractRand.** No anomalies at any length (128 MB up to the length given):

| run | length |
|---|---|
| percall 1, seed 42 | 256 GB |
| percall 1, seed 7 | 256 GB (stopped on the way to 1 TB) |
| percall 2^20 | 256 GB |
| streams 2 | 256 GB |

Forkchain was clean to 8 GB, had one "unusual" result at 16 GB, and failed massively at 32 GB.

**The forkchain failures are the chain's cycle.** `rho.c` (Brent cycle detection) on seed 42, c = 1: tail 5,397,388,819, cycle 829,797,033. The stream repeats from value 6.23e9 (24.9 GB), between the clean 16 GB and the failing 32 GB.

**BigCrush** was stopped for time: 15 of 160 statistics in 1 h 45 min on a loaded host, so it would have run all day. The 12 SerialOver/CollisionOver and 3 BirthdaySpacings statistics all had p in [0.02, 0.92] (`bc-percall1.txt`). The seeds-layout BigCrush was stopped too.

### E12. Semantics checks
`misc.out`, `misc2.out`, `mcl.out`, `samplers-part1.out`: the prefix, row-major, bit-63 and fork-pattern checks, and the `set.seed` consumption check, all as stated in Section 4. `mclapply` segfaulted in the child at compile time and deadlocked when the sampler was precompiled in the parent.

## 6. Recommendations

### Documentation
1. **Add a "precision of samples" note** to `?nv_uniform`, `?nv_normal` and the vignette:
   - the default float is `f32`;
   - `f32` uniforms have 2^23 values;
   - `f32` normals stop at |z| = 5.65 and under-represent |z| > 5;
   - use `dtype = "f64"`, or `options(anvl.default_dtypes = c(float = "f64", int = "i32"))` / `ANVL_DEFAULT_DTYPES`;
   - `f64` (2^-52) is finer than base R.
2. **Replace `nv_runif`'s "open interval (a, b)"** with base R's caveat, and mention that `f32` makes the endpoints much more likely.
3. **Key collisions:** add a user-level reading of the birthday formula to `prim_sample_bits` "Properties" and the vignette:
   - no collisions within a state;
   - about n²/2^65 across states;
   - collisions give identical bits;
   - prefer one seed and many calls over reseeding per replicate.
4. **Fork chains:** warn not to repeatedly continue with a forked child (`prim_fork_random_state` + `nv_set_random_state` in a loop). Such a chain cycles after about 2^32 steps, measured here.
5. **Within-draw distinctness:** a draw samples 64-bit blocks without replacement. This is visible only as raw `ui64` values in a single draw of about 2^31 values.
6. **Parallel use:** forked workers do not work; PSOCK workers derive their state from base R; per-worker states can be made with `prim_fork_random_state()` in the parent.
7. **`nv_sample_int`:** document that probabilities are equal to within 2^-51, while base R's sampling is exact.
8. **`benchmarks/rng-testu01/README.md`:** the layout description and the result tables are from the old layout.

### Code (none of these changes the distribution meaningfully)
1. **`f32` Box-Muller:**
   - either build the radius uniform from all 32 bits with fine resolution near 0 and use `log(u)`, which moves the cut-off to about 6.76;
   - or draw `f64` uniforms and round the normals to `f32`, which gives cut-off 8.49 and exact tails.
2. **`nv_unif_rand` `f64`:** shift by 12 (JAX's convention) and fix the comment, or keep 11 and fix only the comment. Changing the shift changes every `f64` stream.
3. **`sample_indices`:** drop the dead `pmin()` or fix its comment. Rejection sampling or Lemire's method would make it exact, but the bias does not justify the complexity.
4. **`nv_rbinom`:** an O(1)-per-sample algorithm and a non-static `prob` would remove the O(size) cost and allow vector probabilities.

## 7. Verified vs assumed

**Verified exactly:**
- anvl = C implementation = Python copy of JAX 0.11.2. The JAX equivalence rests on that Python reference (Random123 known answers and JAX docstring keys); JAX itself was not run.
- The sampler formulas as implemented.
- The bijection lemma and its consequences.
- No full 64-bit collisions within a draw or within a state.
- The `nv_sample_int` exact tables, assuming XLA's `f64` multiply rounds to nearest as IEEE requires (E1 is consistent with this).
- The grids and the `f32` normal cut-off.
- The fork-chain cycle lengths for seed 42.

**Empirical only, no proof:**
- Uniformity and independence of the bit stream across positions, calls, seeds and forks (E8–E11). These tests only rule out defects of the size they can detect.

**Assumed or not verified:**
- That ThreeFry-2x32-20 has no exploitable related-key structure. This rests on Salmon et al.'s design and round-count claims, cited from memory and not re-verified.
- The birthday model across states (consistent with E10).
- Platform independence of the float samplers, and anything about GPU/TPU.
- Base R's `rnorm` resolution (from memory).
- BigCrush, which was incomplete: 15/160 statistics.

## 8. Files

| file | content |
|---|---|
| `tf.h`, `anvlgen.c`, `keydraw.c`, `rho.c` | C implementation, stream generator, helper, cycle detection |
| `jax_ref.py`, `check_c_vs_ref.py` | Python reference and the C-vs-reference check |
| `validate.R` → `validate.out` | anvl vs C (35 checks) |
| `samplers.R` → `samplers-part1.out`, `samplers-part2.out` | E2–E7 with real anvl |
| `pairs.c`, `sac.c`, `collide.c`, `structure.R` → `structure.out` | E8–E10 |
| `followup.sh`, `followup*.R` → `followup*.out` | replicates of borderline results; base R `sample()` bias |
| `sampleint.c` → `sampleint.out` | exact `nv_sample_int` tables |
| `misc.R`, `misc2.R`, `mcl.R` → `misc.out`, `misc2.out`, `mcl.out`; `rho.out` | semantics checks, fork test, cycle lengths |
| `battery.sh`, `launch.sh`, `smallcrush_set.sh`, `batteries/` | TestU01 / PractRand runs and reports |
