# The dependence a keyed bijection introduces into anvl's random values

A mathematical analysis of the layout on branch `feat/rng-split` (JAX's layout: a fresh key per sampling call, `ui32` = xor of the hash's two words). Formulas were checked against exact small cases; the scripts were written to a scratch directory and are not part of the repository. The statistical/empirical analysis is in `REPORT.md` next to this file.

## 0. Notation and summary

$N=2^{64}$ is the number of 64-bit blocks. A value takes one of $M=2^w$ outputs, and $K=N/M=2^{64-w}$ is the size of each output's preimage (its "fibre"). $(x)_m=x(x-1)\cdots(x-m+1)$ is the falling factorial. A draw has $n$ positions, $Y_i$ is the block at position $i$, and $V_i=f(Y_i)$. $Q$ is the law of $(V_1,\dots,V_n)$ under the model; $P$ is i.i.d. uniform on $[M]^n$. Write $\delta=(M-1)/(N-1)$, $\beta_N(n)=1-(N)_n/N^n$, and $C=\sum_{i<j}\mathbf 1\{V_i=V_j\}$ for the number of coinciding pairs.

Key results:

1. **Law of a draw.** Under the model, the blocks of a draw are a uniform sample without replacement from $N$. Any balanced $w$-bit projection behaves exactly like drawing from an urn with $M$ colours and $K$ balls of each colour.
2. **Pairs.** Any function at one position and the same function at another have correlation exactly $-1/(2^{64}-1)$, whatever $w$. Discarding bits does **not** weaken the dependence between two positions. The pairwise total variation distance is $(1-1/M)/(N-1)\approx2^{-64}$ for every $w$.
3. **Coincidences.** The relative deficit in the coincidence probability is exactly $\delta=(2^w-1)/(2^{64}-1)\approx2^{w-64}$: $2^{-12}$ for f64 uniforms, $2^{-32}$ for ui32 values, $2^{-41}$ for f32 uniforms.
4. **What discarding bits does.** Discarding bits never increases the joint dependence (data processing). It lowers it materially only once $n\gtrsim2^{w/2}$: the total variation distance falls from about $n^2/2^{65}$ to about $n\,2^{w/2}/(2\sqrt\pi\,2^{64})$. The expected shortfall of coinciding pairs is about $n^2/2^{65}$ for **every** $w$. Fewer bits only add noise ($\mathrm{sd}\approx n/2^{(w+1)/2}$) that hides it.
5. **XOR versus keeping 32 bits.** Under the model, XOR and simply keeping 32 bits give exactly the same law.
6. **Statistics.** Monte Carlo means are exactly unbiased, with variance multiplied by $(2^{64}-n)/(2^{64}-1)$. That correction uses $N=2^{64}$, not $2^w$. i.i.d.-based standard errors are conservative, by at most a relative $5.8\cdot10^{-11}$.
7. **Detection.** Counting coinciding f64 uniforms needs about $6.33\cdot10^{14}$ values in draws of the largest size to see the deficit at $z=3$.

## 1. Model and assumptions

- **(F) Facts, no assumption needed.** ThreeFry-2x32-20 with a fixed key $k$ is a permutation $P_k$ of $\{0,1\}^{64}$. Each step is invertible:
  - adding a key word mod $2^{32}$ is undone by subtracting it;
  - the mix $(x_0,x_1)\mapsto(x_0+x_1,\ (x_0+x_1)\oplus\mathrm{rotl}(x_1,d))$ is undone by $x_1=\mathrm{rotr}(x_1'\oplus x_0',d)$, then $x_0=x_0'-x_1$.

  Consequences: the blocks of one draw are distinct; keys derived from distinct counters under one key are distinct.
- **(A1) PRP / ideal-permutation model.** For a call's key $k'$, $P_{k'}$ is a uniformly random permutation of $[N]$.
- **(A2) Ideal-cipher model.** $\{P_\kappa\}_\kappa$ are independent uniform permutations, and the permutation used to derive keys is independent of the others.

(A1) and (A2) are heuristics, not theorems about ThreeFry. Seeds are 32-bit integers (or 53 bits taken from base R), so anyone who knows the algorithm can enumerate them and distinguish the stream with certainty. ThreeFry is derived from Threefish, but Salmon et al. make no cryptographic claim for it; its support is empirical (it passes TestU01's batteries). Every distributional statement below is "exact under (A1)" or "under (A2)", and section 8 sorts them.

## 2. Exact law of a draw

**Theorem 1.** Let $\pi$ be uniform on the permutations of $[N]$ and let $x_1,\dots,x_n$ be distinct. Then $(\pi(x_1),\dots,\pi(x_n))$ is uniform on the $(N)_n$ ordered tuples of distinct elements.

*Proof.* For any distinct $y_1,\dots,y_n$, exactly $(N-n)!$ permutations satisfy $\pi(x_i)=y_i$ for all $i$. So the probability is $(N-n)!/N!=1/(N)_n$. ∎

Inputs $(0,t)$ for $t<n$ are distinct, so a draw's blocks follow this law under (A1). By exchangeability, any $r$ positions are a uniform sample without replacement of size $r$.

**Lemma 2 (balanced maps).** A map $f:[N]\to[M]$ is balanced if every output has exactly $N/M$ preimages.

- (a) $y_0\oplus y_1$ is balanced: for fixed $z$, $y_0\mapsto(y_0,y_0\oplus z)$ is a bijection onto the fibre, which has $2^{32}$ elements.
- (b) Any fixed set $S$ of $w$ bit positions is balanced: the fibre consists of the free choices of the other $64-w$ bits.
- (c) A composition of balanced maps is balanced, with fibre size $(M/L)(N/M)=N/L$.

Hence the following are all balanced:

| Value | Construction | $w$ |
|---|---|---|
| ui8 / ui16 | low bits of the XOR | 8 / 16 |
| f32 uniform | top 23 bits of the XOR (`ui32 >> 9`) | 23 |
| f64 uniform | bits 11–62 of $y_0 2^{32}+y_1$ (= $y_0$ bits 0–30 and $y_1$ bits 11–31) | 52 |

The two float constructions were confirmed bit-exact: $(u\gg 11)\,|\,\texttt{0x3FF0}\ldots$ minus 1 equals $((u\gg11)\bmod 2^{52})/2^{52}$, so bit 63 is absorbed by the exponent; and $(v\gg9)\,|\,\texttt{0x3F80}\ldots$ minus 1 equals $(v\gg9)/2^{23}$.

`nv_runif`'s replacement of 0 by $2^{-24}$ or $2^{-53}$ is a relabelling and stays balanced. A later affine rounding may merge values, and then Theorem 3's general form applies.

**Theorem 3 (urn law).** Let $f$ have fibre sizes $K_c$. Then

$$Q(a_1,\dots,a_n)=\frac{\prod_c (K_c)_{m_c}}{(N)_n},\qquad m_c=\#\{i:a_i=c\}.$$

In the balanced case $K_c=K$, this is the multivariate hypergeometric sequence: an urn with $M$ colours and $K$ balls of each colour, drawn without replacement.

*Proof.* Count the distinct tuples with $f(y_i)=a_i$. The positions of colour $c$ take distinct elements of that colour's fibre in order, which gives $(K_c)_{m_c}$ ways. Elements of different colours are automatically distinct. Divide by $(N)_n$ (Theorem 1). ∎

**Corollary (balanced case).**

- Each value is exactly uniform: $P(V_i=a)=1/M$.
- Same output at two positions: $P(V_i=V_j=a)=\dfrac{K(K-1)}{N(N-1)}=\dfrac{1-\delta}{M^2}$.
- Different outputs: $P(V_i=a,V_j=b)=\dfrac{K^2}{N(N-1)}=\dfrac{1}{M^2}\Big(1+\dfrac1{N-1}\Big)$ for $a\ne b$.
- Coincidence: $P(V_i=V_j)=\dfrac{K-1}{N-1}=\dfrac{2^{64-w}-1}{2^{64}-1}=\dfrac{1-\delta}{M}$.

**Theorem 4 (covariance of any functions).** For any $g_1,g_2:[N]\to\mathbb R$ and $i\neq j$,

$$\operatorname{Cov}(g_1(Y_i),g_2(Y_j))=-\frac{\operatorname{Cov}_{\rm pop}(g_1,g_2)}{N-1},$$

where $\operatorname{Cov}_{\rm pop}$ is the covariance under the uniform distribution on $[N]$.

*Proof.* $E[g_1(Y_i)g_2(Y_j)]=\frac{1}{N(N-1)}\big[(\sum g_1)(\sum g_2)-\sum g_1g_2\big]$. Substitute $\sum g_1 g_2=N(\operatorname{Cov}_{\rm pop}+\mu_1\mu_2)$ and subtract $\mu_1\mu_2$. ∎

Consequences, for any $w$ and any balanced $f$:

- Any non-constant function at one position and the same function at another have correlation exactly $-1/(N-1)=-5.42\cdot10^{-20}$.
- Indicators of the same output: covariance $-(M-1)/(M^2(N-1))$, correlation $-1/(N-1)$.
- Indicators of different outputs $a\ne b$: covariance $+1/(M^2(N-1))$, correlation $+1/((M-1)(N-1))$.
- Integer values: covariance $-(M^2-1)/(12(N-1))$.
- Uniforms $U=V/M$: covariance $-(1-M^{-2})/(12(N-1))$.

**Proposition 5 (collision counts).**

- Means: $E_QC=\binom n2\frac{1-\delta}{M}$ and $E_PC=\binom n2\frac1M$. The shortfall is $\binom n2\frac{M-1}{M(N-1)}\approx n^2/2^{65}$, essentially the same for every $w$: 0.125 for $n=2^{31}-1$.
- Variance under the model: $\operatorname{Var}_QC=\binom n2p_2(1-p_2)+6\binom n3(p_3-p_2^2)+6\binom n4(p_{22}-p_2^2)$, with
  - $p_2=\frac{K-1}{N-1}$,
  - $p_3=\frac{(K-1)(K-2)}{(N-1)(N-2)}$,
  - $p_{22}=\frac{M(K)_4+M(M-1)(K)_2^2}{(N)_4}$.
- Variance under i.i.d.: $\operatorname{Var}_PC=\binom n2\frac{M-1}{M^2}$.
- No coincidence at all: $Q(C=0)=\frac{(M)_nK^n}{(N)_n}=P(C=0)\cdot\frac{N^n}{(N)_n}$.
- Exact law: $Q(C=j)=[x^nz^j]\big(\sum_m\binom Km z^{\binom m2}x^m\big)^M\big/\binom Nn$.
- Distinct values $D$: $E_QD=M\big[1-\binom{N-K}{n}/\binom Nn\big]$, which is at least $E_PD=M[1-(1-1/M)^n]$.

All of Theorem 3, the corollary, Theorem 4 and Proposition 5 were checked by exhaustive enumeration on toy sizes.

## 3. Departure from i.i.d.

**Theorem 6.**

- (a) *Data processing.* Applying any further function (dropping more bits) cannot increase total variation, KL divergence, or any test's advantage.
- (b) *Bijective $f$ (ui64), exact.* $\mathrm{TV}=\beta_N(n)$, with $1-e^{-n(n-1)/2N}\le\beta_N(n)\le n(n-1)/2N$. *Proof:* $Q$ is $P$ conditioned on the event $A$ that all blocks are distinct, and $\mathrm{TV}(P,P(\cdot\mid A))=P(A^c)$ (Freedman 1977, JASA).
- (c) *Any balanced $f$.* $\mathrm{TV}\le\min\Big\{\beta_N(n),\ \tfrac12\sqrt{\tfrac{(M-1)n(n-1)}{(N-1)(N-n+1)}}\Big\}$.
  - *Proof.* Use the chain rule for KL and bound each term by $\chi^2$. At step $s$, given the counts $m_c$ so far, the next value has law $q_c=(K-m_c)/(N-s)$, and $\chi^2(q\Vert u)=(M\sum m_c^2-s^2)/(N-s)^2$. Since $\sum m_c^2=s+2C_s$, its expectation is $s(M-1)/((N-1)(N-s))$. Summing over $s$ and applying Pinsker gives the bound.
  - The classical colour bound of Diaconis and Freedman (1980) is linear in $Mn/N$, weaker by a factor of order $\sqrt M$. Stam (1978) has the matching KL asymptotics.
- (d) *Lower bound.* $\mathrm{TV}\ge Q(C=0)-P(C=0)\ge(1-\lambda)\frac{n(n-1)}{2N}$, with $\lambda=\binom n2/M$. So for $n\ll2^{w/2}$, $\mathrm{TV}=\frac{n(n-1)}{2N}(1+O(\lambda))$, rigorously and exactly as for w=64.
- (e) *Asymptotics (validated, not proved in full).* Linearising $\log(Q/P)=-(C-E_PC)/K+\dots$ gives $\mathrm{TV}\approx\frac{M}{2N}E_P|C-E_PC|$. When collisions are frequent this becomes:
  - for large $M$: $\mathrm{TV}\approx\sqrt{n(n-1)(M-1)/\pi}\big/(2N)\approx n\,2^{w/2}/(2\sqrt\pi\,2^{64})$;
  - for $n\gg M$ (count-based form): $\mathrm{TV}\approx\frac{n}{N}\frac{(d/2)^{d/2}e^{-d/2}}{\Gamma(d/2)}$ with $d=M-1$.

  Exact sums over all count vectors at $N=2^{20}$–$2^{30}$ with $M=2,4,8$ agree: the ratio of exact to normal approximation is 0.858 (predicted 0.858) and 0.947 (predicted 0.946). The bound in (c) exceeds the approximation by a factor of about $\sqrt\pi$.

**Table: total variation of a draw from i.i.d.** Each cell is the approximation, with the rigorous upper bound in brackets. The w=64 row is exact.

| w | n = 10³ | 10⁶ | 10⁹ | 2³¹−1 |
|---|---|---|---|---|
| 64 (ui64) | 2.71e-14 | 2.71e-8 | 0.0267 | **0.1175** |
| 52 (f64 U) | 2.71e-14 | 2.71e-8 | 1.03e-3 [1.82e-3] | 2.2e-3 [3.91e-3] |
| 32 (ui32) | 2.71e-14 | 1.0e-9 [1.78e-9] | 1.0e-6 [1.78e-6] | 2.15e-6 [3.81e-6] |
| 23 (f32 U) | 2.55e-14 | 4.4e-11 [7.9e-11] | 4.4e-8 [7.9e-8] | 9.5e-8 [1.7e-7] |
| 16 | 3.9e-15 [6.9e-15] | 3.9e-12 | 3.9e-9 | 8.4e-9 [1.5e-8] |
| 8 | 2.4e-16 [4.3e-16] | 2.4e-13 | 2.4e-10 | 5.2e-10 [9.3e-10] |

**Is "discarding bits loosens the dependence" exactly right?**

- It is right in the monotone sense of (a): fewer bits never mean more dependence.
- It is false at the level of pairs and second moments: correlations are $-1/(N-1)$ and the pairwise total variation is about $2^{-64}$ for every $w$.
- For the joint law it holds only once $n\gtrsim2^{w/2}$. Before that, the table rows coincide.
- The accurate phrasing: the deficit (about $n^2/2^{65}$ missing coinciding pairs) is the same for every $w$, and discarding bits hides it among the $\binom n2/M$ coincidences that $2^w$ values have anyway.

**XOR versus keeping 32 bits.** By Theorem 3, the law depends on $f$ only through its fibre sizes. Under (A1), $y_0\oplus y_1$, $y_0$ alone, and the low 32 bits are therefore identical in law. XOR can only matter outside the model (for example if one output word were weaker), and it brings no provable gain there. By contrast, emitting **both** words as two ui32 values is a bijective re-encoding of the block: the $2n$ words are then exactly as far from i.i.d. as $n$ ui64 values, namely $\beta_N(n)$.

## 4. Effect on statistics from one draw

**Theorem 7.** Let $g$ be any function of a single value, $\mu$ and $\sigma^2$ its mean and variance under the uniform distribution on the $M$ outputs, and $\bar g=\frac1n\sum_i g(V_i)$.

- $E\bar g=\mu$ exactly: Monte Carlo estimates are unbiased.
- $\operatorname{Var}\bar g=\frac{\sigma^2}{n}\cdot\frac{N-n}{N-1}$. This is the finite-population correction with $N=2^{64}$ (not $2^w$), by Theorem 4.
- For i.i.d.-uniform values, $E s^2=\sigma^2$. Here $E s^2=\sigma^2N/(N-1)$, so $E[s^2/n]/\operatorname{Var}\bar g=N/(N-n)$.

The usual standard error therefore overstates the true one by $\sqrt{N/(N-n)}-1$: conservative, by at most $5.8\cdot10^{-11}$ (at $n=2^{31}$; $2.7\cdot10^{-11}$ at $n=10^9$). The variance factor for the mean is $1-1.16\cdot10^{-10}$ at the largest $n$.

Hoeffding (1963, Thm. 4) strengthens this: for any convex $\phi$, $E\phi(\sum g(V_i))\le$ its i.i.d. value. So Hoeffding/Chernoff tail bounds and i.i.d.-based confidence intervals stay valid for linear statistics.

**Pearson's $X^2$.** For bins that are unions of the $M$ outputs, with $B$ bins, $E X^2=(B-1)\frac{N-n}{N-1}$. Each bin count is hypergeometric. p-values are therefore very slightly too large (conservative). With $B=M$ bins, $X^2=\frac Mn(n+2C)-n$ is the collision test of section 5.

**Bounded statistics of a few positions.** If a statistic's terms depend on at most $r$ positions and lie in a range of width $G$, its expectation moves by at most $G\,r(r-1)/(2N)$, because the law of $r$ positions differs from i.i.d. by $\beta_N(r)$. Example: for the sample variance of uniforms, $|\operatorname{Var}_Q s^2-\operatorname{Var}_P s^2|\le 2/N=2^{-63}$, against $\operatorname{Var}_P s^2\approx1/(180n)$.

**Statistics that are affected** are those built on ties or collisions. Within one draw an estimate of $P(\text{repeat})$ from ui64 values is exactly 0, and for f64 uniforms the coincidence rate is low by a relative $2^{-12}$. For perspective, the finite grid matters far more than the dependence: an f32 uniform has mean $\tfrac12-2^{-24}$, and f32 Box–Muller normals cannot exceed $|z|=\sqrt{46\ln2}\approx5.65$, which removes tail mass of about $1.6\cdot10^{-8}$.

## 5. Distinguishability

**Switching lemma** (Impagliazzo–Rudich 1989; Bellare–Kilian–Rogaway; Bellare–Rogaway 2006; Chang–Nandi 2008). For any adversary making $q$ queries, a uniform permutation and a uniform function on $b$-bit strings are distinguished with advantage at most $q(q-1)/2^{b+1}$.

For anvl's fixed, non-adaptive inputs, the optimal advantage is exactly $\beta_N(n)$ (Theorem 6b). At $n=2^{31}-1$ that is $0.1175$, against the lemma's bound of $0.125$. By (a), no function of the blocks does better. For the real ThreeFry, add its PRP advantage, which is unprovable (see section 1).

**Optimal test for $w<64$.**

- The likelihood ratio is $\frac{Q}{P}=\prod_c\prod_{j<m_c}(1-j/K)\big/\big((N)_n/N^n\big)$.
- With no three equal values this equals $(1-1/K)^C$ times a constant, and in general $e^{-C/K}(1+O(m/K))$.
- By Neyman–Pearson, the most powerful test is therefore (up to negligible terms) "reject the i.i.d. hypothesis when $C$ is small".
- One draw gives $z_1=\dfrac{E_PC-E_QC}{\mathrm{sd}_PC}=\dfrac{\sqrt{\binom n2(M-1)}}{N-1}\approx\dfrac{n\sqrt M}{\sqrt2N}$.
- $\operatorname{Var}_Q C/\operatorname{Var}_P C=1-2.4\cdot10^{-4}$ for $w=52$ (computed exactly), so using the i.i.d. standard deviation is fine.
- $r$ independent draws give $\sqrt r\,z_1$. Reaching $z=3$ needs about $18N^2/(nM)$ values in total.

**Values needed to see the deficit at $z=3$:**

| w | z₁, one draw of 2³¹−1 | values, draws of 2³¹−1 | values, draws of 10⁹ |
|---|---|---|---|
| 64 | exact test: $P_{\rm iid}(\text{no repeat})=0.8825$ per draw | 53 draws = 1.1e11 (P < 0.00135) | — |
| 52 (f64) | 5.5e-3 | **6.33e14** (294,912 draws) | 1.4e15 |
| 32 (ui32) | 5.4e-6 | 6.6e20 | 1.4e21 |
| 23 (f32) | 2.4e-7 | 3.4e23 | 7.3e23 |
| 16 | 2.1e-8 | 4.4e25 | 9.4e25 |
| 8 | 1.3e-9 | 1.1e28 | 2.4e28 |

## 6. Across calls

**Theorem 8.** Under (A2), call $i$ uses key $K_i=P_{k_0}(c_i)$ with distinct counters $c_i$. Conditional on the event $E=\{K_i\ne k_0\ \forall i\}$, the draws of the $q$ calls are independent, and each has the law of Theorem 1. $P(E^c)=q/2^{64}$.

*Proof.* Condition on $P_{k_0}$. The keys are then fixed and distinct (fact F). The permutations $\{P_\kappa\}_{\kappa\ne k_0}$ are i.i.d. uniform and independent of $P_{k_0}$, so the permutations at fixed distinct indices are independent uniform. The conditional law does not depend on the key values, so it is also the unconditional law on $E$. ∎

So the keys being a without-replacement sample creates no dependence. Their only role is to be distinct.

If two calls had equal keys, their blocks would be identical at every common position. That cannot happen within one state. Across states it can: with $q$ keys in total, a collision has probability at most $q^2/2^{65}$. That is $3\cdot10^{-8}$ for $q=2^{20}$, $1.2\cdot10^{-4}$ for $2^{26}$, and $1/32$ for $2^{30}$. A colliding pair of calls produces identical draws, which is worth knowing for large multi-seed runs.

## 7. Help-page text

The "Properties" section of `?prim_sample_bits` is the corrected text derived here.

(The derivation also analysed a counter-based layout — one key per random state, block $t$ of a draw $=P_k(c+t)$, both words emitted as two ui32 values — under which the dependence spans every draw from a seed and becomes detectable after about $1.6\cdot10^{10}$ ui64 or $1.2\cdot10^{12}$ f64 values from one seed. That layout is not the one on this branch; it is one more reason to prefer the per-call key.)

## 8. What is exact, and what depends on what

| Status | Results |
|---|---|
| **Exact, no assumption** | Bijectivity; distinct blocks within a draw (a ui64 draw never repeats a value); distinct keys within a state; period of $2^{64}$; balancedness and the bit layouts; the deterministic grid and discretisation effects. |
| **Exact under (A1)** | Theorems 1, 3, 4; Proposition 5; Theorem 7 and the formulas for $X^2$ and $E s^2$; unbiasedness; total variation for ui64; the upper bounds in Theorem 6(c) and the lower bound in 6(d); the switching-lemma bound. |
| **Requires (A2)** | Independence across calls (Theorem 8); the key-collision probability across states. |
| **Asymptotic within the model (validated numerically, not fully proved)** | Theorem 6(e); the normal-approximation z-scores; the detection counts. |
| **Covered by nothing here** | ThreeFry's actual distance from (A1)/(A2), which rests on empirical testing only; attacks that use knowledge of the algorithm, such as enumerating seeds. |
