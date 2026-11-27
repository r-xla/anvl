// SIMD kernels for the Poisson distribution.
//
// We write each kernel using Google Highway, which compiles it for every SIMD
// instruction set the CPU architecture has. When a kernel first runs, Highway
// provides runtime dispatch, picking the best version the executing CPU
// supports.

#include "poisson-kernels.h"

#include <stddef.h>

#include <limits>

// Highway includes this file again once for each SIMD target. For that to
// work, the compiler needs `-I.`, which src/Makevars sets.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "poisson-kernels.cpp"
#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace anvl_hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// The most terms we ever sum for one element. This is only a safety net. In
// the regions where `ppois_raw()` uses the series, it converges within about
// 170 terms in f64 and about 90 in f32.
constexpr int kMaxTerms = 1000;

// Sums the Poisson tail series, the same series as R's `pd_upper_series()`
// and `pd_lower_series()`. Here k is a whole number k >= 0, and lambda is
// finite and positive.
//
// When lambda <= k, we sum the upper series
//   lambda / (k + 1) + lambda^2 / ((k + 1)(k + 2)) + ...
// Multiplying it by dpois(k, lambda) gives P(X > k).
//
// When lambda > k, we sum the lower series, which has k terms:
//   k / lambda + k (k - 1) / lambda^2 + ...
// Multiplying it by dpois(k, lambda) gives P(X < k).
//
// We build each term from the previous one by multiplying by a ratio r:
//   upper series => r = lambda / (k + n)
//   lower series => r = (k - n + 1) / lambda.
// Numerator and denominator of r are stored in `num` and `den`.
//
// The stopping rule is cheap. Say the next term is t, with ratio r. The terms
// following it shrink at least as fast as a geometric series with ratio r, so
// everything remaining adds up to at most t / (1 - r). Once that is less than
// half an ulp of the sum so far, more terms won't change the answer, and we
// stop without adding t.
//
// The ratios are below one wherever the series converges, so nothing can
// overflow. The sum also stops if a term underflows to zero. The lower
// series stops by itself after k terms, because the numerator reaches zero.
//
// All the lanes of a vector go round the loop together. A lane that has
// converged stops changing via the `live` mask.
//
// If k or lambda is NaN, the result is NaN. For inputs far outside the
// regions where `ppois_raw()` uses the series, we may give up after kMaxTerms
// terms and return the sum so far.
template <class D, class V = hn::Vec<D>>
HWY_INLINE V Series(D d, V k, V lambda) {
  using T = hn::TFromD<D>;
  const T tol_scalar = static_cast<T>(std::numeric_limits<T>::epsilon() / 2);
  const V tol = hn::Set(d, tol_scalar);
  const V one = hn::Set(d, T{1});
  const V zero = hn::Zero(d);

  const auto upper = hn::Le(lambda, k);
  V num = hn::IfThenElse(upper, lambda, k);
  V den = hn::IfThenElse(upper, hn::Add(k, one), lambda);
  const V num_step = hn::IfThenElse(upper, zero, hn::Set(d, T{-1}));
  const V den_step = hn::IfThenElse(upper, one, zero);

  V term = one;
  V sum = zero;
  auto live = hn::Eq(zero, zero);
  for (int n = 0; n < kMaxTerms; ++n) {
    const V r = hn::Div(num, den);
    const V t = hn::Mul(term, r);
    // Keep going while t / (1 - r) > tol * sum. A NaN makes this false, so a
    // lane with a NaN input stops straight away.
    live =
        hn::And(live, hn::Gt(t, hn::Mul(hn::Mul(hn::Sub(one, r), sum), tol)));
    if (hn::AllFalse(d, live)) break;
    sum = hn::IfThenElse(live, hn::Add(sum, t), sum);
    term = t;
    num = hn::Add(num, num_step);
    den = hn::Add(den, den_step);
  }
  // The loop leaves a lane with a NaN input at zero, so we put the NaN back.
  const V probe = hn::Add(k, lambda);
  return hn::IfThenElse(hn::IsNaN(probe), probe, sum);
}

template <typename T>
void SeriesRange(const PoissonSeriesArgs<T>& args, size_t begin, size_t end) {
  const hn::ScalableTag<T> d;
  const size_t lanes = hn::Lanes(d);
  const T* HWY_RESTRICT k = args.k;
  const T* HWY_RESTRICT lambda = args.lambda;
  T* HWY_RESTRICT out = args.out;
  const auto k0 = hn::Set(d, k[0]);
  const auto lambda0 = hn::Set(d, lambda[0]);

  size_t i = begin;
  for (; i + lanes <= end; i += lanes) {
    const auto vk = args.scalar_k ? k0 : hn::LoadU(d, k + i);
    const auto vl = args.scalar_lambda ? lambda0 : hn::LoadU(d, lambda + i);
    hn::StoreU(Series(d, vk, vl), d, out + i);
  }
  if (i < end) {
    // The lanes past the end load zeros. Their first term is zero, so they
    // stop straight away.
    const size_t rem = end - i;
    const auto vk = args.scalar_k ? k0 : hn::LoadN(d, k + i, rem);
    const auto vl =
        args.scalar_lambda ? lambda0 : hn::LoadN(d, lambda + i, rem);
    hn::StoreN(Series(d, vk, vl), d, out + i, rem);
  }
}

}  // namespace

void PoissonSeriesF32(const PoissonSeriesArgs<float>& args, size_t begin,
                      size_t end) {
  SeriesRange(args, begin, end);
}
void PoissonSeriesF64(const PoissonSeriesArgs<double>& args, size_t begin,
                      size_t end) {
  SeriesRange(args, begin, end);
}

}  // namespace HWY_NAMESPACE
}  // namespace anvl_hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace anvl_hwy {

HWY_EXPORT(PoissonSeriesF32);
HWY_EXPORT(PoissonSeriesF64);

void poisson_series_f32(const PoissonSeriesArgs<float>& args, size_t begin,
                        size_t end) {
  HWY_DYNAMIC_DISPATCH(PoissonSeriesF32)(args, begin, end);
}
void poisson_series_f64(const PoissonSeriesArgs<double>& args, size_t begin,
                        size_t end) {
  HWY_DYNAMIC_DISPATCH(PoissonSeriesF64)(args, begin, end);
}

}  // namespace anvl_hwy
#endif  // HWY_ONCE
