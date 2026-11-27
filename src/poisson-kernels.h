// SIMD kernels for the Poisson distribution. Highway compiles them for every
// SIMD instruction set the CPU architecture has, and picks one at run time.

#ifndef ANVL_POISSON_KERNELS_H_
#define ANVL_POISSON_KERNELS_H_

#include <stddef.h>

namespace anvl_hwy {

// The operands of a poisson_series_*() call. `k` and `lambda` each hold either
// a single element, which is used for every result, or one element per result.
template <typename T>
struct PoissonSeriesArgs {
  const T* k;
  const T* lambda;
  T* out;
  bool scalar_k;
  bool scalar_lambda;
};

// Sums the Poisson tail series for elements [begin, end) and writes the sums
// to `args.out`. Series() in poisson-kernels.cpp explains the series.
void poisson_series_f32(const PoissonSeriesArgs<float>& args, size_t begin,
                        size_t end);
void poisson_series_f64(const PoissonSeriesArgs<double>& args, size_t begin,
                        size_t end);

}  // namespace anvl_hwy

#endif  // ANVL_POISSON_KERNELS_H_
