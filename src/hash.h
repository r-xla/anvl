// Shared hashing primitives: Boost's combiner, plus folds for R values that
// agree with R's identical(). Used by the structural tree hash (tree.cpp) and
// the dispatch cache key (dispatch_key.h).

#pragma once

#include <Rcpp.h>

#include <cstdint>

namespace anvl {

// The 64-bit specialization `hash_combine_impl<64>::fn` of Boost.ContainerHash,
// copied verbatim: folds the component hash `k` into the running seed `h`.
// https://github.com/boostorg/container_hash/blob/53c12550fa11221975f58a6c23581b4563153e04/include/boost/container_hash/hash.hpp#L311-L330
//
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0
// (https://www.boost.org/LICENSE_1_0.txt); full text in inst/COPYRIGHTS.
inline std::uint64_t hash_combine(std::uint64_t h, std::uint64_t k) {
  const std::uint64_t m = (std::uint64_t(0xc6a4a793) << 32) + 0x5bd1e995;
  const int r = 47;
  k *= m;
  k ^= k >> r;
  k *= m;
  h ^= k;
  h *= m;
  h += 0xe6546b64;
  return h;
}

// Fold `x` into `h` by its bit pattern. Callers compare doubles bitwise too
// (identical() with IDENT_NUM_AS_BITS), so no value needs canonicalizing: +0.0
// and -0.0, and NaNs of differing payloads, are distinct values and hash apart.
std::uint64_t hash_double(std::uint64_t h, double x);

// Fold an atomic vector's contents into `h`, so that two values compared with
// identical() land in different buckets and skip that call. It must never split
// what the caller's equality joins, hence strings fold their UTF-8 bytes rather
// than their stored ones: identical() compares strings encoding-aware
// ("e-acute" as UTF-8 and as latin1 are equal with different bytes).
//
// Attributes are not folded: they can only make two values unequal, never
// equal, so omitting them keeps the fold conservative. A non-atomic `v` (list,
// closure, environment) folds nothing and leaves `h` untouched.
std::uint64_t hash_atomic(std::uint64_t h, SEXP v);

}  // namespace anvl
