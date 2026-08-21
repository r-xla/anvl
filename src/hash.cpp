// The R-value folds declared in hash.h. Their contract -- never split what
// identical() joins -- is documented there; this file mirrors the rules R's own
// value hashes use (src/main/unique.c: rhash, shash).

#include "hash.h"

#include <Rcpp.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

namespace anvl {

std::uint64_t hash_double(std::uint64_t h, double x) {
  // Hash the raw bit pattern, not the numeric value.
  std::uint64_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return hash_combine(h, bits);
}

std::uint64_t hash_atomic(std::uint64_t h, SEXP v) {
  const R_xlen_t n = Rf_xlength(v);
  // Contents only: TYPEOF/dtype is not folded here. The caller's key already
  // carries the leaf's type/shape, and omitting it can at worst cause a
  // collision (identical() still decides), never split identical() values.
  switch (TYPEOF(v)) {
    case LGLSXP: {
      const int* p = LOGICAL(v);
      // R stores logicals as 32-bit ints (NA included). Reinterpreting them as
      // uint32_t is a bijection, so no two values collide; hash_combine() then
      // zero-extends to uint64_t.
      for (R_xlen_t i = 0; i < n; ++i) {
        h = hash_combine(h, static_cast<std::uint32_t>(p[i]));
      }
      break;
    }
    case INTSXP: {
      const int* p = INTEGER(v);
      for (R_xlen_t i = 0; i < n; ++i) {
        h = hash_combine(h, static_cast<std::uint32_t>(p[i]));
      }
      break;
    }
    case REALSXP: {
      const double* p = REAL(v);
      for (R_xlen_t i = 0; i < n; ++i) h = hash_double(h, p[i]);
      break;
    }
    case CPLXSXP: {  // complex
      const Rcomplex* p = COMPLEX(v);
      for (R_xlen_t i = 0; i < n; ++i) {
        h = hash_double(h, p[i].r);
        h = hash_double(h, p[i].i);
      }
      break;
    }
    case RAWSXP: {
      const Rbyte* p = RAW(v);
      for (R_xlen_t i = 0; i < n; ++i) {
        // Unsigned already, so it widens without a sign bit to worry about.
        h = hash_combine(h, static_cast<std::uint64_t>(p[i]));
      }
      break;
    }
    case STRSXP: {
      for (R_xlen_t i = 0; i < n; ++i) {
        SEXP s = STRING_ELT(v, i);
        if (s == NA_STRING) {
          // NA is a sentinel, not a translatable string, so fold a fixed
          // constant: all that matters is every NA_character_ folds alike. The
          // value is arbitrary; 0x4E41 spells the bytes 'N','A' as a mnemonic.
          h = hash_combine(h, 0x4E41ULL);
          continue;
        }
        // Fold the UTF-8 bytes, not the stored bytes. identical() compares
        // strings after translating to UTF-8 (src/main/unique.c: Seql), so the
        // same text held as latin1 and as UTF-8 -- one value to identical() --
        // must fold alike; this mirrors R's own shash(). A "bytes"-encoded
        // string cannot be translated and needs no normalization (its raw bytes
        // are its identity), so fold those directly.
        //
        // translateCharUTF8() may R_alloc a conversion buffer on R's transient
        // (vmax) stack, which R frees only when the enclosing .Call returns.
        // Over a long vector those buffers would pile up for the whole call, so
        // vmaxget() snapshots the stack top and vmaxset() rewinds it each
        // iteration -- releasing the buffer immediately, keeping memory flat.
        const void* vmax = vmaxget();
        const char* c =
            Rf_getCharCE(s) == CE_BYTES ? CHAR(s) : Rf_translateCharUTF8(s);
        h = hash_combine(h, std::hash<std::string>{}(std::string(c)));
        vmaxset(vmax);
      }
      break;
    }
    default:
      break;
  }
  return h;
}

}  // namespace anvl
