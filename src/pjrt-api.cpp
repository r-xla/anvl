// The pjrt C interface version handshake.
//
// anvl's native dispatcher calls into pjrt through inst/include/pjrt/api.h,
// whose signatures are versioned. Reporting both numbers to R lets .onLoad
// compare them and fail with a legible message, rather than letting a
// mismatched pair call through incompatible signatures later.
//
// The comparison happens in R rather than in R_init_anvl for two reasons: the
// pjrt namespace -- and thus its registered entry points -- is only guaranteed
// to be loaded by the time .onLoad runs, and an error raised from R is a plain
// R error rather than one thrown across the package-load machinery.

#include <Rcpp.h>

#include "pjrt/api.h"

// `built` is the version this build of anvl was compiled against, baked in from
// the header; `runtime` is what the installed pjrt reports.
// [[Rcpp::export]]
Rcpp::IntegerVector impl_pjrt_api_versions() {
  return Rcpp::IntegerVector::create(Rcpp::Named("built") = PJRT_C_API_VERSION,
                                     Rcpp::Named("runtime") =
                                         pjrt_c_api_version());
}
