#include <Rcpp.h>
#include <testthat.h>

#include <string>
#include <vector>

#include "dispatch_key.h"

// The dispatcher's cache key decides which compiled program a call runs. A
// mistake here does not error -- it silently returns someone else's executable.
// So these tests live next to the key rather than behind an R self-test hook:
// they can fabricate a device token and build a kRData leaf directly, neither
// of which an R fixture over PJRTBuffers could express.

namespace {

using anvl::anvl_dtype_from_pjrt;
using anvl::anvl_dtype_from_tengen;
using anvl::anvl_dtype_name;
using anvl::AnvlDtype;
using anvl::Aval;
using anvl::CacheKey;
using anvl::CacheKeyEq;
using anvl::CacheKeyHash;
using anvl::KeyLeaf;
using anvl::RTree;

std::size_t hash_of(const CacheKey& k) { return CacheKeyHash{}(k); }
bool eq(const CacheKey& a, const CacheKey& b) { return CacheKeyEq{}(a, b); }

// The tree impl_dispatch_run() builds for `n` unnamed top-level args: a root
// ListNode over n leaves.
RTree flat_tree(std::size_t n) {
  RTree t;
  t.kind.push_back(RTree::ListNode);
  t.n_children.push_back(static_cast<std::int32_t>(n));
  t.subtree_nodes.push_back(static_cast<std::int32_t>(n + 1));
  t.name_off.push_back(-1);
  for (std::size_t k = 0; k < n; ++k) {
    t.kind.push_back(RTree::LeafNode);
    t.n_children.push_back(0);
    t.subtree_nodes.push_back(1);
    t.name_off.push_back(-1);
  }
  return t;
}

Aval mk_aval(AnvlDtype dtype, std::vector<int64_t> shape, bool ambiguous) {
  Aval a;
  a.dtype = dtype;
  a.shape = std::move(shape);
  a.ambiguous = ambiguous;
  return a;
}

KeyLeaf array_leaf(Aval a) {
  KeyLeaf kl;
  kl.kind = KeyLeaf::kArray;
  kl.aval = std::move(a);
  return kl;
}

KeyLeaf rdata_leaf(Aval a) {
  KeyLeaf kl;
  kl.kind = KeyLeaf::kRData;
  kl.aval = std::move(a);
  return kl;
}

KeyLeaf static_leaf(SEXP value) {
  KeyLeaf kl;
  kl.kind = KeyLeaf::kStatic;
  kl.value = value;
  return kl;
}

// A key over `leaves`, with the tree impl_dispatch_run() would have built.
CacheKey key_of(std::vector<KeyLeaf> leaves) {
  CacheKey k;
  k.in_tree = flat_tree(leaves.size());
  k.leaves = std::move(leaves);
  return k;
}

// Two distinct addresses to stand in for two interned device objects. Device
// tokens are never dereferenced, so any two distinct addresses will do.
const int kDeviceA = 0;
const int kDeviceB = 0;
const anvl::DeviceToken kDevA = &kDeviceA;
const anvl::DeviceToken kDevB = &kDeviceB;

// A tengen DataType: length-1 STRSXP classed "DataType".
inline Rcpp::CharacterVector tengen_dtype(const char* name) {
  Rcpp::CharacterVector s(1);
  s[0] = name;
  s.attr("class") = "DataType";
  return s;
}

// A length-1 character vector holding `bytes` under a chosen encoding, so a
// test can build the same text in latin1 and in UTF-8. The Rcpp wrapper keeps
// the CHARSXP protected for the leaf's lifetime.
Rcpp::CharacterVector str_ce(const char* bytes, int len, cetype_t enc) {
  return Rcpp::CharacterVector(
      Rf_ScalarString(Rf_mkCharLenCE(bytes, len, enc)));
}

}  // namespace

context("AnvlDtype") {
  test_that("every tengen dtype maps to a distinct AnvlDtype") {
    // A fall-through here would key two dtypes alike and run one's program for
    // the other. It has happened; hence one assertion per width.
    expect_true(anvl_dtype_from_tengen(tengen_dtype("bool")) ==
                AnvlDtype::kBool);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("i8")) == AnvlDtype::kI8);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("i16")) == AnvlDtype::kI16);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("i32")) == AnvlDtype::kI32);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("i64")) == AnvlDtype::kI64);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("ui8")) == AnvlDtype::kU8);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("ui16")) ==
                AnvlDtype::kU16);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("ui32")) ==
                AnvlDtype::kU32);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("ui64")) ==
                AnvlDtype::kU64);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("f32")) == AnvlDtype::kF32);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("f64")) == AnvlDtype::kF64);
  }

  test_that(
      "a dtype AnvlDtype cannot name yields kInvalid, never a neighbour") {
    expect_true(anvl_dtype_from_tengen(tengen_dtype("f16")) ==
                AnvlDtype::kInvalid);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("bf16")) ==
                AnvlDtype::kInvalid);
    expect_true(anvl_dtype_from_tengen(tengen_dtype("c64")) ==
                AnvlDtype::kInvalid);
    expect_true(anvl_dtype_from_tengen(Rcpp::CharacterVector::create("f32")) ==
                AnvlDtype::kInvalid);
    expect_true(anvl_dtype_from_tengen(Rcpp::IntegerVector::create(32)) ==
                AnvlDtype::kInvalid);
  }

  test_that("pjrt element types map onto the same names") {
    expect_true(anvl_dtype_from_pjrt(PJRT_Buffer_Type_F32) == AnvlDtype::kF32);
    expect_true(anvl_dtype_from_pjrt(PJRT_Buffer_Type_S32) == AnvlDtype::kI32);
    expect_true(anvl_dtype_from_pjrt(PJRT_Buffer_Type_U64) == AnvlDtype::kU64);
    expect_true(anvl_dtype_from_pjrt(PJRT_Buffer_Type_PRED) ==
                AnvlDtype::kBool);
    // A type PJRT enumerates but anvl cannot represent.
    expect_true(anvl_dtype_from_pjrt(PJRT_Buffer_Type_F16) ==
                AnvlDtype::kInvalid);
    expect_true(std::string(anvl_dtype_name(AnvlDtype::kU8)) == "ui8");
    expect_true(std::string(anvl_dtype_name(AnvlDtype::kI64)) == "i64");
  }
}

context("CacheKey: aval-keyed leaves") {
  const Aval f32_2x3 = mk_aval(AnvlDtype::kF32, {2, 3}, false);

  test_that("equal signatures compare equal and hash alike") {
    CacheKey a = key_of({array_leaf(f32_2x3), array_leaf(f32_2x3)});
    CacheKey b = key_of({array_leaf(f32_2x3), array_leaf(f32_2x3)});
    expect_true(eq(a, b));
    expect_true(hash_of(a) == hash_of(b));
  }

  test_that("dtype, shape, ambiguity and arity each split the key") {
    CacheKey base = key_of({array_leaf(f32_2x3)});

    CacheKey dtype =
        key_of({array_leaf(mk_aval(AnvlDtype::kI32, {2, 3}, false))});
    expect_false(eq(base, dtype));
    expect_false(hash_of(base) == hash_of(dtype));

    CacheKey shape =
        key_of({array_leaf(mk_aval(AnvlDtype::kF32, {3, 2}, false))});
    expect_false(eq(base, shape));
    expect_false(hash_of(base) == hash_of(shape));

    CacheKey rank = key_of({array_leaf(mk_aval(AnvlDtype::kF32, {}, false))});
    expect_false(eq(base, rank));
    expect_false(hash_of(base) == hash_of(rank));

    CacheKey ambig =
        key_of({array_leaf(mk_aval(AnvlDtype::kF32, {2, 3}, true))});
    expect_false(eq(base, ambig));
    expect_false(hash_of(base) == hash_of(ambig));

    CacheKey arity = key_of({array_leaf(f32_2x3), array_leaf(f32_2x3)});
    expect_false(eq(base, arity));
    expect_false(hash_of(base) == hash_of(arity));
  }

  test_that("a kArray and a kRData leaf of one aval are one key") {
    // They compile to the same program; only where execution finds the input
    // differs, and that is decided per call. Keying them apart would compile
    // `f(x, y)` and `f(x, 1)` twice.
    CacheKey arr = key_of({array_leaf(f32_2x3)});
    CacheKey lit = key_of({rdata_leaf(f32_2x3)});
    expect_true(eq(arr, lit));
    expect_true(hash_of(arr) ==
                hash_of(lit));  // or the map never compares them
  }

  test_that("an aval-keyed leaf never equals a value-keyed one") {
    Rcpp::IntegerVector v = Rcpp::IntegerVector::create(1);
    CacheKey arr = key_of({array_leaf(f32_2x3)});
    CacheKey stat = key_of({static_leaf(v)});
    expect_false(eq(arr, stat));
  }
}

context("CacheKey: device and tree") {
  const Aval f32 = mk_aval(AnvlDtype::kF32, {2}, false);

  test_that("the device token splits the key and is folded into the hash") {
    CacheKey a = key_of({array_leaf(f32)});
    a.device = kDevA;
    CacheKey b = key_of({array_leaf(f32)});
    b.device = kDevB;
    CacheKey a2 = key_of({array_leaf(f32)});
    a2.device = kDevA;

    expect_false(eq(a, b));
    expect_false(hash_of(a) == hash_of(b));
    expect_true(eq(a, a2));
    expect_true(hash_of(a) == hash_of(a2));

    // No device (all-literal call under a dispatcher with no resolver) is its
    // own key, not a wildcard.
    CacheKey none = key_of({array_leaf(f32)});
    expect_false(eq(a, none));
  }

  test_that("the argument tree splits the key") {
    // Same single leaf, but reached through a differently-shaped tree.
    CacheKey a = key_of({array_leaf(f32)});
    CacheKey b = key_of({array_leaf(f32)});
    b.in_tree.names.push_back("x");
    b.in_tree.name_off[0] = 0;
    expect_false(eq(a, b));
    expect_false(hash_of(a) == hash_of(b));
  }
}

context("CacheKey: value-keyed (static) leaves") {
  test_that("equal statics compare equal and hash alike") {
    Rcpp::IntegerVector a = Rcpp::IntegerVector::create(42);
    Rcpp::IntegerVector b = Rcpp::IntegerVector::create(42);
    CacheKey ka = key_of({static_leaf(a)});
    CacheKey kb = key_of({static_leaf(b)});
    expect_true(eq(ka, kb));
    expect_true(hash_of(ka) == hash_of(kb));
  }

  test_that("distinct atomic statics hash apart, not merely compare unequal") {
    // If they only compared unequal, every static of one type and length would
    // share a bucket and the map would fall back on identical() per lookup.
    auto differ = [](SEXP x, SEXP y) {
      CacheKey a = key_of({static_leaf(x)});
      CacheKey b = key_of({static_leaf(y)});
      return !eq(a, b) && hash_of(a) != hash_of(b);
    };
    expect_true(differ(Rcpp::LogicalVector::create(true),
                       Rcpp::LogicalVector::create(false)));
    expect_true(
        differ(Rcpp::IntegerVector::create(1), Rcpp::IntegerVector::create(2)));
    expect_true(differ(Rcpp::CharacterVector::create("a"),
                       Rcpp::CharacterVector::create("b")));
    expect_true(differ(Rcpp::NumericVector::create(1.5),
                       Rcpp::NumericVector::create(2.5)));
    expect_true(differ(Rcpp::NumericVector::create(1, 2),
                       Rcpp::NumericVector::create(2, 1)));
  }

  test_that("doubles are keyed bitwise: +0 and -0 are distinct") {
    // bit64 stores NA_integer64_ as the bit pattern of -0.0. Under R's default
    // `==` semantics it would share a cache entry with 0 and silently run its
    // program. IDENT_NUM_AS_BITS splits them; the cost is a redundant compile.
    Rcpp::NumericVector pos = Rcpp::NumericVector::create(0.0);
    Rcpp::NumericVector neg = Rcpp::NumericVector::create(-0.0);
    expect_true(std::signbit(neg[0]));  // the compiler did not fold it away
    CacheKey a = key_of({static_leaf(pos)});
    CacheKey b = key_of({static_leaf(neg)});
    expect_false(eq(a, b));
    expect_false(hash_of(a) == hash_of(b));
  }

  test_that("NaN equals itself; NaN and NA_real_ are distinct") {
    Rcpp::NumericVector nan1 = Rcpp::NumericVector::create(R_NaN);
    Rcpp::NumericVector nan2 = Rcpp::NumericVector::create(R_NaN);
    Rcpp::NumericVector na = Rcpp::NumericVector::create(NA_REAL);
    CacheKey a = key_of({static_leaf(nan1)});
    CacheKey b = key_of({static_leaf(nan2)});
    CacheKey c = key_of({static_leaf(na)});
    expect_true(eq(a, b));
    expect_true(hash_of(a) == hash_of(b));
    expect_false(eq(a, c));
  }

  test_that(
      "a non-atomic static is separated by identical(), not by its hash") {
    // hash_atomic() folds nothing for a list, so both keys land in one bucket.
    // Equality must still tell them apart -- a coarser hash costs a collision,
    // never a wrong cache hit.
    Rcpp::List l1 = Rcpp::List::create(Rcpp::Named("a") = 1);
    Rcpp::List l2 = Rcpp::List::create(Rcpp::Named("a") = 2);
    CacheKey a = key_of({static_leaf(l1)});
    CacheKey b = key_of({static_leaf(l2)});
    expect_true(hash_of(a) == hash_of(b));  // same type, same length
    expect_false(eq(a, b));                 // identical() separates them
  }

  test_that("closure statics hash on their formals and body") {
    // A closure has no vector contents, so it used to fold nothing: every
    // function static shared one bucket. hash_closure() folds the formal names
    // and the body, so distinct functions land in distinct buckets -- while
    // still never splitting what identical() joins (below).
    auto fun = [](const char* src) {
      Rcpp::Environment base = Rcpp::Environment::base_env();
      Rcpp::Function parse_fn = base["parse"];
      Rcpp::Function eval_fn = base["eval"];
      return eval_fn(parse_fn(Rcpp::Named("text") = src));
    };
    Rcpp::RObject relu = fun("function(x) pmax(x, 0)");
    Rcpp::RObject tanh_f = fun("function(x) tanh(x)");
    Rcpp::RObject arity = fun("function(x, y) pmax(x, 0)");  // same body

    CacheKey a = key_of({static_leaf(relu)});
    CacheKey b = key_of({static_leaf(tanh_f)});
    CacheKey c = key_of({static_leaf(arity)});
    expect_false(eq(a, b));
    expect_true(hash_of(a) != hash_of(b));  // body splits the bucket
    expect_false(eq(a, c));
    expect_true(hash_of(a) != hash_of(c));  // formals split it too

    // The contract: never split what identical() joins. The same closure --
    // and a byte-compiled copy of it, whose BODY is a BCODESXP -- must hash
    // alike, or the map would look in the wrong bucket and cache it twice.
    Rcpp::Function cmpfun =
        Rcpp::Environment::namespace_env("compiler")["cmpfun"];
    Rcpp::RObject relu_bc = cmpfun(relu);
    CacheKey a2 = key_of({static_leaf(relu)});
    CacheKey a_bc = key_of({static_leaf(relu_bc)});
    expect_true(eq(a, a2));
    expect_true(hash_of(a) == hash_of(a2));
    expect_true(hash_of(a) == hash_of(a_bc));
  }

  test_that("strings key on their UTF-8 content, across encodings") {
    // identical() compares strings after translating to UTF-8, so the same text
    // in latin1 and in UTF-8 is one value and must share a key -- while two
    // *distinct* non-ASCII strings must land in different buckets.
    Rcpp::CharacterVector cafe_utf8 =
        str_ce("caf\xC3\xA9", 5, CE_UTF8);  // café
    Rcpp::CharacterVector cafe_latin1 =
        str_ce("caf\xE9", 4, CE_LATIN1);                               // café
    Rcpp::CharacterVector uuml_utf8 = str_ce("\xC3\xBC", 2, CE_UTF8);  // ü

    CacheKey cafe1 = key_of({static_leaf(cafe_utf8)});
    CacheKey cafe2 = key_of({static_leaf(cafe_latin1)});
    CacheKey uuml = key_of({static_leaf(uuml_utf8)});

    // Same text, two encodings: identical() joins them, so must the hash.
    expect_true(eq(cafe1, cafe2));
    expect_true(hash_of(cafe1) == hash_of(cafe2));

    // Distinct non-ASCII strings hash apart, not merely compare unequal.
    expect_false(eq(cafe1, uuml));
    expect_false(hash_of(cafe1) == hash_of(uuml));
  }

  test_that("statics of different arity never merge") {
    Rcpp::IntegerVector one = Rcpp::IntegerVector::create(1);
    Rcpp::IntegerVector two = Rcpp::IntegerVector::create(2);
    CacheKey a = key_of({static_leaf(one)});
    CacheKey b = key_of({static_leaf(one), static_leaf(two)});
    expect_false(eq(a, b));
    expect_false(hash_of(a) == hash_of(b));
  }
}
