// The dispatcher's backend abstraction.
//
// The dispatcher core (dispatch.cpp) is backend-agnostic: it flattens a call,
// classifies the leaves, builds the cache key, and drives the LRU + compile
// protocol. Everything a backend has an opinion about lives behind the Engine
// interface defined here:
//
//   * how an Aval is read off one of the backend's arrays,
//   * what a cache entry holds (the compile callback's artifacts),
//   * how a call executes against an entry, and how its outputs are wrapped.
//
// PjrtEngine is the fast path for the "pjrt" backend: avals come off a
// PJRTBuffer's cached native metadata, inputs are assembled and executed
// natively, and the output buffers are wrapped into AnvlArrays and re-nested
// without leaving C++. ClosureEngine is the generic vehicle for any other
// backend (anvl's quickr, or one pjrt has never heard of): the entry holds a
// compiled R closure that is called on the flat leaves and returns the finished
// value, so the backend keeps full control over execution and wrapping.

#pragma once

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dispatch_key.h"
#include "tree.h"

namespace anvl {

// What the dispatcher core needs from one array input, read by the engine that
// owns the leaf's backend (Engine::read_array). The core applies input policy
// -- plain-reject, backend-match, the device rules -- to these values; it never
// parses the array's own fields. Only `$data` is guaranteed by anvl's backend
// contract, so `aval`/`device`/`backend` come from the buffer (PjrtEngine) or
// an extractor (ClosureEngine), never from a fixed field layout.
//
// `device` is an RObject rather than a bare SEXP because an extractor may
// fabricate it (a backend need not intern its devices), in which case nothing
// else roots it between read_array() returning and Engine::canonical_device()
// taking its own reference.
struct ArrayLeaf {
  Aval aval;
  Rcpp::RObject data;    // $data: the execute-time input (buffer or R value)
  Rcpp::RObject device;  // the R device object: the key's device token
  std::string backend;   // the leaf's backend tag (policy + error messages)
};

// Classification of a bare (class-less) R value as an uploadable
// literal/array leaf. Mirrors anvl's is_valid_r_lit / is_valid_r_array and its
// default dtypes (double -> f32, integer -> i32, logical -> pred -- also
// pjrt_scalar()'s defaults).
struct RDataInfo {
  AnvlDtype dtype = AnvlDtype::kInvalid;
  std::vector<int64_t> shape;  // empty for a rank-0 literal
};

// Nullopt for anything that is not uploadable bare R data -- NA literals
// included: they have no dtype.
std::optional<RDataInfo> classify_rdata(SEXP leaf);

// class(x)[1L], as R would print it, for naming a rejected leaf's type.
std::string r_class_name(SEXP x);

// The subject of a rejection message: "input `x$w[[2]]`". Every leaf has a
// non-empty path, because the dispatcher roots the tree at a ListNode over the
// arguments, so even an unnamed one is reached as `[[1]]`. `leaf_index` is
// 0-based; tree_path() is 1-based.
std::string leaf_subject(const RTree& in_tree, std::size_t leaf_index);

// One execute-time input: a leaf the engine must supply to the compiled
// program, in the order the program expects. Static leaves never appear here --
// they are baked into the program as constants and live only in the cache key
// -- so the inputs to supply are exactly this sequence.
struct ExecInput {
  SEXP value = R_NilValue;     // an array leaf's `$data`, or the bare R leaf
  const Aval* aval = nullptr;  // an upload needs its shape
  bool upload = false;         // bare R data: upload it. Else: ready to use
};

// An engine's per-entry material -- what the compile callback produced, in the
// engine's own shape. Owned by the CacheEntry. An R object an engine holds is
// held in an Rcpp type (XPtr, List, Function, RObject), which roots it for as
// long as the entry lives and drops it when the entry is destroyed.
struct EntryData {
  // Virtual because CacheEntry deletes the derived entry through this base
  // pointer; a non-virtual destructor would skip the derived one.
  virtual ~EntryData() = default;
};

// One cache entry: the engine's data plus the key's own R objects, rooted for
// the entry's lifetime and released when it is evicted or the dispatcher is
// torn down.
//
// Everything here roots itself, so there is no preserve/release bookkeeping to
// get wrong: an entry abandoned mid-construction (because the compile callback
// threw) releases exactly what it had taken.
struct CacheEntry {
  // The key's static leaf values. The engine's own R objects live in `data`
  // and are rooted by it, not here.
  std::vector<Rcpp::RObject> keep;
  std::unique_ptr<EntryData> data;

  // Root `x` for this entry's lifetime. R_NilValue is a no-op.
  void keep_alive(SEXP x) {
    if (x != R_NilValue) keep.emplace_back(x);
  }
};

// What a backend implements so the dispatcher core can stay agnostic. One
// engine instance per Dispatcher (an engine may hold per-dispatcher state, e.g.
// PjrtEngine's reusable execute options, or the canonical-device table).
class Engine {
 public:
  Engine() = default;
  virtual ~Engine() = default;
  // An engine owns R objects and is held by one Dispatcher; copying one would
  // duplicate that ownership for no purpose.
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // The canonical representative of a device object; its address is the cache
  // key's DeviceToken. Resolution is by object identity first -- free for a
  // backend that interns its devices -- and r_identical() as a fallback, so a
  // backend handing out equal-but-distinct device objects still lands on one
  // token rather than splitting the cache. Canonical objects are preserved for
  // the dispatcher's lifetime (that is what keeps a token's address stable), so
  // a backend should hand out few distinct device objects.
  virtual SEXP canonical_device(SEXP device);

  // Read one array input into the material the core needs. Nullopt when `leaf`
  // is not an AnvlArray this engine can read, so the core falls through to
  // classify_rdata(). Backend-specific structural validation (e.g. "$data must
  // be a PJRTBuffer") is done only once the leaf's backend tag matches this
  // engine's, so the core's plain-reject / backend-match policy still reports
  // first for a foreign leaf. `in_tree`/`leaf_index` name the leaf on
  // rejection.
  virtual std::optional<ArrayLeaf> read_array(SEXP leaf, const RTree& in_tree,
                                              std::size_t leaf_index) = 0;

  // Build the entry's data from the compile callback's result, validating it
  // first: a malformed result must throw rather than produce an entry.
  virtual void build_entry(const Rcpp::List& res, CacheEntry& e) const = 0;

  // Execute one call against a cached entry and return the finished R value.
  // `inputs` is the call's execute-time inputs, already classified and in order
  // (see ExecInput): the engine supplies exactly these, and knows nothing of
  // cache keys or static arguments. Placing an input on the entry's device
  // under `move_inputs` is the engine's own business (it is the only party that
  // knows what `$data` is), so that policy is engine state, not a per-call
  // argument.
  //
  // The entry is const: build_entry() constructs it whole, so execution only
  // ever reads it.
  virtual SEXP run(const CacheEntry& e,
                   const std::vector<ExecInput>& inputs) const = 0;

 private:
  // The distinct devices this dispatcher has seen, each rooted for the
  // engine's lifetime; owned by the default canonical_device().
  std::vector<Rcpp::RObject> canonical_devices_;
};

// `engine_name` is the R-facing selector: "pjrt" or "closure"; throws on any
// other value. `backend` is the tag the dispatcher's arrays carry (the engine
// needs it to stamp the arrays it wraps and to recognize its own leaves).
// `extractor` is the R closure ClosureEngine calls to read a leaf's metadata
// via the backend's accessors (required for "closure"; ignored, pass
// R_NilValue, for "pjrt"). See ?dispatcher.
std::unique_ptr<Engine> make_engine(const std::string& engine_name,
                                    std::string backend, bool move_inputs,
                                    SEXP extractor);

}  // namespace anvl
