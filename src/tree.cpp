// The R-exposed Rtree API over the flat RTree module in tree.h. A tree handed
// to R is a heap RTree behind an Rcpp::XPtr with S3 class "RTree"; it is opaque
// to R -- every operation that walks the tree walks it here and returns either
// a fully-owned result RTree or a plain R value (mask, sizes, names, strings).
// No sub-tree handles escape to R (a subtree is a node range, not a handle).

#include "tree.h"

#include <Rcpp.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "hash.h"

namespace anvl {

// Structural hash of an RTree, consistent with tree_eq (declared in tree.h):
// hashes exactly the fields tree_eq compares -- kinds, child counts, name
// offsets, and child names -- as flat linear scans. The node count seeds the
// hash and the name count precedes the names, so that the flat stream of
// values cannot be regrouped into a different tree.
std::uint64_t tree_hash(const RTree& tree) {
  std::uint64_t h = tree.kind.size();
  for (std::uint8_t k : tree.kind) h = hash_combine(h, k);
  for (std::int32_t c : tree.n_children) h = hash_combine(h, c);
  for (std::int32_t off : tree.name_off) h = hash_combine(h, off);
  h = hash_combine(h, tree.names.size());
  for (const std::string& s : tree.names) {
    h = hash_combine(h, std::hash<std::string>{}(s));
  }
  return h;
}

// Wrap a heap-allocated RTree in an external pointer handed to R. Ownership
// transfers to R: `true` registers a delete finalizer, so ~RTree() runs
// (freeing the arrays) when the handle is garbage-collected.
SEXP tree_xptr(RTree* n) {
  Rcpp::XPtr<RTree> ptr(n, true);  // true: delete `n` on GC
  ptr.attr("class") = "RTree";     // tag with the S3 class R dispatches on
  return ptr;
}

static const char* kind_string(std::uint8_t k) {
  switch (k) {
    case RTree::LeafNode:
      return "leaf";
    case RTree::ListNode:
      return "list";
    default:
      return "null";
  }
}

// Append the subtree rooted at node `c` of `src` onto `dst`, rebasing each
// copied node's name offset into `dst.names` (subtree sizes and child counts
// are position-independent and copy verbatim).
static void append_subtree(RTree& dst, const RTree& src, std::size_t c) {
  const std::size_t span = static_cast<std::size_t>(src.subtree_nodes[c]);
  for (std::size_t q = c; q < c + span; ++q) {
    dst.kind.push_back(src.kind[q]);
    dst.n_children.push_back(src.n_children[q]);
    dst.subtree_nodes.push_back(src.subtree_nodes[q]);
    if (src.is_named(q)) {
      dst.name_off.push_back(static_cast<std::int32_t>(dst.names.size()));
      const std::int32_t off = src.name_off[q];
      for (int k = 0; k < src.n_children[q]; ++k) {
        dst.names.push_back(src.names[off + k]);
      }
    } else {
      dst.name_off.push_back(-1);
    }
  }
}

// The canonical structural string: "*" for a leaf, "NULL" for a null tree,
// "list(a = *, list(*, NULL))" for lists. A named list is tagged
// "list<named>(...)" so the named/unnamed distinction stays visible even when
// the names are empty or the list has no children (individual "" names are
// otherwise printed positionally). `p` advances past the rendered subtree.
static void repr_rec(const RTree& t, std::size_t& p, std::string& out) {
  const std::size_t node = p++;
  switch (t.kind[node]) {
    case RTree::LeafNode:
      out += "*";
      return;
    case RTree::NullNode:
      out += "NULL";
      return;
    case RTree::ListNode: {
      const int n = t.n_children[node];
      const bool named = t.is_named(node);
      const std::int32_t off = named ? t.name_off[node] : 0;
      out += named ? "list<named>(" : "list(";
      for (int k = 0; k < n; ++k) {
        if (k > 0) out += ", ";
        if (named && !t.names[off + k].empty()) {
          out += t.names[off + k];
          out += " = ";
        }
        repr_rec(t, p, out);
      }
      out += ")";
      return;
    }
  }
}

static std::string subtree_repr(const RTree& t, std::size_t p) {
  std::string out;
  repr_rec(t, p, out);
  return out;
}

// The path suffix for child `j` of list node `parent`: "$name" / "name" (at the
// root) for a named slot, "[[j]]" for an unnamed one.
static std::string path_suffix(const RTree& t, std::size_t parent, int j,
                               const std::string& prefix) {
  std::string nm;
  if (t.is_named(parent)) nm = t.names[t.name_off[parent] + j];
  if (!nm.empty()) {
    return prefix.empty() ? nm : "$" + nm;
  }
  return "[[" + std::to_string(j + 1) + "]]";
}

// Find the human-readable path to the leaf with 1-based flat index `target`
// (e.g. "a$b", "l[[2]]"); used for error reporting. Walks the subtree in
// preorder, counting leaves until it reaches `target`.
//
//   t       the tree being walked.
//   p       node cursor: the current node is node `p`, and `p` is advanced past
//           every node visited (so on return it sits past this subtree, or
//           wherever the search stopped once the leaf was found).
//   target  1-based index of the target leaf, in flatten (preorder) order.
//   next    running count of leaves seen, shared by reference across the whole
//           recursion; each leaf visited takes index `++next`.
//   prefix  path accumulated from the root down to the current node ("" at the
//           root).
//   out     set to the target leaf's path on success; left untouched otherwise.
//
// Returns true (and sets `out`) when the target leaf lies in this subtree,
// false otherwise. A leaf at the root has path "".
static bool tree_path_rec(const RTree& t, std::size_t& p, int target, int& next,
                          const std::string& prefix, std::string& out) {
  const std::size_t node = p++;
  switch (t.kind[node]) {
    case RTree::LeafNode:
      if (++next != target) return false;
      out = prefix;
      return true;
    case RTree::NullNode:
      return false;
    case RTree::ListNode: {
      const int n = t.n_children[node];
      for (int k = 0; k < n; ++k) {
        const std::string child_prefix =
            prefix + path_suffix(t, node, k, prefix);
        if (tree_path_rec(t, p, target, next, child_prefix, out)) return true;
      }
      return false;
    }
  }
  return false;
}

// Walk two trees in parallel; report the first structural divergence as the
// path prefix plus the node indices of the two diverging subtrees. `pa`/`pb`
// advance in lockstep.
static bool tree_diff_rec(const RTree& A, std::size_t& pa, const RTree& B,
                          std::size_t& pb, const std::string& prefix,
                          std::string& out_prefix, std::size_t& out_a,
                          std::size_t& out_b) {
  const std::size_t na = pa++;
  const std::size_t nb = pb++;
  auto diverge = [&]() {
    out_prefix = prefix;
    out_a = na;
    out_b = nb;
    return true;
  };
  if (A.kind[na] != B.kind[nb]) return diverge();
  if (A.kind[na] != RTree::ListNode) return false;

  const int an = A.n_children[na];
  const int bn = B.n_children[nb];
  bool names_equal = (A.is_named(na) == B.is_named(nb));
  if (names_equal && A.is_named(na) && an == bn) {
    for (int k = 0; k < an; ++k) {
      if (A.names[A.name_off[na] + k] != B.names[B.name_off[nb] + k]) {
        names_equal = false;
        break;
      }
    }
  }
  if (an != bn || !names_equal) return diverge();
  for (int k = 0; k < an; ++k) {
    const std::string child_prefix = prefix + path_suffix(A, na, k, prefix);
    if (tree_diff_rec(A, pa, B, pb, child_prefix, out_prefix, out_a, out_b)) {
      return true;
    }
  }
  return false;
}

// Recover the RTree behind an external pointer handed back from R. Guards
// against foreign or plain SEXPs: without the type/class/NULL checks below, a
// non-RTree argument (e.g. a PJRTBuffer handle or a bare `1L`) would be
// reinterpreted as an `RTree*` and dereferenced, segfaulting the R session.
static const RTree& as_tree(SEXP handle) {
  if (TYPEOF(handle) != EXTPTRSXP || !Rf_inherits(handle, "RTree")) {
    Rcpp::stop("expected an `RTree` (as returned by `build_tree()`)");
  }
  Rcpp::XPtr<RTree> ptr(handle);
  return *ptr;
}

static void require_list(const RTree& t, const char* what) {
  if (t.kind.empty() || t.kind[0] != RTree::ListNode) {
    Rcpp::stop("%s requires a tree whose root is a list node", what);
  }
}

}  // namespace anvl

// [[Rcpp::export]]
SEXP impl_tree_build(SEXP x) {
  using namespace anvl;
  auto tree = std::make_unique<RTree>();
  std::vector<SEXP> leaves;
  flatten_rec(x, leaves, *tree);
  return tree_xptr(tree.release());
}

// Build the tree and extract the leaves in a single traversal; returns
// `list(tree = <RTree>, leaves = <list>)`. This is the only flatten entry
// point: flatten() keeps just `$leaves`, while map_tree()/pmap_tree() use both
// -- one walk regardless of which is needed.
// [[Rcpp::export]]
Rcpp::List impl_tree_build_flatten(SEXP x) {
  using namespace anvl;
  auto tree = std::make_unique<RTree>();
  std::vector<SEXP> leaves;
  flatten_rec(x, leaves, *tree);
  Rcpp::List out(leaves.size());
  for (std::size_t k = 0; k < leaves.size(); ++k) out[k] = leaves[k];
  return Rcpp::List::create(Rcpp::Named("tree") = tree_xptr(tree.release()),
                            Rcpp::Named("leaves") = out);
}

// [[Rcpp::export]]
SEXP impl_tree_unflatten(SEXP tree, Rcpp::List x) {
  using namespace anvl;
  const RTree& t = as_tree(tree);
  const int need = tree_size_rec(t);
  if (static_cast<int>(x.size()) != need) {
    Rcpp::stop("unflatten(): got %d leaves but the tree has %d",
               static_cast<int>(x.size()), need);
  }
  std::vector<SEXP> leaves(x.begin(), x.end());
  std::size_t p = 0;
  std::size_t li = 0;
  return unflatten_rec(t, leaves, p, li);
}

// [[Rcpp::export]]
int impl_tree_size(SEXP tree) {
  return anvl::tree_size_rec(anvl::as_tree(tree));
}

// [[Rcpp::export]]
bool impl_tree_equal(SEXP a, SEXP b) {
  return anvl::tree_eq(anvl::as_tree(a), anvl::as_tree(b));
}

// The 64-bit hash is returned as a decimal string: R has no native 64-bit
// integer, and the value is only ever compared for (in)equality.
// [[Rcpp::export]]
std::string impl_tree_hash(SEXP tree) {
  return std::to_string(anvl::tree_hash(anvl::as_tree(tree)));
}

// [[Rcpp::export]]
std::string impl_tree_kind(SEXP tree) {
  using namespace anvl;
  return kind_string(as_tree(tree).kind[0]);
}

// [[Rcpp::export]]
SEXP impl_tree_child_names(SEXP tree) {
  using namespace anvl;
  const RTree& t = as_tree(tree);
  if (t.kind[0] != RTree::ListNode || !t.is_named(0)) return R_NilValue;
  const int n = t.n_children[0];
  Rcpp::CharacterVector out(n);
  const std::int32_t off = t.name_off[0];
  for (int k = 0; k < n; ++k) out[k] = t.names[off + k];
  return out;
}

// [[Rcpp::export]]
Rcpp::CharacterVector impl_tree_child_kinds(SEXP tree) {
  using namespace anvl;
  const RTree& t = as_tree(tree);
  require_list(t, "child_kinds()");
  const std::vector<std::size_t> kids = child_nodes(t, 0);
  Rcpp::CharacterVector out(kids.size());
  for (std::size_t k = 0; k < kids.size(); ++k) {
    out[k] = kind_string(t.kind[kids[k]]);
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::IntegerVector impl_tree_child_sizes(SEXP tree) {
  using namespace anvl;
  const RTree& t = as_tree(tree);
  require_list(t, "child_sizes()");
  const std::vector<std::size_t> kids = child_nodes(t, 0);
  Rcpp::IntegerVector out(kids.size());
  for (std::size_t k = 0; k < kids.size(); ++k) {
    out[k] = subtree_leaf_count(t, kids[k]);
  }
  return out;
}

namespace anvl {

std::string tree_path(const RTree& t, int i) {
  std::string out;
  std::size_t p = 0;
  int next = 0;
  if (!tree_path_rec(t, p, i, next, "", out)) {
    Rcpp::stop("tree_path(): no leaf with index %d in the tree", i);
  }
  return out;
}

}  // namespace anvl

// [[Rcpp::export]]
std::string impl_tree_path(SEXP tree, int i) {
  return anvl::tree_path(anvl::as_tree(tree), i);
}

// [[Rcpp::export]]
SEXP impl_tree_filter_by_names(SEXP tree, Rcpp::CharacterVector names) {
  using namespace anvl;
  const RTree& t = as_tree(tree);
  require_list(t, "filter_by_names()");
  if (!t.is_named(0)) Rcpp::stop("filter_by_names(): tree must have names");
  std::unordered_set<std::string> keep(names.size());
  for (R_xlen_t k = 0; k < names.size(); ++k) {
    keep.insert(std::string(names[k]));
  }
  const std::vector<std::size_t> kids = child_nodes(t, 0);

  auto out = std::make_unique<RTree>();
  std::vector<std::size_t> kept;
  std::vector<std::string> kept_names;
  for (std::size_t k = 0; k < kids.size(); ++k) {
    const std::string nm = t.names[t.name_off[0] + k];
    if (keep.count(nm) > 0) {
      kept.push_back(kids[k]);
      kept_names.push_back(nm);
    }
  }
  // Emit the new named root, then its kept children's subtrees. Root names go
  // first (offset 0); append_subtree rebases each child's name offsets after.
  out->kind.push_back(RTree::ListNode);
  out->n_children.push_back(static_cast<std::int32_t>(kept_names.size()));
  out->subtree_nodes.push_back(0);  // backpatched below
  out->name_off.push_back(0);
  for (const std::string& nm : kept_names) out->names.push_back(nm);
  for (std::size_t c : kept) append_subtree(*out, t, c);
  out->subtree_nodes[0] = static_cast<std::int32_t>(out->kind.size());
  return tree_xptr(out.release());
}

// [[Rcpp::export]]
SEXP impl_tree_concat(Rcpp::List children, SEXP names) {
  using namespace anvl;
  // RAII: as_tree() below rejects a non-RTree child by throwing, so `out` must
  // free itself if we never reach tree_xptr() (which transfers it to R's GC).
  const bool has_names = (names != R_NilValue);
  Rcpp::CharacterVector nms;
  if (has_names) {
    nms = Rcpp::CharacterVector(names);
    if (nms.size() != children.size()) {
      Rcpp::stop("tree_concat(): names must match the number of children");
    }
  }
  auto out = std::make_unique<RTree>();
  out->kind.push_back(RTree::ListNode);
  out->n_children.push_back(static_cast<std::int32_t>(children.size()));
  out->subtree_nodes.push_back(0);  // backpatched below
  if (has_names) {
    out->name_off.push_back(0);
    for (R_xlen_t k = 0; k < nms.size(); ++k) {
      if (STRING_ELT(names, k) == NA_STRING) {
        Rcpp::stop("tree_concat(): names must not be NA");
      }
      out->names.push_back(std::string(nms[k]));
    }
  } else {
    out->name_off.push_back(-1);
  }
  for (R_xlen_t k = 0; k < children.size(); ++k) {
    append_subtree(*out, as_tree(children[k]), 0);  // whole child tree
  }
  out->subtree_nodes[0] = static_cast<std::int32_t>(out->kind.size());
  return tree_xptr(out.release());
}

// [[Rcpp::export]]
Rcpp::LogicalVector impl_tree_mask_from_names(SEXP tree,
                                              Rcpp::CharacterVector names) {
  using namespace anvl;
  const RTree& t = as_tree(tree);
  require_list(t, "mask_from_names()");
  std::unordered_set<std::string> marked(names.size());
  for (R_xlen_t k = 0; k < names.size(); ++k) {
    marked.insert(std::string(names[k]));
  }
  const std::vector<std::size_t> kids = child_nodes(t, 0);
  Rcpp::LogicalVector out(tree_size_rec(t));
  R_xlen_t pos = 0;
  for (std::size_t k = 0; k < kids.size(); ++k) {
    const bool m =
        t.is_named(0) && marked.count(t.names[t.name_off[0] + k]) > 0;
    const int sz = subtree_leaf_count(t, kids[k]);
    for (int j = 0; j < sz; ++j) out[pos++] = m;
  }
  return out;
}

// [[Rcpp::export]]
std::string impl_tree_repr(SEXP tree) {
  return anvl::subtree_repr(anvl::as_tree(tree), 0);
}

// [[Rcpp::export]]
SEXP impl_tree_diff(SEXP a, SEXP b) {
  using namespace anvl;
  const RTree& A = as_tree(a);
  const RTree& B = as_tree(b);
  std::string prefix;
  std::size_t pa = 0, pb = 0, sub_a = 0, sub_b = 0;
  if (!tree_diff_rec(A, pa, B, pb, "", prefix, sub_a, sub_b)) {
    return R_NilValue;
  }
  return Rcpp::List::create(Rcpp::Named("prefix") = prefix,
                            Rcpp::Named("a") = subtree_repr(A, sub_a),
                            Rcpp::Named("b") = subtree_repr(B, sub_b));
}
