#' @include graph.R
#' @include promotion.R
NULL

# The RData layer: an R value that has entered a program but has not taken a
# data type yet. `RData` is its aval, `materialize_rdata()` / `build_r_at()`
# build it where it is used, and `finalize_rdata_inputs()` settles the single
# data type a jitted function's R argument is uploaded at.

#' @title R Data Class
#' @description
#' The [`AbstractArray`] of a non-static R value that was passed to a jit-compiled
#' function.
#' It is special, because it does not have a data type, i.e., calling `dtype()` on it
#' results in an error.
#' @details
#' It only exists *during* tracing, never in a finalized graph.
#' @section Extractors:
#' Just like [`AbstractArray`], with the exception that `dtype()` errs.
#'
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the value: `()` for a length-1 vector, its `dim()` for an
#'   R array.
#' @param r_type (`character(1)`)\cr
#'   The R storage type: `"double"`, `"integer"` or `"logical"`.
#'
#' @return (`RData`)
#' @seealso [AbstractArray], [AnvlGraph]
#' @examplesIf pjrt::plugins_downloaded()
#' RData(c(2, 3), "double")
#' # is equivalent to
#' nv_aval("double", c(2, 3))
#' # Below, the `RData` input is materialized in 32 and 64-bit precisions, so the input
#' # dtype becomes f64.
#' # By NOT converting RData to their default data type we prevent loss of precision
#' # (double -> f32 -> f64 roundrips)
#' graph <- trace_fn(function(x) {
#'     print(x)
#'     list(x + nv_scalar(1, "f64"), x + nv_scalar(1, "f32"))
#'   }, list(x = nv_aval("double", c()))
#' )
#' print(graph)
#' # The actual inputs to the compiled program
#' graph$inputs
#' # The data types of the R values; AnvlArrays get NA here
#' graph$rdata_types
#' @export
RData <- function(shape, r_type) {
  shape <- as_shape(shape)
  r_type <- match.arg(r_type, c("double", "integer", "logical"))
  structure(
    # Deliberately no `dtype`: the value has none until it commits, and the
    # default it would commit to depends on the backend in force
    # (`default_dtype_r()`), so code that reaches for `$dtype` must not
    # silently get one.
    list(r_type = r_type, shape = shape),
    class = c("RData", "AbstractArray")
  )
}

is_rdata <- function(x) {
  inherits(x, "RData")
}

#' @method dtype RData
#' @export
dtype.RData <- function(x, ...) {
  abort_no_dtype(default_dtype_r(x$r_type))
}


#' @method shape numeric
#' @export
shape.numeric <- function(x, ...) {
  if (!is.null(dim(x))) {
    return(as.integer(dim(x)))
  }
  if (length(x) != 1L) {
    cli_abort(c(
      "{.fn shape} is undefined for a length-{length(x)} R vector.",
      i = "Only a length-1 R value and an {.fn array} are arrayish; use {.fn nv_array} to make one an array."
    ))
  }
  integer()
}

#' @method shape logical
#' @export
shape.logical <- shape.numeric

#' @method dtype numeric
#' @export
dtype.numeric <- function(x, ...) {
  abort_no_dtype(default_dtype(x))
}

#' @method dtype logical
#' @export
dtype.logical <- function(x, ...) {
  abort_no_dtype(default_dtype(x))
}

abort_no_dtype <- function(default_dtype) {
  cli_abort(
    c(
      "An R value has no data type of its own until it is used.",
      i = "{.fn dtype} is undefined here for the same reason {.code dtype(1.5)} is: the value only takes a data type when it meets a typed array, or when it commits to the default ({.val {as.character(default_dtype)}}).", # nolint
      i = "Give it one explicitly with {.fn nv_convert}."
    ),
    call = NULL
  )
}

#' @export
format.RData <- function(x, ...) {
  sprintf("RData(%s, %s)", x$r_type, shape2string(x$shape))
}

#' @export
print.RData <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
repr.RData <- function(x, ...) {
  sprintf("%s[%s]", x$r_type, repr(x$shape))
}

is_rdata_box <- function(x) {
  inherits(x, "GraphBox") && is_rdata(x$gnode$aval)
}

# The values the trace built an open R argument at, keyed by dtype name. Lives
# on the descriptor that registered the input rather than on the value, so a
# GraphValue stays a value.
rdata_mat <- function(desc, gval) {
  desc$rdata_mat[[gval]] %||% list()
}

rdata_mat_set <- function(desc, gval, key, box) {
  mat <- rdata_mat(desc, gval)
  mat[[key]] <- box
  desc$rdata_mat[[gval]] <- mat
  invisible(box)
}

rdata_natural_dtype <- function(r_type) {
  switch(r_type, double = as_dtype("f64"), integer = as_dtype("i32"), logical = as_dtype("bool"))
}

# The dtype an R value is built at on its way to `dtype`, for a `dtype` it
# cannot be built at directly. Building it where it is exact is what keeps the
# conversion the program's rather than R's -- but where that is wider than the
# dtype the value would commit to on its own, the program acquires one nothing
# asked for. An R double reaches `i32` through `f64`, so a program with no `f64`
# in it gets one, and a backend without `f64` (Metal) cannot run it at all.
#
# That is worth saying out loud rather than refusing: the alternative rounds the
# value through its default before converting, which is a different answer
# (`2^25 + 1` reaches `i32` as 33554432 rather than 33554433), and the caller may
# well want the exact one. `double` is the only R type this can happen for --
# an integer and a logical are exact at their own defaults.

# This is one of THE core functios of the RData mechanism, as it specifies how
# we obtain RData inputs at the requested data type.
# When an RData input is only requested at it's category, we use the dtype within that category
# that holds enough precision so everybody gets the value at its requested dtype without losing precision.
# To understand this, consider a program that receives an R double and compare two programs:
# 1. a single primitive call uses the R double and requires it in f32 -> input becomes f32
# 2. one primitive requires f32, the other f64. -> input becomes f64
#    this only works, because the f32 obtained from the f32 is the same as we obtain from the f64.
# Where we do cross-category concversions, a double input must e.g. always be materialized at
# f64, otherwise the result of an i32 request might depend on the float requests of that double,
# which MUST NOT HAPPEN. This is also why prim_convert(1, "i32") errs:
# We need to introduce an f64 into the program although nobody ever requested one, but otherwise
# would be even worse.
rdata_staging_dtype <- function(r_type, dtype) {
  staged <- rdata_natural_dtype(r_type)
  if (staged != default_dtype_r(r_type)) {
    cli_warn(
      c(
        "Converting an R {r_type} to {.val {as.character(dtype)}} brings {.val {as.character(staged)}} into the program.", # nolint
        x = "An R {r_type} cannot be built at {.val {as.character(dtype)}} directly, so it is built at {.val {as.character(staged)}} and the program converts -- and nothing else here asked for {.val {as.character(staged)}}.", # nolint
        i = "To keep it out, convert in its own category first: {.code nv_convert(nv_convert(x, {.str {as.character(default_dtype_r(r_type))}}), {.str {as.character(dtype)}})}. The result differs for values its data type cannot hold exactly." # nolint
      ),
      class = "anvl_staging_widens_warning"
    )
  }
  staged
}

rdata_in_category <- function(r_type, dtype) {
  dtype_category(dtype) == rdata_category(r_type)
}

# The category (see `dtype_category()`) of an R storage type. Fixed, whatever
# width the default dtype of that type happens to have.
rdata_category <- function(r_type) {
  switch(r_type, double = 3L, integer = 2L, logical = 1L)
}

# TODO: bit64 support
rdata_builds_directly <- function(r_type, dtype) {
  rdata_in_category(r_type, dtype) &&
    (r_type != "integer" || (is_dtype_int(dtype) && dtype_width(dtype) >= 32L))
}

# Bring an R value of storage type `r_type` into the program at `dtype`. `build`
# makes it at a data type it can be built at faithfully; a target it cannot is
# reached by building at the natural one and letting the *program* convert the
# rest of the way, so narrowing follows XLA's semantics rather than R's.
#
# The three ways an R value enters a program -- a literal in a traced body, the
# input an open argument is supplied at, an array built eagerly -- differ only
# in `build`, and this is what they share.
build_r_staged <- function(r_type, dtype, build) {
  if (rdata_builds_directly(r_type, dtype)) {
    return(build(dtype))
  }
  prim_convert(build(rdata_staging_dtype(r_type, dtype)), dtype = dtype)
}

# Build `box`'s R argument into the graph at `dtype`, and return the GraphBox
# for it. Memoized per dtype on the node: an argument used twice at one dtype
# takes one input.
materialize_rdata <- function(box, dtype) {
  hit <- rdata_mat_hit(box, dtype)
  if (!is.null(hit)) {
    return(hit)
  }
  out <- build_r_staged(box$gnode$aval$r_type, dtype, function(dt) rdata_input_at(box, dt))
  rdata_mat_set(box$desc, box$gnode, as.character(dtype), out)
}

# The input `box`'s R argument is supplied at, for a dtype it can be uploaded at
# directly. Its value is unknown here (the compiled program must not depend on
# it), so it becomes an input of this dtype and the call uploads the R data at
# it; `finalize_rdata_inputs()` puts it in the input list. Memoized like the
# converts above it, so a staged build and a later request for the same natural
# dtype share one input.
rdata_input_at <- function(box, dtype) {
  hit <- rdata_mat_hit(box, dtype)
  if (!is.null(hit)) {
    return(hit)
  }
  aval <- box$gnode$aval
  rdata_mat_set(
    box$desc,
    box$gnode,
    as.character(dtype),
    register_gval(box$desc, GraphValue(AbstractArray(dtype = dtype, shape = aval$shape)))
  )
}

# A memoized box is only reusable where it can actually be reached. One built in
# the node's own descriptor can: a sub-graph captures it like any other outer
# value. A `prim_convert` is recorded in whatever descriptor was current,
# though, so a sibling sub-graph -- `prim_if()`'s other branch, `prim_while()`'s
# body after its condition -- has to build its own rather than reference a value
# only the first one computes.
rdata_mat_hit <- function(box, dtype) {
  hit <- rdata_mat(box$desc, box$gnode)[[as.character(dtype)]]
  if (is.null(hit)) {
    return(NULL)
  }
  reachable <- identical(hit$desc, box$desc) || identical(hit$desc, .current_descriptor(silent = TRUE))
  if (reachable) hit else NULL
}

# Build the bare R value `x` -- an actual length-1 vector or array -- into
# `desc` at `dtype` as a constant, and return its GraphBox. Tracing only. The
# value is built from the R data itself, so it arrives with every digit it had,
# which is what keeps `x_f64 / sqrt(2)` exact.
build_r_at <- function(x, dtype, desc = .current_descriptor()) {
  force(desc)
  if (!is_valid_r_lit(x) && !is_valid_r_array(x)) {
    # An `NA` reaches here: it is a length-1 numeric, but there is no dtype it
    # obviously belongs at. Refused the same way it is anywhere else a value
    # enters a graph.
    cli_abort("Expected arrayish value, but got {.cls {class(x)[1]}}")
  }
  build_r_staged(typeof(x), as_dtype(dtype), function(dt) r_const_at(x, dt, desc))
}

# The graph's constant for the R value `x` at `dtype`: an inlined literal for a
# scalar, a registered constant for an array. Deduplicated per descriptor, so
# the same value written twice is built once.
r_const_at <- function(x, dtype, desc) {
  if (is.null(dim(x))) {
    return(get_box_or_register_const(
      desc,
      GraphLiteral(LiteralArray(x, shape = shape(x), dtype = dtype))
    ))
  }
  get_box_or_register_const(desc, nv_array(x, dtype = dtype))
}

#' @title Peek at a Data Type
#' @description
#' The data type `x` would use if it was converted to an `AnvlArray`.
#' Relevant for R objects and their [`RData`] trace-time analogon: for those it
#' is the default of the backend in force (see [`default_dtypes()`]), which the
#' value has not committed to yet.
#'
#' @param x ([`arrayish`] | [`AbstractArray`])\cr
#'   The value to ask about.
#' @return ([`tengen::DataType`])
#' @seealso [as_anvl_arrays()], [RData], [shape()][tengen::shape]
#' @examplesIf pjrt::plugins_downloaded()
#' peek_dtype(1.5)
#' peek_dtype(1L)
#' peek_dtype(nv_array(1:3, dtype = "i8"))
#' @export
peek_dtype <- function(x) {
  aval <- to_abstract(x)
  if (is_rdata(aval)) default_dtype_r(aval$r_type) else aval$dtype
}

# A traced box, with any R value in it committed to its default dtype. Anything
# that already has a dtype is returned unchanged.
commit_rdata_box <- function(x) {
  if (is_rdata_box(x)) {
    materialize_rdata(x, peek_dtype(x))
  } else {
    x
  }
}


# Whether a value has no dtype yet -- an RData box, or a bare R value that has
# not been boxed. Deliberately cheap: it runs once per argument of every traced
# primitive call.
has_no_dtype <- function(x) {
  is_rdata_box(x) || is_valid_r(x)
}

# Input slots -----------------------------------------------------------------

# Reserve `desc`'s next input slot for an R argument: an input like any other,
# except that its aval is an `RData` and so has no data type yet. The slot holds
# it until finalize_rdata_inputs() (or, for an inline trace,
# finalize_inline_rdata_inputs()) replaces it with the value the body
# materialized, which keeps the input order the same as the argument order (the
# caller supplies its inputs in that order).
#
# `outer` links an inline trace's input to the enclosing trace's box the
# argument came from. The value is fresh rather than shared so the inline body's
# materializations land in `desc` -- where transform_gradient() can
# differentiate the converts between them -- and cannot clobber the outer
# input's memo.
register_rdata_input <- function(desc, aval, outer = NULL) {
  gval <- GraphValue(aval)
  desc$inputs <- c(desc$inputs, list(gval))
  if (!is.null(outer)) {
    desc$rdata_outer[[gval]] <- outer
  }
  GraphBox(gval, desc)
}

# Which of a descriptor's inputs are R arguments whose data type is still open.
is_open_rdata_input <- function(gval) {
  is_rdata(gval$aval)
}

# modifies the descriptor in-place
# and finalizes the rdata inputs by:
# 1. Resolving the data type the input gets
# 2. Appending pre_calls with the conversions.
finalize_rdata_inputs <- function(desc) {
  inputs <- desc$inputs
  is_open <- vapply(inputs, is_open_rdata_input, logical(1L))
  if (!any(is_open)) {
    return(invisible(NULL))
  }
  r_types <- rep(NA_character_, length(inputs))
  pre_calls <- list()
  for (i in which(is_open)) {
    gval <- inputs[[i]]
    aval <- gval$aval
    mat <- rdata_mat(desc, gval)
    requested <- rdata_requested_dtypes(aval, mat)
    resolved <- resolve_upload_dtype(aval, requested, desc$default_dtypes)
    main <- mat[[resolved]] %||%
      GraphBox(GraphValue(AbstractArray(resolved, aval$shape)), desc)
    inputs[[i]] <- main$gnode
    r_types[[i]] <- aval$r_type
    for (other in setdiff(requested, resolved)) {
      # The invariance we need to uphold is we resolve the inputs in such a way, that this convert
      # always results in the same value
      pre_calls[[length(pre_calls) + 1L]] <- PrimitiveCall(
        primitive = prim_convert,
        inputs = list(main$gnode),
        params = list(dtype = as_dtype(other)),
        outputs = list(mat[[other]]$gnode)
      )
    }
  }
  desc$inputs <- inputs
  desc$rdata_types <- r_types
  desc$pre_calls <- c(desc$pre_calls, pre_calls)
  invisible(NULL)
}

# The dtypes a finished trace built `node`'s R value at directly. Only these
# can be uploaded (or serve as a graph input); the memo also holds the results
# of converting out of the value's category, which the program computes from
# one of these.
rdata_requested_dtypes <- function(aval, mat) {
  Filter(
    function(dt) rdata_builds_directly(aval$r_type, as_dtype(dt)),
    names(mat)
  )
}

# Finalize Rdata inputs of a sub-trace.
# E.g. used with gradent()
finalize_inline_rdata_inputs <- function(desc) {
  inputs <- desc$inputs
  is_open <- vapply(inputs, is_open_rdata_input, logical(1L))
  if (!any(is_open)) {
    return(invisible(NULL))
  }
  pre_calls <- list()
  add_convert <- function(input, dtype, output) {
    # The invariance we need to uphold is we resolve the inputs in such a way, that this convert
    # always results in the same value
    pre_calls[[length(pre_calls) + 1L]] <<- PrimitiveCall(
      primitive = prim_convert,
      inputs = list(input),
      params = list(dtype = as_dtype(dtype)),
      outputs = list(output)
    )
  }
  for (i in which(is_open)) {
    gval <- inputs[[i]]
    aval <- gval$aval
    mat <- rdata_mat(desc, gval)
    requested <- rdata_requested_dtypes(aval, mat)
    resolved <- resolve_upload_dtype(aval, requested, desc$default_dtypes)
    outer <- desc$rdata_outer[[gval]]
    local_box <- mat[[resolved]]
    # In the examples below, `x` is `nv_scalar(1, "f64")` and `b` is the R
    # value this iteration settles.
    if (is.null(local_box)) {
      # The sub-trace never built at the resolved data type
      # (e.g., if sub-trace uses input 1 at f16 and bf16, resolved would be f32, i.e. the one that
      # can hold both, even though nobody requested it)
      main <- materialize_rdata(outer, as_dtype(resolved))$gnode
    } else if (is.null(rdata_mat(outer$desc, outer$gnode)[[resolved]])) {
      # The value at `resolved` exists in this graph and nowhere else: hand
      # the gval itself up, so the enclosing trace's own finalize defines it
      # (as the upload input, or a convert from it) ahead of this graph's
      # calls, and later uses there reuse it.
      # e.g. the body builds `q` at `f64`, the enclosing trace never uses `b`:
      #   jit(\(a, b) gradient(\(p, q) p * q, wrt = "p")(a, b))(x, 2)
      main <- local_box$gnode
      rdata_mat_set(outer$desc, outer$gnode, resolved, register_gval(outer$desc, main))
    } else {
      # The enclosing trace built `resolved` too, so the body's gval is a
      # second value for it: take the outer one as the input and define the
      # body's from it.
      # e.g. `a + b` builds `b` at `f64` before the body builds its own:
      #   jit(\(a, b) { w <- a + b; gradient(\(p, q) p * q, wrt = "p")(a, b) })(x, 2)
      main <- materialize_rdata(outer, as_dtype(resolved))$gnode
      add_convert(main, resolved, local_box$gnode)
    }
    inputs[[i]] <- main
    for (other in setdiff(requested, resolved)) {
      add_convert(main, other, mat[[other]]$gnode)
    }
  }
  desc$inputs <- inputs
  desc$pre_calls <- c(desc$pre_calls, pre_calls)
  invisible(NULL)
}

# What a dtype can hold, as (precision, range): a float's mantissa and exponent
# width, an integer's width and whether it is signed -- an R integer is signed,
# so an unsigned dtype of the same width holds less of it. Used only to compare
# dtypes of one category, which is the only comparison that means anything.
dtype_capacity <- function(dtype) {
  switch(
    as.character(dtype),
    f64 = c(52L, 11L),
    f32 = c(23L, 8L),
    f16 = c(10L, 5L),
    bf16 = c(7L, 8L),
    c(dtype_width(dtype), if (is_dtype_int(dtype)) 1L else 0L)
  )
}

# Whether building an R value at `dtype` and converting to `other` gives what
# building it at `other` directly would -- i.e. whether `dtype` holds every
# value `other` can express.
dtype_holds <- function(dtype, other) {
  all(dtype_capacity(dtype) >= dtype_capacity(other))
}

# The dtypes an R value of this storage type can be built at, narrowest first.
# `resolve_upload_dtype()` searches these when no dtype the trace asked for holds
# the others, so the upload widens only as far as it must.
rdata_build_candidates <- function(r_type) {
  switch(
    r_type,
    double = c("f16", "bf16", "f32", "f64"),
    # An R integer is signed and is not built below 32 bits
    # (`rdata_builds_directly()`), so these are all of them.
    integer = c("i32", "i64"),
    logical = "bool",
    cli_abort("No build candidates for R type {.val {r_type}}")
  )
}

# The single dtype an R argument is uploaded at, given every dtype the trace
# built it at. Those are all in the value's own category (materialize_rdata()
# converts out of it inside the program instead), so one of them can serve every
# use site: the upload has to *hold* them all, and each site then converts down
# from it, rounding exactly once.
#
# Not simply the widest. `f16` and `bf16` are both 16 bits and neither holds the
# other -- `f16` has three more mantissa bits, `bf16` a far wider exponent. When
# no dtype the trace asked for holds them all, the narrowest one of the value's
# category that does is used (`f32` for those two), so a program with no `f64`
# in it does not acquire one here.
resolve_upload_dtype <- function(aval, requested, defaults = current_default_dtypes()) {
  if (!length(requested)) {
    return(as.character(default_dtype_r(aval$r_type, defaults)))
  }
  dtypes <- lapply(requested, as_dtype)
  holds_all <- vapply(dtypes, function(d) all(vapply(dtypes, dtype_holds, logical(1L), dtype = d)), logical(1L))
  if (!any(holds_all)) {
    # None of them holds the others, so the upload has to widen -- but only as
    # far as it takes. The natural dtype would always do (`f64` holds every
    # float), and it is what the value would be built at on its own; taking it
    # here would put an `f64` input into a program that has no other `f64` in
    # it, purely because one argument was asked for at two narrow floats.
    holds <- function(cand) all(vapply(dtypes, dtype_holds, logical(1L), dtype = as_dtype(cand)))
    narrowest <- Find(holds, rdata_build_candidates(aval$r_type))
    return(narrowest %||% as.character(rdata_natural_dtype(aval$r_type)))
  }
  # Several candidates hold them all only when they are equivalent; ordering by
  # name keeps the choice independent of the order the trace happened to ask in.
  names <- vapply(dtypes[holds_all], as.character, character(1L))
  sort(names)[[1L]]
}

# The data type each of a finished graph's inputs is uploaded at, for the
# inputs supplied as bare R data -- `NA` for one the caller passes through as an
# array. pjrt's dispatcher and the quickr wrapper read it to know what to make
# of the R value before handing it over.
graph_input_dtypes <- function(graph) {
  r_types <- graph$rdata_types
  if (is.null(r_types) || !any(!is.na(r_types))) {
    return(NULL)
  }
  ifelse(
    !is.na(r_types),
    vapply(graph$inputs, function(gval) as.character(gval$aval$dtype), character(1L)),
    NA_character_
  )
}
