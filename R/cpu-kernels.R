#' @useDynLib anvl, .registration = TRUE, .fixes = "C_"
NULL

## CPU kernels as XLA custom calls ---------------------------------------------
#
# A few computations are run on the CPU by anvl kernels (see `src/`), then used
# by XLA as custom calls. These are mostly reserved for parts of a function that
# XLA compiles slowly, with the best candidates being those that already sit
# behind a fusion boundary anyway (such as a while loop): this way, removing
# them from XLA compilation does not lose any fusion opportunity with their
# consumers.
#
# Kernels are written with Google Highway and so automatically compiled for
# every SIMD instruction set of each CPU architecture, with the best one
# available being chosen by runtime dispatch. Multi-core parallelism uses XLA's own intra-op
# thread pool.
#
# Each such computation is a primitive whose StableHLO rule emits the custom
# call when the kernel applies (see `use_cpu_kernel()`), and otherwise lowers
# an R implementation in place (`lower_inline()`), so it runs on any platform.

# Registers anvl's kernels with pjrt, each under the custom-call target name
# that its primitive's StableHLO rule emits. pjrt registers each handler with
# any plugin already loaded, and with the CPU plugin when it loads later. When
# anvl is unloaded, pjrt forgets the handlers, but a plugin that already has
# them keeps them.
register_cpu_kernels <- function(pkgname) {
  handlers <- .Call(C_cpu_kernel_handlers)
  for (target in names(handlers)) {
    pjrt::pjrt_register_custom_call(target, list(cpu = handlers[[target]]), .package = pkgname)
  }
}

# Returns TRUE if a primitive with operand `x` should use a CPU kernel. This
# requires that the primitive is being lowered for the CPU and that `x` has a
# data type the kernels are compiled for (f32 or f64). Setting the option
# `anvl.cpu_kernels = FALSE` disables the kernels, which is useful for comparing
# them against the lowering used on other platforms.
# The option is read when lowering, so it only affects functions compiled
# after it is set.
use_cpu_kernel <- function(x) {
  identical(current_platform(), "cpu") &&
    as.character(dtype(x)) %in% c("f32", "f64") &&
    !isFALSE(getOption("anvl.cpu_kernels"))
}

# Emits a custom call to anvl's CPU kernel `target`, applied to the FuncValues
# `operands`, with result types `output_types`. The kernels treat each buffer
# as a flat array and index all of them alike, so every operand and the result
# must share one layout. They are given row-major, XLA's default, so normally
# no copies need to be inserted.
cpu_custom_call <- function(target, operands, output_types) {
  row_major <- function(x) rev(seq_along(shape(x)) - 1L)
  rlang::exec(
    hlo_custom_call,
    !!!unname(operands),
    call_target_name = target,
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = output_types,
    operand_layouts = lapply(operands, \(x) row_major(x$value_type)),
    result_layouts = lapply(output_types, row_major)
  )
}

# Lowers `fn`, an R function built from anvl operations, straight into the
# function we are currently building. `operands` is a list of FuncValues, named
# after `fn`'s arguments. We trace `fn` at the types of those operands, then
# emit its operations in place. Returns one FuncValue for each output of `fn`.
lower_inline <- function(fn, operands) {
  avals <- lapply(operands, \(x) nv_aval(dtype(x), r_shape(x)))
  # We run the same graph passes as jit(). Among other things, they turn scalar
  # constants, such as those made by nv_scalar(), into literals.
  graph <- optimize_graph(trace_fn(fn, avals), optimize = TRUE)
  if (length(graph$constants)) {
    cli_abort("Internal error: a function lowered in place must not capture arrays.")
  }
  env <- HloEnv()
  for (i in seq_along(graph$inputs)) {
    env_add(env, graph$inputs[[i]], operands[[i]])
  }
  lower_graph_calls(graph, env, stablehlo::.current_func())
}
