#' @keywords internal
NULL

#' @section Options:
#' * `anvl.backend` (`character(1)`, default `"pjrt"`): the backend every
#'   operation runs on -- `"pjrt"` or `"quickr"`. Arrays are allocated with it
#'   and jitted functions are compiled for it.
#'   Also see  [`active_backend()`], [`local_backend()`] and [`with_backend()`].
#' * `anvl.default_dtypes` (named `character()` | named `list()`): the data
#'   types an R double and integer materialize at when it cannot be inferred from
#'   another operand.
#'   See [`default_dtypes()`] for more details.
#'
#' @section Environment variables:
#' * `PJRT_PLATFORM`: the platform the `"pjrt"` backend allocates on and
#'   compiles for when a call names no device -- `"cpu"` (the default),
#'   `"cuda"`, `"metal"`, ... It is read afresh whenever a default device is
#'   needed; see [`default_device()`] and [`nv_device()`]. The variable is
#'   pjrt's, anvl only follows it.
#'
#' The remaining ones affect only anvl's own test suite, not the package:
#'
#' * `ANVL_TEST`: `tests/testthat.R` runs the tests only when this is `"1"`,
#'   so `R CMD check` in a shell without it runs none of them.
#' * `ANVL_SKIP_QUICKR`: when set to anything non-empty, the tests that need
#'   the quickr backend are skipped -- they are comparatively slow.
#' * `ANVL_DEFAULT_DTYPES`: `category=dtype` pairs such as
#'   `"float=f64,int=i64"`, which the test setup turns into the
#'   `anvl.default_dtypes` option for the whole run, so that anything
#'   hardcoding `f32` / `i32` where it should read [`default_dtypes()`] fails.
#'
#' @section Third-Party Licenses:
#' The `anvl` package itself is MIT-licensed. The CUDA backend dynamically
#' loads NVIDIA software which is not bundled with `anvl`, but downloaded
#' from NVIDIA's official redistributable channels by the CUDA toolkit R
#' package (e.g. `pjrt.cuda`) at install time. Its use is governed by the
#' [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), with the
#' exception of cuDNN, which is covered by the
#' [NVIDIA cuDNN SLA](https://docs.nvidia.com/deeplearning/cudnn/sla/index.html),
#' and NCCL, which is covered by its [own license](https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt).
#' By installing or using the CUDA backend you accept those terms.
"_PACKAGE"

## usethis namespace: start
#' @importFrom stablehlo repr Shape FuncId Func FuncValue
#' @importFrom stablehlo local_func TensorType
# `hlo_scalar` and `hlo_tensor` are imported statically because we
# register S3 methods for them
#' @importFrom stablehlo hlo_scalar hlo_tensor
#' @evalNamespace paste0("importFrom(stablehlo,", setdiff(grep("^hlo_", getNamespaceExports("stablehlo"), value = TRUE), c("hlo_scalar", "hlo_tensor")), ")")
#' @import checkmate
#' @import tengen
#' @importFrom pjrt pjrt_buffer pjrt_scalar pjrt_execute pjrt_compile pjrt_program elt_type
#' @importFrom utils gethash hashtab maphash numhash
#' @importFrom xlamisc seq_len0 seq_along0
#' @importFrom utils head tail getFromNamespace install.packages
#' @importFrom cli cli_abort cli_warn
#' @importFrom rlang %||%
#' @importFrom methods formalArgs is
#' @importFrom utils capture.output
#' @importFrom stats median setNames
## usethis namespace: end
NULL

globals <- new.env()
globals$nv_types <- "AnvlArray"
globals$interpretation_rules <- c("stablehlo", "quickr", "reverse")
globals[["DESCRIPTOR_STASH"]] <- list()
globals[["CURRENT_DESCRIPTOR"]] <- NULL
globals[["LOWERING_PLATFORM"]] <- NULL
utils::globalVariables(c("globals", "self"))
