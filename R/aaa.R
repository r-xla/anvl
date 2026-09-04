#' @keywords internal
NULL

#' @section Options:
#' * `anvl.backend` (`character(1)`, default `"pjrt"`): the backend in force.
#'   Every array is built on it and every jitted function runs on it. See
#'   [`default_backend()`], [`local_backend()`] and [`with_backend()`].
#' * `anvl.default_float` (`"f32"` | `"f64"`) and `anvl.default_int` (`"i32"` |
#'   `"i64"`): the data types an R double and an R integer commit to when
#'   nothing else decides one. Unset by default, in which case the backend in
#'   force decides. See [`default_dtypes()`], [`local_default_dtypes()`] and
#'   [`with_default_dtypes()`].
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
#' @importFrom stats median
## usethis namespace: end
NULL

globals <- new.env()
globals$nv_types <- "AnvlArray"
globals$interpretation_rules <- c("stablehlo", "quickr", "reverse")
globals[["DESCRIPTOR_STASH"]] <- list()
globals[["CURRENT_DESCRIPTOR"]] <- NULL
globals[["LOWERING_PLATFORM"]] <- NULL
utils::globalVariables(c("globals", "self"))
