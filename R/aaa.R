#' @keywords internal
NULL

#' @section Options:
#' * `anvl.backends` (`character()`, default `c("pjrt", "quickr")`): the
#'   backends anvl activates; `"pjrt"` and `"quickr"` are the known ones. This
#'   option is read **once, when the anvl namespace is loaded**; setting it
#'   afterwards has no effect, so set it before `library(anvl)` (e.g. in your
#'   `.Rprofile`):
#'
#'   ```r
#'   options(anvl.backends = "pjrt")
#'   library(anvl)
#'   ```
#'
#'   Requesting a backend that is not active (via [`jit()`], [`local_backend()`],
#'   [`nv_array()`], ...) is an error. The first entry is the initial value of
#'   `anvl.default_backend`, and [`jit()`] infers the backend per call while more
#'   than one backend is active.
#' * `anvl.default_backend` (`character(1)`, default: the first entry of
#'   `anvl.backends`): the concrete backend that arrays are created on when
#'   nothing else names one, see [`default_backend()`]. Change it for a scope
#'   with [`local_backend()`] or [`with_backend()`].
#'
#' @section Environment Variables:
#' * `PJRT_PLATFORM`: the platform [`default_device()`] uses for the `"pjrt"`
#'   backend (e.g. `"cpu"`, `"cuda"`). Defaults to `"cpu"`.
#'
#' @section Third-Party Licenses:
#' The `anvl` package itself is MIT-licensed. The CUDA backend dynamically
#' loads NVIDIA software which is not bundled with `anvl`, but downloaded
#' from NVIDIA's official redistributable channels by the CUDA toolkit R
#' package (e.g. `cuda12.8`) at install time. Its use is governed by the
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
#' @importFrom utils head tail getFromNamespace
#' @importFrom cli cli_abort
#' @importFrom methods formalArgs
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
