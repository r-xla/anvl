#' @title Get the shape of an array
#'
#' @description Returns the shape of an array as an `integer()` vector.
#'
#' @details This is implemented via the generic [`tengen::shape()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns `integer()`
#' @name shape
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' shape(x)
NULL

#' @rdname shape
#' @importFrom tengen shape
#' @export
tengen::shape

#' @title Get the device of an array
#'
#' @description Returns the device on which an array is allocated.
#'
#' @details This is implemented via the generic [`tengen::device()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns [`PJRTDevice`][pjrt::pjrt_device]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' device(x)
#' @name device
NULL

#' @rdname device
#' @importFrom tengen device
#' @export
tengen::device

#' @title Convert to an R array
#'
#' @description
#' Transfers array data to R and returns it as an R [`array`].
#' Only in the case of scalars is the result a vector of length 1, as R `arrays` cannot have 0 axes.
#'
#' @details
#' This is implemented via the generic [`tengen::as_array()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns An R [`array`] or `vector` of length 1.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' as_array(x)
#' y <- nv_scalar(1L)
#' # R arrays can't have 0 axes:
#' as_array(y)
#' @name as_array
NULL

#' @rdname as_array
#' @importFrom tengen as_array
#' @export
tengen::as_array

#' @title Convert an array to a raw vector
#'
#' @description Returns the underlying bytes of an array as a [raw] vector.
#'
#' @details This is implemented via the generic [`tengen::as_raw()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to method:
#'   - `row_major` (`logical(1)`)\cr
#'     Whether to write the bytes in row-major order.
#' @returns A [`raw`] vector.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, shape = c(2, 2), dtype = "f32")
#' as_raw(x, row_major = TRUE)
#' as_raw(x, row_major = FALSE)
#' @name as_raw
NULL

#' @rdname as_raw
#' @importFrom tengen as_raw
#' @export
tengen::as_raw

#' @title Get the data type of an array
#'
#' @description
#' Returns the data type of an array (e.g. `f32`, `i64`).
#'
#' @details This is implemented via the generic [`tengen::dtype()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns A [`DataType`][tengen::DataType].
#' @seealso [tengen::dtype()]
#' @name dtype
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' dtype(x)
NULL

#' @rdname dtype
#' @importFrom tengen dtype
#' @export
tengen::dtype

#' @title Get the number of axes of an array
#'
#' @description Returns the number of axes (sometimes also refered to as rank) of an array.
#' Equivalent to `length(shape(x))`.
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @returns `integer(1)`
#' @seealso [tengen::naxes()]
#' @name naxes
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' naxes(x)
NULL

#' @rdname naxes
#' @importFrom tengen naxes
#' @export
tengen::naxes

#' @title Get the axes of an array
#'
#' @description Returns the axis indices of an array, i.e. `seq_len(naxes(x))`.
#' For a `20x5x3` array, the axes are `1`, `2` and `3`, while its dimensions
#' are `20`, `5` and `3`.
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @returns `integer()`
#' @seealso [tengen::axes()], [naxes()]
#' @name axes
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(array(1:24, dim = c(2, 3, 4)), dtype = "f32")
#' axes(x)
NULL

#' @rdname axes
#' @importFrom tengen axes
#' @export
tengen::axes

#' @title Check if an object is a DataType
#'
#' @description Tests whether `x` is a `DataType` object.
#'
#' @param x An object to test.
#' @returns `TRUE` or `FALSE`.
#' @seealso [as_dtype()], [tengen::is_dtype()]
#' @name is_dtype
#' @examples
#' is_dtype("f32")
#' is_dtype(as_dtype("f32"))
NULL

#' @rdname is_dtype
#' @importFrom tengen is_dtype
#' @export
tengen::is_dtype

#' @title Convert to a DataType
#'
#' @description Coerces a value to a `DataType`. Accepts data type strings
#' (e.g. `"f32"`, `"i64"`, `"bool"`) or existing `DataType` objects (they are returned unchanged).
#'
#' @details
#' This is implemented via the generic [`tengen::as_dtype()`].
#'
#' @param x A character string or `DataType` to convert.
#' @returns A `DataType` object.
#' @seealso [is_dtype()], [tengen::as_dtype()], [`tengen::DataType`]
#' @name as_dtype
#'
#' @examplesIf pjrt::plugins_downloaded()
#' as_dtype("f32")
#' as_dtype("i32")
NULL

#' @rdname as_dtype
#' @importFrom tengen as_dtype
#' @export
tengen::as_dtype

#' @title Create a Shape object
#'
#' @description Constructs a `Shape` representing array dimensions.
#'
#' @param dims An `integer()` vector of dimension sizes (>= 0).
#' @returns A `Shape` object.
#' @seealso [shape()], [stablehlo::Shape()]
#' @name Shape
#' @rdname Shape-constructor
#' @examples
#' Shape(c(2L, 3L))
NULL

#' @rdname Shape-constructor
#' @importFrom stablehlo Shape
#' @export
stablehlo::Shape

#' @title Get the platform of an array or buffer
#'
#' @description
#' Returns the platform name (e.g. `"cpu"`, `"cuda"`) identifying
#' the compute backend.
#'
#' @details
#' Implemented via the generic [`pjrt::platform()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns `character(1)`
#' @seealso [pjrt::platform()]
#' @name platform
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' platform(x)
NULL

#' @rdname platform
#' @importFrom pjrt platform
#' @export
pjrt::platform

#' @title Block until an async operation completes
#'
#' @description
#' Block until the array's underlying computation has finished, and return the
#' object invisibly. Useful for benchmarking, where the dispatch of an
#' asynchronous operation should not be confused with its execution.
#'
#' @details
#' Implemented via the generic [`pjrt::await()`]. For backends without
#' asynchronous execution (e.g. `"quickr"`), this is a no-op.
#'
#' @param x ([`AnvlArray`] or other awaitable)\cr
#'   An object with an [`await()`] method.
#' @param ... Additional arguments passed to methods (unused).
#' @returns `x`, invisibly.
#' @seealso [pjrt::await()], [map_tree()] (to await a tree of outputs)
#' @name await
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' await(x)
#'
#' # Await all leaves of a (possibly nested) list of arrays.
#' map_tree(list(x, list(y = x)), await)
NULL

#' @rdname await
#' @importFrom pjrt await
#' @export
pjrt::await

#' @importFrom pjrt flatten
#' @export
pjrt::flatten

#' @importFrom pjrt build_tree
#' @export
pjrt::build_tree

#' @importFrom pjrt unflatten
#' @export
pjrt::unflatten

#' @importFrom pjrt tree_size
#' @export
pjrt::tree_size

#' @importFrom pjrt tree_path
#' @export
pjrt::tree_path

#' @importFrom pjrt map_tree
#' @export
pjrt::map_tree

#' @importFrom pjrt pmap_tree
#' @export
pjrt::pmap_tree
