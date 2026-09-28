#' @title Get the Shape of an Array
#'
#' @description Returns the shape of an array as an `integer()` vector.
#'
#' @details
#' An R value has a shape when it is arrayish: a length-1 vector is a scalar
#' with shape `integer()`, and an R array has its `dim()`. Any other R vector is
#' an error; use [`nv_array()`] to make it an array.
#'
#' This is implemented via the generic [`tengen::shape()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns (`integer()`)
#' @name shape
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' shape(x)
#' shape(nv_array(1:6, shape = c(2, 3)))
#' # a length-1 R vector is a scalar
#' shape(1)
#' shape(TRUE)
#' # an R array has its `dim()`
#' shape(array(1:6, dim = c(2, 3)))
#' # any other R vector has no shape
#' try(shape(c(2, 3)))
NULL

#' @rdname shape
#' @importFrom tengen shape
#' @export
tengen::shape

#' @title Get the Device of an Array
#'
#' @description Returns the device on which an array is allocated.
#'
#' @details This is implemented via the generic [`tengen::device()`].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns (device object)\cr
#' Backend-dependent device object. One of:
#'   - [`PJRTDevice`][pjrt::pjrt_device]
#'   - [`quickr_device`]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' device(x)
#' @name device
NULL

#' @rdname device
#' @importFrom tengen device
#' @export
tengen::device

#' @title Convert to an R Array
#'
#' @description
#' Transfers array data to R and returns it as an R [`array`][base::array].
#' Only in the case of scalars is the result a vector of length 1, as R `arrays` cannot have 0 axes.
#'
#' @details
#' This is implemented via the generic [`tengen::as_array()`].
#'
#' @section Data types:
#' R has fewer data types than anvl, so the values are converted to the R
#' type that can represent them:
#'
#' | Data type | R type |
#' | --- | --- |
#' | `f32`, `f64` | `double` |
#' | `i8`, `i16`, `i32`, `ui8`, `ui16` | `integer` |
#' | `i64`, `ui32`, `ui64` | [`bit64::integer64`] |
#' | `bool` | `logical` |
#'
#' This has two consequences:
#' * An `f32` value is widened to a `double` and keeps the rounding error of
#'   the 32-bit float, e.g. `as_array(nv_scalar(0.1))` is not exactly `0.1`.
#' * Some integer values cannot be represented in R: an `i32` or `i64` value
#'   equal to the smallest representable integer is read as `NA`, and a
#'   `ui64` value `>= 2^63` wraps to a negative number. The `check` argument
#'   decides whether this is reported.
#'
#' To obtain a different R type, convert the array with [`nv_convert()`]
#' first, or use the coercion functions described in [`as.double()`][as-AnvlArray].
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @param ... Passed on to methods.
#' @returns ([`array`][base::array] | `vector(1)`)\cr
#'   An R array with the input's shape, or -- for a scalar, which R cannot
#'   represent as an array -- a vector of length 1.
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

#' @title Convert an Array to a Raw Vector
#'
#' @description Returns the underlying bytes of an array as a [raw] vector.
#'
#' @details This is implemented via the generic [`tengen::as_raw()`].
#'
#' @param x ([`AnvlArray`])\cr
#'   An array.
#' @param ... Additional arguments passed to method:
#'   - `row_major` (`logical(1)`)\cr
#'     Whether to write the elements in row-major order. The default,
#'     `FALSE`, writes them in column-major order, the order in which R
#'     stores an array.
#' @returns ([`raw`])
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, shape = c(2, 2), dtype = "f32")
#' # column-major, the default
#' as_raw(x)
#' as_raw(x, row_major = TRUE)
#' @name as_raw
NULL

#' @rdname as_raw
#' @importFrom tengen as_raw
#' @export
tengen::as_raw

#' @title Get the Data Type of an Array
#'
#' @description
#' Returns the data type of an array (e.g. `f32`, `i64`).
#'
#' @details
#' An R value has no data type of its own: it only takes one when it meets a
#' typed array, or when it materializes at the default. So `dtype()` is an
#' error for a plain R value, and also for an [`RData`], the [`AbstractArray`]
#' of an R value passed to a jit-compiled function, and so for the
#' [`GraphBox`] that carries one during tracing. Use [`peek_dtype()`] for the
#' data type such a value would take.
#'
#' This is implemented via the generic [`tengen::dtype()`].
#'
#' @param x ([`AnvlArray`] | [`GraphBox`] | [`AbstractArray`])\cr
#'   An array. See Details for the values that have no data type.
#' @param ... Additional arguments passed to methods (unused).
#' @returns ([`DataType`][tengen::DataType])
#' @seealso [tengen::dtype()], [peek_dtype()], [RData]
#' @name dtype
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' dtype(x)
#' # an R value passed to a jit-compiled function has no data type
#' try(jit(dtype)(1))
#' # an array passed to it does
#' jit(function(x) {
#'   print(dtype(x))
#'   x
#' })(nv_scalar(1))
NULL

#' @rdname dtype
#' @importFrom tengen dtype
#' @export
tengen::dtype

#' @title Get the Number of Axes of an Array
#'
#' @description Returns the number of axes (sometimes also referred to as rank) of an array.
#' Equivalent to `length(shape(x))`.
#'
#' @param x ([`arrayish`])\cr
#'   An array-like object.
#' @returns (`integer(1)`)
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

#' @title Check If an Object Is a DataType
#'
#' @description Tests whether `x` is a `DataType` object.
#'
#' @param x An object to test.
#' @returns (`logical(1)`)
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
#' @returns (`DataType`)
#' @seealso [is_dtype()], [tengen::as_dtype()], [`tengen::DataType`]
#' @name as_dtype
#'
#' @examples
#' as_dtype("f32")
#' as_dtype("i32")
NULL

#' @rdname as_dtype
#' @importFrom tengen as_dtype
#' @export
tengen::as_dtype

#' @title Create a Shape Object
#'
#' @description
#' Constructs a `Shape`, the axis sizes of an array. A `Shape` *is* its integer
#' vector, with a class attached, so `length()` is the number of axes and
#' `shape[i]` is the size of axis `i`.
#'
#' @param dims An `integer()` vector of axis sizes (>= 0). `NA` marks an axis
#'   whose size is only known at run time.
#' @returns (`Shape`)
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

#' @title Get the Platform Name of an Array or Buffer
#'
#' @description
#' Returns the name of the hardware platform (e.g. `"cpu"`, `"cuda"`) the data
#' lives on. This is not the backend; see [`backend()`] for that.
#'
#' @details
#' Implemented via the generic [`pjrt::platform()`].
#'
#' @param x ([`AnvlArray`] | [`PJRTBuffer`][pjrt::pjrt_buffer])\cr
#'   An array or buffer.
#' @param ... Additional arguments passed to methods (unused).
#' @returns (`character(1)`)
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

#' @title Block Until an Async Operation Completes
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
#' @returns (`any`)\cr
#'   `x`, invisibly.
#' @seealso [pjrt::await()], [map_tree()] (to await a tree of outputs)
#' @name await
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:4, dtype = "f32")
#' await(x)
#'
#' # await all leaves of a (possibly nested) list of arrays
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

#' @title Map a Function over Trees
#' @description
#' Apply a function to each leaf of a (possibly nested) list, keeping its
#' structure, e.g. to a list of arrays a jitted function returns.
#'
#' * `map_tree()` maps over one tree.
#' * `pmap_tree()` maps over several trees of the same structure in parallel,
#'   calling `.f` with one leaf from each.
#'
#' @details
#' These are implemented in [`pjrt::map_tree()`] and [`pjrt::pmap_tree()`].
#' @param .x (any)\cr
#'   A leaf or a (nested) list of leaves.
#' @param .l (`list`)\cr
#'   A non-empty list of trees, all with the same structure.
#' @param .f (`function`)\cr
#'   Function to apply. `map_tree()` calls it with each leaf of `.x`,
#'   `pmap_tree()` with one leaf from each tree in `.l`, in order.
#' @param ... Additional arguments passed to `.f` after the leaves.
#' @return A tree with the same structure as `.x` (or `.l[[1]]`), where each
#'   leaf is the result of `.f`.
#' @examplesIf pjrt::plugins_downloaded()
#' out <- list(a = nv_array(1:2), b = list(c = nv_scalar(3)))
#' map_tree(out, dtype)
#' map_tree(out, as_array)
#' pmap_tree(list(list(a = 1, b = 2), list(a = 10, b = 20)), `+`)
#' @name map_tree
NULL

#' @rdname map_tree
#' @importFrom pjrt map_tree
#' @export
pjrt::map_tree

#' @rdname map_tree
#' @importFrom pjrt pmap_tree
#' @export
pjrt::pmap_tree
