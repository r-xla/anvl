describe("AnvlBackendPjrt", {
  it("stores an array with its shape reversed, keeping R's bytes", {
    x <- nv_array(array(1:24, c(2L, 3L, 4L)), dtype = "i32")
    expect_equal(shape(x), c(2L, 3L, 4L))
    expect_equal(xlamisc::shape(x$data), c(4L, 3L, 2L))
    expect_equal(as_array(x), array(1:24, c(2L, 3L, 4L)))
  })

  it("reads the raw bytes in either order", {
    x <- nv_array(matrix(1:6, nrow = 2L), dtype = "i32")
    expect_equal(readBin(as_raw(x), "integer", n = 6L, size = 4L), 1:6)
    expect_equal(readBin(as_raw(x, row_major = TRUE), "integer", n = 6L, size = 4L), c(1L, 3L, 5L, 2L, 4L, 6L))
  })
})

describe("nv_array(<PJRTBuffer>)", {
  it("takes the array the buffer holds, as pjrt reads it", {
    arr <- array(as.double(1:24), c(2L, 3L, 4L))
    x <- nv_array(pjrt::pjrt_buffer(arr, dtype = "f32"))
    expect_equal(shape(x), c(2L, 3L, 4L))
    expect_equal(as_array(x), arr)
    expect_equal(as_array(nv_array(pjrt::pjrt_buffer(1:3))), array(1:3))
  })

  it("errors on a shape, data type or device the buffer does not have", {
    buf <- pjrt::pjrt_buffer(matrix(1:6, nrow = 2L))
    expect_error(nv_array(buf, shape = c(3L, 2L)))
    expect_error(nv_array(buf, dtype = "f32"))
    skip_if_not(is_cpu())
    expect_error(nv_array(buf, device = "cpu:1"), "same device")
  })
})
