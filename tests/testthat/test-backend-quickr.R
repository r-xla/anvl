describe("the quickr backend", {
  it("writes the bytes of an array with as_raw()", {
    skip_if_no_quickr()
    local_backend("quickr")
    x <- nv_matrix(c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5), nrow = 2L)
    expect_identical(as_raw(x), writeBin(c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5), raw(), size = 8L, endian = "little"))
    expect_identical(
      as_raw(x, row_major = TRUE),
      writeBin(c(1.5, 3.5, 5.5, 2.5, 4.5, 6.5), raw(), size = 8L, endian = "little")
    )
  })

  it("lays out the bytes of an array as the pjrt backend does", {
    skip_if_no_quickr()
    data <- list(
      f64 = c(1.5, -2, 3e10, 0, 5, 6),
      i32 = c(1L, -2L, 3L, 0L, 5L, 6L),
      bool = c(TRUE, FALSE, TRUE, TRUE, FALSE, FALSE)
    )
    for (dt in names(data)) {
      m <- matrix(data[[dt]], nrow = 2L)
      x_pjrt <- with_backend("pjrt", nv_array(m, dtype = dt))
      x_quickr <- with_backend("quickr", nv_array(m, dtype = dt))
      expect_identical(as_raw(x_quickr), as_raw(x_pjrt))
      expect_identical(as_raw(x_quickr, row_major = TRUE), as_raw(x_pjrt, row_major = TRUE))
    }
  })

  it("builds an array from a raw payload", {
    skip_if_no_quickr()
    local_backend("quickr")
    payload <- writeBin(1:6, raw(), size = 4L, endian = "little")
    x <- nv_array(payload, dtype = "i32", shape = c(2L, 3L))
    expect_equal(backend(x), "quickr")
    expect_identical(as_array(x), matrix(1:6, nrow = 2L))
    x_row <- nv_array(payload, dtype = "i32", shape = c(2L, 3L), byrow = TRUE)
    expect_identical(as_array(x_row), matrix(1:6, nrow = 2L, byrow = TRUE))
    b <- nv_array(as.raw(c(1L, 0L, 1L)), dtype = "bool", shape = 3L)
    expect_identical(as_array(b), array(c(TRUE, FALSE, TRUE)))
  })

  it("refuses a raw payload of the wrong size", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_error(nv_array(as.raw(1:4), dtype = "f64", shape = 2L), "does not hold 2 elements")
  })

  it("refuses a raw payload at a data type it does not support", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_error(nv_array(as.raw(1:4), dtype = "ui8", shape = 4L), "quickr")
  })

  it("refuses the bytes of a bool array with missing values", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_error(as_raw(nv_array(c(TRUE, NA))), "missing values")
  })

  it("takes a device given as a string", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_equal(device(nv_array(1:3, device = "cpu")), quickr_device("cpu"))
    expect_equal(device(nv_empty(shape = 2L, dtype = "f64", device = "cpu")), quickr_device("cpu"))
    expect_error(nv_array(1:3, device = "cuda"), "cpu")
  })
})
