describe("dapply", {
  device_names <- function(xs) vapply(xs, \(x) as.character(device(x)), character(1L))
  devices_named <- function(ids) vapply(ids, \(id) as.character(nv_device(id)), character(1L), USE.NAMES = FALSE)

  it("returns what lapply() returns, with the names of X", {
    f <- jit(function(x, y) x * y)
    xs <- list(a = nv_array(1:3), b = nv_array(4:6), c = nv_array(7:9))
    out <- dapply(xs, f, y = 2L)
    expect_named(out, c("a", "b", "c"))
    expect_equal(lapply(out, as_array), lapply(lapply(xs, f, y = 2L), as_array))
  })

  it("assigns the elements to the devices in turn", {
    f <- jit(function(x) x + 1L)
    out <- dapply(as.list(1:5), f, devices = c("cpu:0", "cpu:1"))
    expect_equal(device_names(out), devices_named(rep(c("cpu:0", "cpu:1"), length.out = 5L)))
    expect_equal(vapply(out, as.integer, integer(1L)), 2:6)
  })

  it("runs on every device of the default device's platform by default", {
    skip_if_not(is_cpu())
    out <- dapply(1:2, jit(function(x) x * 2L))
    expect_equal(device_names(out), devices_named(c("cpu:0", "cpu:1")))
  })

  it("copies the arrays of X and ... to the call's device", {
    f <- jit(function(x, y) list(sum = x$a + y))
    x <- list(a = nv_array(1, device = "cpu:0"))
    y <- nv_array(10, device = "cpu:0")
    out <- dapply(list(x, x), f, y, devices = c("cpu:0", "cpu:1"))
    expect_equal(device_names(lapply(out, `[[`, "sum")), devices_named(c("cpu:0", "cpu:1")))
    expect_equal(as.numeric(as_array(out[[2L]]$sum)), 11)
  })

  it("creates the arrays of a call that has none to read a device from on its device", {
    f <- jit(function(seed) nv_runif(2L, nv_rng_state(seed))$values)
    out <- dapply(c(1L, 1L), f, devices = c("cpu:0", "cpu:1"))
    expect_equal(device_names(out), devices_named(c("cpu:0", "cpu:1")))
    expect_equal(as_array(out[[1L]]), as_array(out[[2L]]))
  })

  it("honours the static arguments of a jitted FUN", {
    f <- jit(function(x, n) x + nv_fill(0, shape = n), static = "n")
    out <- dapply(list(1, 2), f, n = 3L, devices = c("cpu:0", "cpu:1"))
    expect_equal(lapply(out, \(x) as.numeric(as_array(x))), list(rep(1, 3L), rep(2, 3L)))
  })

  it("compiles one program per set of devices and keeps it on the jitted function", {
    f <- jit(function(x) x + 1L)
    dapply(list(1L, 2L), f, devices = c("cpu:0", "cpu:1"))
    dapply(list(3L, 4L), f, devices = c("cpu:0", "cpu:1"))
    expect_length(ls(environment(f)$.jit_dapply), 1L)
    dapply(list(1L, 2L, 3L), f, devices = c("cpu:0", "cpu:1"))
    expect_length(ls(environment(f)$.jit_dapply), 2L)
  })

  it("runs a plain function", {
    out <- dapply(list(nv_array(1:2), nv_array(3:4)), function(x) x * 2L, devices = c("cpu:0", "cpu:1"))
    expect_equal(lapply(out, as.integer), list(c(2L, 4L), c(6L, 8L)))
  })

  it("errors when the elements would run different programs", {
    f <- jit(function(x) x)
    expect_error(dapply(list(nv_array(1:2), nv_array(1:3)), f), "Element 2 differs")
  })

  it("errors for a FUN compiled for a device of its own", {
    f <- jit(function(x) x, device = "cpu:0")
    expect_error(dapply(list(1), f), "device of its own")
  })

  it("returns an empty list for an empty X", {
    expect_equal(dapply(list(), jit(function(x) x)), list())
  })

  it("runs on the quickr backend's single device", {
    skip_if_no_quickr()
    local_backend("quickr")
    out <- dapply(list(nv_array(1:2), nv_array(3:4)), jit(function(x) x * 2L))
    expect_equal(lapply(out, as.integer), list(c(2L, 4L), c(6L, 8L)))
  })

  it("errors when called while tracing", {
    f <- jit(function(x) dapply(list(x), identity))
    expect_error(f(nv_array(1)), "cannot be called inside")
  })

  it("errors when X is an array", {
    expect_error(dapply(nv_array(1:2), identity), "must be a list or a vector")
  })
})
