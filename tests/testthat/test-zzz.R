describe("read_env_defaults()", {
  it("reads ANVL_DEFAULT_DTYPES and ANVL_DEFAULT_DEVICE", {
    withr::defer(read_env_defaults())
    withr::local_envvar(ANVL_DEFAULT_DTYPES = "float=f64", ANVL_DEFAULT_DEVICE = "cpu:1")
    read_env_defaults()
    expect_identical(globals[["ENV_DEFAULT_DTYPES"]], c(float = "f64"))
    expect_identical(globals[["ENV_DEFAULT_DEVICE"]], "cpu:1")
  })

  it("reads nothing when they are unset", {
    withr::defer(read_env_defaults())
    withr::local_envvar(ANVL_DEFAULT_DTYPES = NA, ANVL_DEFAULT_DEVICE = NA)
    read_env_defaults()
    expect_null(globals[["ENV_DEFAULT_DTYPES"]])
    expect_null(globals[["ENV_DEFAULT_DEVICE"]])
  })
})
