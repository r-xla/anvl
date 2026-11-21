# Skip a Test When No PJRT Plugin Is Available

Skips the calling
[testthat](https://CRAN.R-project.org/package=testthat) test unless the
PJRT plugin for `platform` can be used without a prompt or an error: it
is downloaded, the `PJRT_PLUGIN_PATH_<PLATFORM>` environment variable
(e.g. `PJRT_PLUGIN_PATH_CPU`) points to it, or `PJRT_INSTALL=1` allows
downloading it on first use.

The PJRT plugins are downloaded rather than shipped with
[pjrt](https://CRAN.R-project.org/package=pjrt), so no plugin is
available on CRAN. A package that depends on anvl calls this at the top
of each test that runs anvl code on the `"pjrt"` backend; see the
[Depending on
anvl](https://r-xla.github.io/anvl/articles/depending-on-anvl.html)
article.

## Usage

``` r
skip_if_no_plugin(platform = NULL)
```

## Arguments

- platform:

  (`NULL` \| `character(1)`)  
  Platform whose plugin the test needs, e.g. `"cpu"` or `"cuda"`. `NULL`
  takes the platform of the [default
  device](https://r-xla.github.io/anvl/dev/reference/default_device.md),
  which is where a test that names no device runs. It is read off the
  device's identifier, so no plugin is loaded to find it.

## Value

(`NULL`)  
Invisibly, when the plugin is available. Otherwise, the test is skipped.

## See also

[`install_anvl()`](https://r-xla.github.io/anvl/dev/reference/install_anvl.md)

## Examples

``` r
testthat::test_that("arrays add up", {
  skip_if_no_plugin()
  testthat::expect_equal(as_array(nv_scalar(1) + 2), 3)
})
#> Test passed with 1 success 🌈.
```
