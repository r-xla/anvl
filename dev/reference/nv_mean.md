# Mean

Computes the arithmetic mean along the specified axes. You can also use
[`mean()`](https://rdrr.io/r/base/mean.html).

## Usage

``` r
nv_mean(x, axes = NULL, drop = TRUE, nan_rm = FALSE)

# S3 method for class 'AnvlArray'
mean(x, trim = 0, na.rm = FALSE, ..., axes = NULL, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce over. Negative values count from the end, i.e. `-1`
  refers to the last axis. If `NULL` (default), reduces over all axes.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced axes: removed from the output shape if
  `TRUE`, set to 1 if `FALSE`.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in float inputs. If `FALSE` (default),
  `NaN` propagates. If `TRUE`, `NaN` values are skipped.

- trim:

  Currently not supported.

- na.rm:

  Forwarded to `nv_mean()`'s `nan_rm` argument.

- ...:

  No additional arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type where that is a float, and the default float
data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
otherwise. The shape is the input's with the reduced axes removed
(`drop = TRUE`) or set to 1 (`drop = FALSE`).

## See also

[`nv_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_sum.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
# an integer input is averaged at the default float data type
nv_mean(x)
#> AnvlArray
#>  3.5000
#> [ CPUf32{} ] 

# a float input keeps its own, whatever the default float is
nv_mean(nv_array(c(1, 2), dtype = "f64"))
#> AnvlArray
#>  1.5000
#> [ CPUf64{} ] 

# reducing axis 1 removes it, drop = FALSE keeps it at size 1
nv_mean(x, axes = 1L)
#> AnvlArray
#>  1.5000
#>  3.5000
#>  5.5000
#> [ CPUf32{3} ] 
nv_mean(x, axes = 1L, drop = FALSE)
#> AnvlArray
#>  1.5000 3.5000 5.5000
#> [ CPUf32{1,3} ] 

# NaN propagates unless nan_rm = TRUE
nv_mean(nv_array(c(1, NaN, 3)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_mean(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
```
