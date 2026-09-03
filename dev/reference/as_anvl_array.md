# Convert to AnvlArray

Use this to canonicalize inputs at the start of a function so it works
both with eager executing and in combination with
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md). Use
`as_anvl_array()` for a single input and `as_anvl_arrays()` for multiple
inputs. The latter will also ensure all arrays are from the same backend
and live on the same device, and can additionally apply type promotion
rules via the `.promote` argument.

## Usage

``` r
as_anvl_array(x, device = NULL)

as_anvl_arrays(..., .promote = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input to standardize.

- device:

  (`NULL` \|
  [`device`](https://r-xla.github.io/anvl/dev/reference/device.md))  
  Target device. If `x` is an `AnvlArray` on a different device, an
  error is raised.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Inputs to align. Name them to be able to point `.promote` at one of
  them.

- .promote:

  (`NULL` \| `function`)  
  Which dtype every input is brought to. See
  [`promotion_rule`](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md)
  for more information.

## Value

(One or more
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
values).

## See also

[`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md),
[`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md)

## Examples

``` r
as_anvl_array(1L)
#> AnvlArray
#>  1
#> [ CPUi32{} ] 
as_anvl_arrays(nv_array(1:3), 1L)
#> [[1]]
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1
#> [ CPUi32{} ] 
#> 
as_anvl_arrays(nv_array(1L), nv_array(1.5), .promote = promote_common())
#> [[1]]
#> AnvlArray
#>  1
#> [ CPUf32{1} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1.5000
#> [ CPUf32{1} ] 
#> 
```
