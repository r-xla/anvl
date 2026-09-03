# While Loop

Executes a functional while loop.

## Usage

``` r
nv_while(init, cond, body)
```

## Arguments

- init:

  (`named list()`)  
  Named list of initial state values. Each one becomes a parameter of
  the loop's sub-graphs. R values are committed at their default data
  type.

- cond:

  (`function`)  
  Condition function returning a scalar boolean. Receives the state
  values as arguments.

- body:

  (`function`)  
  Body function returning the updated state as a named list with the
  same structure as `init`.

## Value

Final state after the loop terminates (same structure as `init`).

## See also

[`prim_while()`](https://r-xla.github.io/anvl/dev/reference/prim_while.md)
for the underlying primitive.

## Examples

``` r
nv_while(
  init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
  cond = function(i, total) i < 5L,
  body = function(i, total) list(
    i = i + 1L,
    total = total + i
  )
)
#> $i
#> AnvlArray
#>  5
#> [ CPUi32{} ] 
#> 
#> $total
#> AnvlArray
#>  10
#> [ CPUi32{} ] 
#> 
```
