# Assert Shape Vector

Check whether an input is a valid shape vector: whole, non-negative axis
sizes.

## Usage

``` r
assert_shapevec(
  x,
  min_len = 0L,
  var_name = rlang::caller_arg(x),
  call = rlang::current_env()
)
```

## Arguments

- x:

  Object to check.

- min_len:

  (`integer(1)`)  
  Minimum number of axes. Default is 0.

- var_name:

  (`character(1)`)  
  Name of the variable to use in error messages.

- call:

  (`environment`)  
  The frame an error is reported for.

## Value

([`integer()`](https://rdrr.io/r/base/integer.html))  
`x` as an integer vector.
