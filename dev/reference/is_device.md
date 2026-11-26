# Test Whether an Object Is a Device

Test Whether an Object Is a Device

## Usage

``` r
is_device(x)
```

## Arguments

- x:

  An object to test.

## Value

(`logical(1)`)  
Whether `x` is a device of one of the backends: a `PJRTDevice`, or an
object whose class inherits from `"AnvlDevice"` (see
[`AnvlBackend()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackend.md)).
