# Get the default backend

Returns the current default backend from
`getOption("anvl.default_backend", "pjrt")`.

## Usage

``` r
default_backend()
```

## Value

`character(1)` — the backend name (e.g. `"pjrt"`, `"quickr"`).

## See also

[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
