# Get Active Backend

Retrieves the active backend (option `anvl.backend`), falling back to
the default `"pjrt"` backend.

## Usage

``` r
active_backend()
```

## Value

`character(1)` — the backend name (e.g. `"pjrt"`, `"quickr"`).

## See also

[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md),
[`with_backend()`](https://r-xla.github.io/anvl/dev/reference/with_backend.md)
