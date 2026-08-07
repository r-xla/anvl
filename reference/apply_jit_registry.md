# Apply a `@jit` registry

Iterates over a registry produced by
[`jit_roclet()`](https://r-xla.github.io/anvl/reference/jit_roclet.md)
and rebinds each listed function in `envir` to
`jit(f, backend = "auto", static = entry$static)`.

Call this from the top level of your package's `R/zzz.R`, right next to
`.onLoad`, so the wrappers are byte-compiled during package install
instead of being rebuilt on every `.onLoad`:

    anvl::apply_jit_registry(.jit_registry)

`.jit_registry` is the variable defined by `R/jit-registry.R`, which is
regenerated on every `devtools::document()`.

## Usage

``` r
apply_jit_registry(registry, envir = parent.frame())
```

## Arguments

- registry:

  (`list`)  
  List of `list(name = <chr>, static = <chr|int>)` entries. Typically
  the `.jit_registry` object emitted by the roclet.

- envir:

  (`environment`)  
  Environment in which to look up and rebind functions. Defaults to
  [`parent.frame()`](https://rdrr.io/r/base/sys.parent.html), which at
  top-level package source time is the package namespace.

## Value

Invisibly returns `envir`.

## See also

[`jit_roclet()`](https://r-xla.github.io/anvl/reference/jit_roclet.md),
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md)
