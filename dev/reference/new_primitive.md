# Create a Primitive

Create a new primitive. For details on how to do this, see the article
on *Adding a Primitive*. Like every jitted function it runs on the
active backend when called.

## Usage

``` r
new_primitive(
  name,
  fn,
  subgraphs = character(),
  static = character(),
  register = TRUE
)
```

## Arguments

- name:

  (`character(1)`)  
  Primitive name.

- fn:

  (`function`)  
  Body of the primitive. Its formals become the formals of the returned
  JIT-compiled callable. Inside `fn`, the primitive is accessible via
  the lexically-bound symbol `self` (an
  [`AnvlPrimitive`](https://r-xla.github.io/anvl/dev/reference/AnvlPrimitive.md));
  pass it as the first argument to
  [`graph_desc_add()`](https://r-xla.github.io/anvl/dev/reference/graph_desc_add.md).

- subgraphs:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters that are subgraphs (for higher-order primitives).

- static:

  ([`character()`](https://rdrr.io/r/base/character.html) \|
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  Passed to
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

- register:

  (`logical(1)`)  
  If `TRUE` (default), register the result under `name` in the primitive
  registry.

## Value

A callable of class `c("JitPrimitive", "JitFunction")`.
