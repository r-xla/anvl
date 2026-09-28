# Create a Primitive

`new_primitive()` creates a new primitive: an `AnvlPrimitive`, the
function the primitive is called through (e.g.
[`prim_add()`](https://r-xla.github.io/anvl/dev/reference/prim_add.md)).
For details on how to do this, see the [Adding a
Primitive](https://r-xla.github.io/anvl/articles/extending_primitive.html)
article. Like every jitted function it runs on the active backend when
called.

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
  Primitive name, without the `prim_` prefix (`"add"` for
  [`prim_add()`](https://r-xla.github.io/anvl/dev/reference/prim_add.md)).

- fn:

  (`function`)  
  Body of the primitive. Its formals become the formals of the returned
  JIT-compiled callable. Inside `fn`, the primitive is accessible via
  the lexically-bound symbol `self` (an
  [`AnvlPrimitiveDef`](https://r-xla.github.io/anvl/dev/reference/AnvlPrimitiveDef.md));
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
  Whether to add the primitive to anvl's internal registry of
  primitives, under `name`, replacing one registered under the same
  name. The quickr backend reads that registry to know which primitives
  it can lower, so a primitive created with `register = FALSE` is
  rejected on quickr even if it has a `quickr` rule. The other backends
  only read the rules of the primitive itself. This does not bind the
  result to a `prim_<name>` variable; assign it yourself.

## Value

(`AnvlPrimitive`)  
The function the primitive is called through, of class
`c("AnvlPrimitive", "JitFunction")`. Its
[`AnvlPrimitiveDef`](https://r-xla.github.io/anvl/dev/reference/AnvlPrimitiveDef.md)
is `attr(<fn>, "definition")`, and `[[` / `[[<-` on it access the rules
of that definition.
