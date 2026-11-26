# Register a Backend

**\[experimental\]**

Makes a backend selectable under `name`:
[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
and
[`with_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
accept the name, and so does the option `anvl.backend`. A package that
provides a backend calls this from its `.onLoad()` hook, together with
[`register_rule_type()`](https://r-xla.github.io/anvl/dev/reference/register_rule_type.md)
for the rules it attaches to the primitives. Registering a backend under
a name that is already taken replaces the backend registered there.

## Usage

``` r
register_backend(name, backend)
```

## Arguments

- name:

  (`character(1)`)  
  The name of the backend: the `backend` field of its arrays, and what
  [`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md)
  returns for its arrays and devices. It must not be `"float"` or
  `"int"`, which the option `anvl.default_dtypes` reserves for the data
  type categories.

- backend:

  ([`AnvlBackend`](https://r-xla.github.io/anvl/dev/reference/AnvlBackend.md))  
  The backend. It is evaluated on first use rather than in this call.

## Value

(`NULL`)  
Invisibly. Called for its side effect.

## See also

[`AnvlBackend()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackend.md),
[`register_rule_type()`](https://r-xla.github.io/anvl/dev/reference/register_rule_type.md),
[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
