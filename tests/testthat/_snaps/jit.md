# rejecting a reference-semantics static names it helpfully

    Code
      f(list(a = 1, opts = list(env = e)), nv_array(1))
    Condition
      Error:
      ! Static argument `s$opts$env` has reference semantics: it is an environment.
      x A static value is part of the compilation cache key and is compared by identity, so mutating it in place would silently reuse a program compiled from its old contents.
      i Pass it as a regular argument, or extract the plain values you need from it.

