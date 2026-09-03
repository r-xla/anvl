# JIT Deep Dive

This vignette explains what actually happens when you wrap a function
with [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).
Understanding this is what lets you avoid common pitfals on
tracing-based compilers. If you haven’t yet, read the [Get
Started](https://r-xla.github.io/anvl/dev/articles/anvl.md) vignette
first.

We will use the simple `linear` function as the running example.

``` r

library(anvl)
set.seed(42)

linear <- function(x, w, b) nv_add(nv_mul(x, w), b)
```

## How `jit()` works

In pseudo-R, `jit(f)` returns roughly this closure:

``` r

jit <- function(f, static = character()) {
  cache <- hashtab()
  function(...) {
    # abstract representations of dynamic args, values of static args
    key <- input_signature(..., static = static)
    if (is.null(cache[[key]])) {
      # step 1: record prim_* calls into an AnvlGraph
      graph <- trace_fn(f, list(...), static = static)
      # step 2: lower to an XLA executable, store
      cache[[key]] <- compile(graph)
    }
    # step 3: run the cached executable on this call's inputs
    cache[[key]](...)
  }
}
```

Three things happen:

1.  **Trace** the R code once with placeholder values (except for
    `static` arguments covered below), recording the sequence of
    primitive operations into an intermediate representation called an
    `AnvlGraph`.
2.  **Compile** that graph to an XLA executable and cache it under a key
    derived from the inputs, which uses abstract types for `AnvlArray`s
    and the actual values for static inputs.
3.  On subsequent calls with a cache hit, **skip tracing and
    compilation** and run the cached executable directly.

### The compilation cache

When a call’s inputs match a cached entry, both tracing and compilation
are skipped. We can observe this directly by inspecting the size of the
cache held inside the jitted function:

``` r

cache_size <- function(f) environment(f)$cache$size

linear_jit <- jit(linear)
cache_size(linear_jit)   # 0: nothing cached yet
#> NULL

linear_jit(nv_scalar(2), nv_scalar(3), nv_scalar(1))
#> AnvlArray
#>  7
#> [ CPUf32{} ]
cache_size(linear_jit)   # 1: a new entry was added
#> NULL

# same shapes -> cache hit, size unchanged
linear_jit(nv_scalar(2), nv_scalar(3), nv_scalar(5))
#> AnvlArray
#>  11
#> [ CPUf32{} ]
cache_size(linear_jit)
#> NULL

# different shapes -> a second entry is added
linear_jit(
  nv_array(c(1, 2)),
  nv_array(c(3, 4)),
  nv_array(c(1, 1))
)
#> AnvlArray
#>  4
#>  9
#> [ CPUf32{2} ]
cache_size(linear_jit)
#> NULL
```

Each input to the function contributes to the cache key differently,
depending on whether it is *dynamic* (an arrayish value, the default) or
*static* (marked via the `static =` argument of
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)):

- **Dynamic inputs** contribute their *abstract value* – the
  `nv_aval(dtype, shape)` pair, or, for a bare R value, its R type and
  shape. Two arrays with the same abstract value but different data hit
  the same cache entry.
- **Static inputs** contribute their *exact R value*, compared with
  [`identical()`](https://rdrr.io/r/base/identical.html). They stay as
  regular R values during the compilation, but their value is fixed for
  a compiled program. Two calls with `flag = TRUE` and `flag = FALSE`
  therefore land on different cache entries.

In the snippet above all three inputs are dynamic, so the first two
calls share a key (three `f32[]` scalars) and hit the cache, while the
third call presents three `f32[2]` vectors and forces a retrace.

You’ll want `static =` when the body of the function needs to look at a
concrete R value – typically a flag, a small integer, or a string. It’s
also the only way to do R-level input validation on a value: a dynamic
input is just a shape/dtype placeholder during tracing, so a check like
`stopifnot(abs(sum(p) - 1) < 1e-6)` – verifying that `p` is a proper
probability vector – only works if `p` is static.

``` r

linear_maybe <- function(x, w, b, use_bias) {
  if (use_bias) linear(x, w, b) else x * w
}
linear_maybe_jit <- jit(linear_maybe, static = "use_bias")

linear_maybe_jit(2, 3, 1, use_bias = TRUE)
#> AnvlArray
#>  7
#> [ CPUf32{} ]
linear_maybe_jit(2, 3, use_bias = FALSE)
#> AnvlArray
#>  6
#> [ CPUf32{} ]
```

Each call with a new static value forces a re-trace and re-compile, so
static arguments cause more re-compiles than dynamic ones. Only use them
when you really need them.

The cache key also includes information about the input structure (if
`AnvlArray`s are nested in lists) or the device to compile for, but we
will ignore them as they are less relevant to understand for using
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

Two important notes to be aware of:

- Call [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) once
  on a function and reuse the result; calling
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) inside a
  loop creates a fresh cache on every iteration and defeats the point:
- For functions that will be called many times with consistent shapes,
  compilation is a one-time cost. For computations on shapes that won’t
  recur, the compile time may dominate – see [Padding Inputs to Avoid
  Recompilation](https://r-xla.github.io/anvl/dev/articles/efficiency.html#padding-inputs-to-avoid-recompilation)
  in the efficiency vignette for one way to keep the cache small.

## Tracing

*Tracing* is how the R code is translated into a form that the XLA
compiler can understand. It works by replacing the dynamic inputs with
special (`GraphBox`) values and runnig it. Instead of doing the array
computations, this will instead record every primitive operation in an
`AnvlGraph`, which represents the *evaluation trace*. For the purposes
of this vignette you can think of tracing and the subsequent XLA
compilation as a single phase that runs once per cache miss. Note that
this is different from R’s
[`trace()`](https://rdrr.io/r/base/trace.html) function, which lets you
insert code into functions. In {anvl}, this tracing machinery is
available via
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md).
Although you will probably never need to use this function directly, we
will use it to show what’s happening under the hood. Below, we trace
`linear` with a length-3 vector for `x` and scalars for `w` and `b`:

``` r

f32_scalar <- nv_aval("f32", integer())
f32_scalar
#> AbstractArray(dtype=f32, shape=)
f32_vec3   <- nv_aval("f32", 3)
f32_vec3
#> AbstractArray(dtype=f32, shape=3)
trace_fn(linear, args = list(x = f32_vec3, w = f32_scalar, b = f32_scalar))
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f32[3]
#>     %x2: f32[]
#>     %x3: f32[]
#>   Body:
#>     %1: f32[3] = broadcast_in_axes [shape = 3, broadcast_axes = <any>] (%x2)
#>     %2: f32[3] = mul(%x1, %1)
#>     %3: f32[3] = broadcast_in_axes [shape = 3, broadcast_axes = <any>] (%x3)
#>     %4: f32[3] = add(%2, %3)
#>   Outputs:
#>     %4: f32[3]
```

The printed `AnvlGraph` is like an R function: it has inputs, a body and
outputs. However, the content is more structured:

- There are only calls into primitives and not closures; i.e., function
  calls are inlined.
- Types are fully specified.
- Every variable is assigned to once, i.e. the program is in SSA (Single
  Static Assignment) form.

Crucially, only `prim_*` calls (and the `nv_*` API functions or
overloaded operators that delegate to them) get recorded. Any other R
code in the body might influence what the evaluation trace is, but is
not present in the traced graph itself.

### The Tracing Contract

Tracing only produces correct results for *pure* functions – functions
whose output depends on their arguments and nothing else, and that have
no R-level side effects. This contract is a consequence of both how the
caching works and that the compiled program runs outside of R and only
communicates back to the R interpreter via it’s return values.
Concretely, the function’s execution path – the specific sequence of
primitive calls it performs – must depend only on:

1.  The *abstract* representation (shape and dtype) of each dynamic
    input, and
2.  The *value* of each static input.

The next subsections each show what tracing does to particular R code
and where its behavior might be surprising.

### R loops are unrolled

Tracing runs your R code and records primitive operations in the graph.
Because `for` is not an anvl primitive, it will be executed as usual and
all the primitive calls encountered will be recorded in the graph. Here
we apply the `linear` function `n` times.

``` r

linear_repeated <- function(x, w, b, n) {
  for (i in seq_len(n)) x <- linear(x, w, b)
  x
}
trace_fn(linear_repeated, args = list(x = f32_scalar, w = f32_vec3, b = f32_vec3, n = 2L))
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f32[]
#>     %x2: f32[3]
#>     %x3: f32[3]
#>   Body:
#>     %1: f32[3] = broadcast_in_axes [shape = 3, broadcast_axes = <any>] (%x1)
#>     %2: f32[3] = mul(%1, %x2)
#>     %3: f32[3] = add(%2, %x3)
#>     %4: f32[3] = mul(%3, %x2)
#>     %5: f32[3] = add(%4, %x3)
#>   Outputs:
#>     %5: f32[3]
```

The graph contains a single `broadcast_in_axes` (lifting the scalar `x`
to `f32[3]`) followed by two `mul`/`add` pairs in sequence – not a loop
construct. The broadcast happens only once because by the second
iteration `x` is already an `f32[3]` and matches `w` and `b` directly.
The compiled executable will contain those five operations laid out one
after another. For a different value of `n`, the loop would get unrolled
for this specific iteration number. Long loops also lead to long compile
times and large executables. You can instead use
[`nv_while()`](https://r-xla.github.io/anvl/dev/reference/nv_while.md),
which records a single higher-order primitive call in the graph
regardless of how many iterations the loop runs.

### R `if` statements pick one branch

Tracing runs only the `if`-branch that the condition selects at trace
time. One common scenario is where the branch depends on a static input
flag:

``` r

linear_maybe <- function(x, w, b, use_bias) {
  if (use_bias) linear(x, w, b) else x * w
}
trace_fn(
  linear_maybe,
  args = list(x = f32_scalar, w = f32_scalar, b = f32_scalar, use_bias = TRUE)
)
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f32[]
#>     %x2: f32[]
#>     %x3: f32[]
#>   Body:
#>     %1: f32[] = mul(%x1, %x2)
#>     %2: f32[] = add(%1, %x3)
#>   Outputs:
#>     %2: f32[]
```

The graph contains one `mul` and one `add` operation, but no
conditional. Tracing with `use_bias = FALSE` would produce a graph
containing the other branch.

Where things go wrong is if the evaluation trace depends on something
that does not influence the cache key, such as a value from the
enclosing environment:

``` r

threshold <- 0.5
h <- function(x) {
  if (threshold > 0.5) x * 2 else x + 1
}
trace_fn(h, args = list(x = f32_scalar))
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f32[]
#>   Body:
#>     %1: f32[] = add(%x1, 1:f32)
#>   Outputs:
#>     %1: f32[]
```

The trace itself runs correctly, but it becomes a problem in combination
with [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)’s
caching mechanism. The closed-over `threshold` does not influence the
cache key, so subsequent calls with the same dynamic input type would
reuse the executable produced when `threshold` was `0.5`, no matter how
`threshold` is at call time. The only fix is to not let the graph depend
on values outside the function’s signature – in this case, make
`threshold` an explicit static argument.

### Closed-over values become constants

We just saw an extreme version of this in the `if` section: a
closed-over R variable picked the *branch* that ended up in the graph.
The same dynamic applies to closed-over values used as plain operands –
their value at trace time is read once and baked into the graph.

Here we close over a default bias instead of taking it as an argument:

``` r

default_b <- 5
linear_default_b <- function(x, w) linear(x, w, default_b)
trace_fn(linear_default_b, args = list(x = f32_scalar, w = f32_scalar))
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f32[]
#>     %x2: f32[]
#>   Body:
#>     %1: f32[] = mul(%x1, %x2)
#>     %2: f32[] = add(%1, 5:f32)
#>   Outputs:
#>     %2: f32[]
```

The graph contains `add(%1, 5:f32?)` – the value `5` is hard-wired into
the program. Changing `default_b` afterwards has no effect on the graph
(and, once the graph is compiled, no effect on the executable either).

### Side effects only fire during tracing

The *Tracing Contract* above already noted that a jitted function must
be pure. This subsection makes the consequence concrete: **R-level side
effects only have an effect while the graph is being built**, not on
subsequent calls.

A common R pattern for stateful objects is to wrap them in an
environment, since environments give you reference semantics:

``` r

new_model <- function(beta) {
  e <- new.env()
  e$beta <- beta
  e$grad_step <- function(beta_grad, lr) {
    e$beta <- e$beta - beta_grad * lr
    e$beta
  }
  e
}
```

If this was executed in “standard R”, every call to `model$grad_step()`
would nudge `model$beta` one step further along the gradient. But
wrapping `grad_step` with
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) breaks the
function on two levels:

``` r

model <- new_model(nv_array(c(0, 0, 0), dtype = "f32"))
grad_step_jit <- jit(model$grad_step)
g <- nv_array(c(1, 1, 1), dtype = "f32")

grad_step_jit(g, 0.1)   # expected c(-0.1, -0.1, -0.1)
#> AnvlArray
#>  -0.1000
#>  -0.1000
#>  -0.1000
#> [ CPUf32{3} ]
grad_step_jit(g, 0.1)   # expected c(-0.2, -0.2, -0.2) -- but identical to call 1
#> AnvlArray
#>  -0.1000
#>  -0.1000
#>  -0.1000
#> [ CPUf32{3} ]
grad_step_jit(g, 0.1)   # expected c(-0.3, -0.3, -0.3) -- but identical to call 1
#> AnvlArray
#>  -0.1000
#>  -0.1000
#>  -0.1000
#> [ CPUf32{3} ]

class(model$beta)       # not even an AnvlArray any more
#> [1] "GraphBox" "AnvlBox"
```

Two things went wrong, both for the same reason: the only thing
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) records
into the graph is `prim_*` calls.

- The mutation `e$beta <- ...` is a plain R assignment, not a `prim_*`
  call, so it doesn’t get recorded into the `AnvlGraph`. It runs once,
  during tracing – but the value being assigned is anvl’s internal trace
  placeholder (a `GraphBox`), not a real array. So after the first call,
  `model$beta` is *broken*: it holds a leaked tracer instead of the
  array you expected.
- On every subsequent call only the cached XLA executable runs – never
  the R body, so the `e$beta <- ...` line will not be executed again.
  And because `e$beta` was a closed-over R value at trace time, its
  initial value (`c(0, 0, 0)`) is baked into the graph as a literal (per
  *Closed-over values become constants* above), so every call returns
  the same `-0.1`.

This is why
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)-compiled
functions must be *pure*: their output depends only on their arguments,
and any state updates have to happen at the call site, not inside the
function:

``` r

grad_step <- jit(function(beta, beta_grad, lr) beta - beta_grad * lr)

beta <- nv_array(c(0, 0, 0), dtype = "f32")
beta <- grad_step(beta, g, 0.1)
beta <- grad_step(beta, g, 0.1)
beta <- grad_step(beta, g, 0.1)
beta
#> AnvlArray
#>  -0.3000
#>  -0.3000
#>  -0.3000
#> [ CPUf32{3} ]
```

## Other `jit()` arguments

### Donating inputs

By default, a compiled executable treats its inputs as read-only: the
R-visible input arrays remain valid after the call, and XLA has to
allocate fresh memory for any output of matching shape. For long
training loops over large parameters, this means every step allocates a
new parameter buffer and leaves the previous one for the garbage
collector.

Via the `donate` argument of
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md), you can
tell XLA that an input will not be used after the call, so it is free to
reuse the input array’s memory for an output.

``` r

step <- jit(function(w, g) w - 0.1 * g, donate = "w")

w <- nv_array(c(1, 2, 3), dtype = "f32")
g <- nv_array(c(0.1, 0.1, 0.1), dtype = "f32")

w <- step(w, g)
w
#> AnvlArray
#>  0.9900
#>  1.9900
#>  2.9900
#> [ CPUf32{3} ]
```

The compiled executable now consumes `w`’s buffer as part of the call.
The caller must not reuse an array that was donated to a function,
otherwise an error is thrown:

``` r

w_old <- nv_array(c(1, 2, 3), dtype = "f32")
w_new <- step(w_old, g)
w_old     # the old buffer has been donated
```

### Device placement

Every `AnvlArray` lives on a concrete device (CPU, a specific GPU,
etc.), and a compiled executable is itself bound to a specific device –
so all inputs to one call must live on that same device. How the device
is chosen for a
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)-compiled
call depends on how you called
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md). The two
most relevant options are:

- **Inferred from inputs (default).** If you don’t pass `device =`, anvl
  looks at the devices of the array inputs at call time, requires them
  to agree, and compiles for that device. If there are no array inputs,
  it falls back to the
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md).
- **Pinned at
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) time.**
  Passing a concrete device – e.g. `jit(f, device = "cpu:0")` or
  `jit(f, device = nv_device("cuda:0"))` – forces every call to run on
  that device. Inputs living on a different device are copied over
  automatically.

See the documentation of the `device` arg in
[`help(jit)`](https://r-xla.github.io/anvl/dev/reference/jit.md) for
more information.

``` r

add <- jit(function(x, y) x + y)
add_cpu <- jit(function(x, y) x + y, device = "cpu")

a <- nv_array(c(1, 2, 3), dtype = "f32")
b <- nv_array(c(4, 5, 6), dtype = "f32")

device(add(a, b))                            # inferred from inputs
#> <CpuDevice(id=0)>
device(add_cpu(a, b))                        # pinned
#> <CpuDevice(id=0)>
```

Because the device is part of the cache key (see *The compilation cache*
above), a single
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted function
can hold compiled binaries for several devices at once, unless `device`
was specified explicitly during
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).
