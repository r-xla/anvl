# Internals

## Transforming Code

While a real anvil is made for reshaping metal, this package is a tool
for reshaping code. We refer to such a rewriting of code as a
**transformation**, of which there are three types:

1.  `R` \\\rightarrow\\ `AnvlGraph`: Generic `R` functions are too
    complicated to handle, so the first step in {anvl} is always to
    convert them into a computational `AnvlGraph` object via
    **tracing**. Such an `AnvlGraph` is similar to `Jaxpr` objects in
    JAX. It operates only on `GraphNode`s – the graph’s stand-ins for
    arrays – and applies `AnvlPrimitive` operations to them.
2.  `AnvlGraph` \\\rightarrow\\ `AnvlGraph`: It is possible to transform
    `AnvlGraph`s into other `AnvlGraph`s. Their purpose is to change the
    functionality of the code. At the time of writing, there is
    essentially only one such transformation, namely reverse-mode
    automatic differentiation via
    [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md).
3.  `AnvlGraph` \\\rightarrow\\ executable: In order to perform the
    actual computation, the `AnvlGraph` needs to be converted into an
    executable. The main backend is `"pjrt"` (via {stablehlo} and
    {pjrt}, compiling with XLA). There is also an experimental
    [quickr](https://github.com/t-kalinowski/quickr) backend.

### Tracing R Functions into Graphs

All functionality in the {anvl} package is centered around the
`AnvlGraph` class. While it is in principle possible to create
`AnvlGraph`s by hand, these are usually created by tracing R functions.
In general, when we want to convert some code into another form (in our
case, R code into an `AnvlGraph`), there are two approaches:

1.  Static analysis, which would require operating on the abstract
    syntax tree (AST) of the code.
2.  Dynamic analysis (aka “tracing”), which executes the code and
    records selected operations.

The former approach is followed by the {quickr} package, while we go
with tracing. We start with a simple yet illustrative example that
either adds or multiplies two inputs `x` and `y` depending on the value
of `op`.

``` r

library(anvl)
f <- function(x, y, op) {
  if (op == "add") {
    nv_add(x, y)
  } else if (op == "mul") {
    nv_mul(x, y)
  } else {
    stop("Unsupported operation")
  }
}
```

To do this, we use
[`anvl::trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md),
which takes in an `R` function and a list of `AbstractArray` inputs that
specify the input types.

``` r

aten <- nv_aval("f32", c())
aten
```

    ## AbstractArray(dtype=f32, shape=)

``` r

graph <- trace_fn(f, list(x = aten, y = aten, op = "mul"))
graph
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Body:
    ##     %1: f32[] = mul(%x1, %x2)
    ##   Outputs:
    ##     %1: f32[]

The output of
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md)
is now an `AnvlGraph` object that represents the computation. The fields
of the `AnvlGraph` are:

- `inputs`, which are `GraphValue`s that represent the inputs to the
  function.
- `outputs`, which are `GraphValue`s that represent the outputs of the
  function.
- `calls`, which are `PrimitiveCall`s that take in `GraphNode`s (and
  parameters) and produce output `GraphValue`s.
- `constants`, which are the `GraphValue`s for values closed over by the
  traced function (see *Constant Handling*).
- `in_tree`, `out_tree`, which record the nesting structure of the
  function’s inputs and outputs.

What happens during
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md)
is that a new `GraphDescriptor` is created and the inputs `x` and `y`
are converted into `GraphBox` objects. Then, the function `f` is simply
evaluated with the `GraphBox` objects as inputs. During this evaluation,
we need to distinguish between two cases:

1.  A “standard” `R` function is called: Here, nothing special happens
    and the function is simply evaluated.
2.  An `anvl` function is called: Here, the operation that underlies the
    function is recorded in the `GraphDescriptor`.

The evaluation of the `if` statement is an example for the first
category. Because we set `op = "mul"`, only the second branch is
executed. Then, we are calling `nv_mul`, which attaches a
`PrimitiveCall` that represents the multiplication of the two arrays to
the `$calls` of the `GraphDescriptor`. Note that `nv_mul` is itself not
a primitive: it performs some type promotion and broadcasting if needed
before calling into the primitive `prim_mul`.

A `PrimitiveCall` object consists of the following fields:

- `primitive`: The primitive function that was called.
- `inputs`: The inputs to the primitive function.
- `params`: The parameters (non-arrays) to the primitive function.
- `outputs`: The outputs of the primitive function.

When the evaluation of `f` is complete, the `$outputs` field of the
`GraphDescriptor` is set and the `AnvlGraph` is subsequently created
from the `GraphDescriptor`. The only difference between the `AnvlGraph`
and the `GraphDescriptor` is that the latter has some utility fields
that are useful during graph creation, but for the purposes of this
tutorial, you can think of them as being the same.

### Transforming Graphs into other Graphs

Once the `R` function is staged out into a simpler format, it is ready
to be transformed. The {anvl} package does not in any way dictate how
such an `AnvlGraph` to `AnvlGraph` transformation can be implemented.
For most interesting transformations, however, we need to store some
information for each {anvl} primitive function. In the case of the
gradient, we need to store the derivative rules. For this, the
`AnvlPrimitive` metadata object attached to each primitive has a `rules`
field that can be populated. The derivative rules are stored as
functions under the `"reverse"` name. Each primitive is an exported
`prim_*` function; `[[` on it reads a rule:

``` r

prim_mul[["reverse"]]
```

    ## $forward
    ## NULL
    ## 
    ## $backward
    ## function (inputs, outputs, grads, params, required) 
    ## {
    ##     lhs <- inputs[[1L]]
    ##     rhs <- inputs[[2L]]
    ##     grad <- grads[[1L]]
    ##     list(if (required[[1L]]) prim_mul(grad, rhs), if (required[[2L]]) prim_mul(grad, 
    ##         lhs))
    ## }
    ## <environment: namespace:anvl>
    ## 
    ## attr(,"class")
    ## [1] "anvl_rule_reverse"

The
[`transform_gradient()`](https://r-xla.github.io/anvl/dev/reference/transform_gradient.md)
function uses these rules to compute the gradient of a function. For
this specific transformation, we walk the graph backwards and apply the
derivative rules, which appends the “reverse pass” to the graph. Besides
the forward graph, the transformation takes in the `wrt` argument, which
specifies with respect to which arguments to compute the gradient.

``` r

bwd_graph <- transform_gradient(graph, wrt = c("x", "y"))
bwd_graph
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Constants:
    ##     %c1: f32[]
    ##   Body:
    ##     %1: f32[] = mul(%x1, %x2)
    ##     %2: f32[] = mul(%c1, %x2)
    ##     %3: f32[] = mul(%c1, %x1)
    ##   Outputs:
    ##     %2: f32[]
    ##     %3: f32[]

### Lowering a Graph

In order to execute an `AnvlGraph`, we need to convert it into a – wait
for it – executable. Here, we show how to compile using the PJRT
backend. First, we will translate the `AnvlGraph` into the StableHLO
representation via the {stablehlo} package. Then, we will compile this
program using the XLA compiler that is accessible via the {pjrt}
package.

Like for the gradient transformation, the rules of how to do this
transformation are attached to each primitive.

``` r

prim_mul[["stablehlo"]]
```

    ## function (lhs, rhs, output_types) 
    ## {
    ##     list(hlo_multiply(lhs, rhs, output_types = output_types))
    ## }
    ## <environment: namespace:anvl>

The
[`stablehlo()`](https://r-xla.github.io/anvl/dev/reference/stablehlo.md)
function creates a
[`stablehlo::Func`](https://r-xla.github.io/stablehlo/reference/Func.html)
object and sequentially translates the `PrimitiveCall`s into StableHLO
operations.

``` r

func <- stablehlo(graph)[[1L]]
func
```

    ## func.func @main (%0: tensor<f32>, %1: tensor<f32>) -> tensor<f32> {
    ## %2 = stablehlo.multiply %0, %1 : tensor<f32>
    ## return %2 : tensor<f32>
    ## }

Now, we can compile the function via `pjrt_compile()`.

``` r

hlo_str <- stablehlo::repr(func)
program <- pjrt::pjrt_program(src = hlo_str, format = "mlir")
exec <- pjrt::pjrt_compile(program)
```

To run the function, we need to extract the underlying buffers from the
arrays before passing them to the executable, which will output a
`PJRTBuffer` that we can easily convert to an `AnvlArray`.

``` r

x <- nv_scalar(3, "f32")
y <- nv_scalar(4, "f32")
out <- pjrt::pjrt_execute(exec, x$data, y$data)
out
```

    ## PJRTBuffer 
    ##  12
    ## [ CPUf32{} ]

``` r

nv_array(out)
```

    ## AnvlArray
    ##  12
    ## [ CPUf32{} ]

## The User Interface

In the previous section, we have shown how the transformations are
implemented under the hood. The actual user interface is a little more
convenient and follows JAX’s interface.

### `jit()`

The [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)
function allows converting a regular `R` function into a just-in-time
compiled function that can be executed on `AnvlArray`s. We apply it to
our simple example function, where we mark the non-array parameter `op`
as “static”. This means that the value of this parameter needs to be
known at compile time.

``` r

f_jit <- jit(f, static = "op")
f_jit(x, y, "add")
```

    ## AnvlArray
    ##  7
    ## [ CPUf32{} ]

One might think that
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) first calls
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md),
then runs
[`stablehlo()`](https://r-xla.github.io/anvl/dev/reference/stablehlo.md),
followed by `pjrt_compile()`. This is, however, not what is happening,
as this requires the input types to be known. Instead, `f_jit` is a
“lazy” function that will only perform these steps once the inputs are
provided. However, if those steps were applied every time the `f_jit`
function is called, this would be very inefficient, because tracing and
compiling take some time. Therefore, the function `f_jit` also contains
a cache (implemented as an
[`xlamisc::LRUCache`](https://rdrr.io/pkg/xlamisc/man/LRUCache.html)),
which will check whether there is already a compiled executable for the
given inputs. For this, the types of all `AnvlArray`s need to match
exactly (data type and shape) and all static arguments need to be
identical. For example, if we run the function with `AnvlArray`s of the
same type, but different values, the function won’t be recompiled, which
we can see by checking the size of the cache, which is already 1,
because we have called it on `x` and `y` above.

``` r

cache_size <- function(f) environment(f)$cache$size
cache_size(f_jit)
```

    ## NULL

After calling it with arrays of the same types and identical static
argument values, the size of the cache remains 1:

``` r

f_jit(nv_scalar(-99, "f32"), nv_scalar(2, "f32"), "add")
```

    ## AnvlArray
    ##  -97
    ## [ CPUf32{} ]

``` r

cache_size(f_jit)
```

    ## NULL

When we execute the function with arrays of different `dtype` or
`shape`, the function will be recompiled:

``` r

f_jit(nv_scalar(1, "i32"), nv_scalar(2, "i32"), "add")
```

    ## AnvlArray
    ##  3
    ## [ CPUi32{} ]

``` r

cache_size(f_jit)
```

    ## NULL

Also, if we provide different values for static arguments, the function
will be recompiled:

``` r

f_jit(nv_scalar(1, "f32"), nv_scalar(2, "f32"), "mul")
```

    ## AnvlArray
    ##  2
    ## [ CPUf32{} ]

``` r

cache_size(f_jit)
```

    ## NULL

### `gradient()`

Just like [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md),
[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
also returns a function that will lazily create the graph and transform
it, once the inputs are provided.

``` r

g <- gradient(f, wrt = c("x", "y"))
```

To actually compute the gradient, we wrap it in
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md):

``` r

g_jit <- jit(g, static = "op")
g_jit(x, y, "add")
```

    ## $x
    ## AnvlArray
    ##  1
    ## [ CPUf32{} ] 
    ## 
    ## $y
    ## AnvlArray
    ##  1
    ## [ CPUf32{} ]

We can also use `g` inside another function:

``` r

h <- function(x, y) {
  z <- nv_add(x, y)
  g(z, x, "mul")
}
h_jit <- jit(h)
h_jit(x, y)
```

    ## $x
    ## AnvlArray
    ##  3
    ## [ CPUf32{} ] 
    ## 
    ## $y
    ## AnvlArray
    ##  7
    ## [ CPUf32{} ]

So, what is happening here? Once the inputs `x` and `y` are provided to
`h_jit`, a new `GraphDescriptor` is created and the inputs `x` and `y`
are converted into `GraphBox` objects. Then, the addition of `x` and `y`
is recorded in the `GraphDescriptor`. The call into `g()` is a bit more
involved. First, a new `GraphDescriptor` is created and the forward
computation of `g` is recorded. Subsequently, the reverse pass will be
added to the descriptor, after which it will be converted into an
`AnvlGraph`. This `AnvlGraph` will then be inlined into the parent
`GraphDescriptor` (representing the whole function `h`), which is then
converted into the main `AnvlGraph`. We can look at this graph below,
where `trace_fn` internally converts the `AnvlArray`s `x` and `y` into
their abstract representation.

``` r

h_graph <- trace_fn(h, list(x = x, y = y))
h_graph
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Constants:
    ##     %c1: f32[]
    ##   Body:
    ##     %1: f32[] = add(%x1, %x2)
    ##     %2: f32[] = mul(%1, %x1)
    ##     %3: f32[] = mul(%c1, %x1)
    ##     %4: f32[] = mul(%c1, %1)
    ##   Outputs:
    ##     %3: f32[]
    ##     %4: f32[]

Afterwards, this graph is lowered to StableHLO and subsequently
compiled.

## More Internals

### Constant Handling

Constants are handled specially in {anvl}. Consider the program below:

``` r

y <- nv_array(rnorm(1000000L))
graph <- trace_fn(function(x) {
  x + y + 1
}, list(x = nv_scalar(1L)))
graph
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: i32[]
    ##   Constants:
    ##     %c1: f32[1000000]
    ##   Body:
    ##     %1: f32[] = convert [dtype = f32] (%x1)
    ##     %2: f32[1000000] = broadcast_in_axes [shape = 1000000, broadcast_axes = <any>] (%1)
    ##     %3: f32[1000000] = add(%2, %c1)
    ##     %4: f32[1000000] = broadcast_in_axes [shape = 1000000, broadcast_axes = <any>] (1:f32)
    ##     %5: f32[1000000] = add(%3, %4)
    ##   Outputs:
    ##     %5: f32[1000000]

Here, `y` is a closed-over constant and it is included in the
`$constants` field of the graph, just like the literal `1`.

``` r

graph$constants
```

    ## [[1]]
    ## GraphValue(ConcreteArray(f32, (1000000)))

When compiling such a program to StableHLO, constants are treated
differently depending on their shape (we follow JAX’s approach here).
That is, constants with 1 element are **inlined** into the program,
whereas other constants are added as inputs to the StableHLO program.
This is because inlining large constants into the executable is
inefficient. However, if we didn’t inline small scalars, the compiler
would be unable to do constant folding.

``` r

out <- stablehlo(graph)
out[[1L]]
```

    ## func.func @main (%0: tensor<1000000xf32>, %1: tensor<i32>) -> tensor<1000000xf32> {
    ## %2 = "stablehlo.convert" (%1): (tensor<i32>) -> (tensor<f32>)
    ## %3 = "stablehlo.broadcast_in_dim" (%2) {
    ## broadcast_dimensions = array<i64>
    ## }: (tensor<f32>) -> (tensor<1000000xf32>)
    ## %4 = stablehlo.add %3, %0 : tensor<1000000xf32>
    ## %5 = "stablehlo.constant" () {
    ## value = dense<1.00000000e+00> : tensor<f32>
    ## }: () -> (tensor<f32>)
    ## %6 = "stablehlo.broadcast_in_dim" (%5) {
    ## broadcast_dimensions = array<i64>
    ## }: (tensor<f32>) -> (tensor<1000000xf32>)
    ## %7 = stablehlo.add %4, %6 : tensor<1000000xf32>
    ## return %7 : tensor<1000000xf32>
    ## }

``` r

out[[2L]]
```

    ## [[1]]
    ## GraphValue(ConcreteArray(f32, (1000000)))

Also, before compiling, we remove unused constants. Captured constants
can become unused when we apply code transformations like below, where
the gradient of the function w.r.t. `x` does not depend on the captured
`y`:

``` r

f <- function(x) {
x + y
}
transform_gradient(trace_fn(f, list(x = nv_scalar(1))))
```

In principle, the compiler is able to do this itself, but because we
pass constants as inputs to the program, we need to handle it ourselves.

Further note that:

1.  R literals are embedded directly into the program.
2.  Currently, constants with the same value (that refer to different
    `AnvlArray`s) are not deduplicated, which we might change in the
    future.

### R Values in Compiled Programs

R values can appear in two forms in compiled programs:

1.  As constants
2.  As non-static R inputs (`RData`).

The `RData` object can be though of as the dynamic version of an R value
within the program and they behave similarly.

Both have a shape, but no data type.

``` r

trace_fn(function(x) {
  print(x)
  print(shape(x))
  print(dtype(x))
  x
}, list(RData(c(), "integer")))
```

    ## GraphBox(GraphValue(RData(integer, ()))) 
    ## integer(0)

    ## Error:
    ## ! An R value has no data type of its own until it is used.
    ## ℹ `dtype()` is undefined here for the same reason `dtype(1.5)` is: the value
    ##   only takes a data type when it meets a typed array, or when it commits to the
    ##   default ("i32").
    ## ℹ Give it one explicitly with `nv_convert()`.

``` r

trace_fn(function() {
  x <- 1
  print(shape(x))
  print(dtype(x))
  x
}, list())
```

    ## integer(0)

    ## Error:
    ## ! An R value has no data type of its own until it is used.
    ## ℹ `dtype()` is undefined here for the same reason `dtype(1.5)` is: the value
    ##   only takes a data type when it meets a typed array, or when it commits to the
    ##   default ("f32").
    ## ℹ Give it one explicitly with `nv_convert()`.

An `RData` object resolves its data type when something materializes it,
which is either a primitive or a call to
[`apply_promotion()`](https://r-xla.github.io/anvl/dev/reference/apply_promotion.md)
or
[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md).
Generally, there are two situations:

1.  An `RData` object is combined with an object that has a concrete
    data type
2.  None of the inputs to a primitive has a concrete data type.

In the first case, the `RData` object yields
([`promote_rdata_common()`](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md))
to the concrete data type. Below, `nv_aval("integer", c())` is
equivalent to `RData(c(), "integer")`. In the resulting graph, the `%x1`
input has the data type it yielded to, and the `<- integer` records that
the caller supplies it as an R integer, which the runtime uploads at
that data type.

``` r

trace_fn(\(x) {
  prim_add(x, nv_scalar(1L, "i64"))
}, list(nv_aval("integer", c())))
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: i64[] <- integer
    ##   Constants:
    ##     %c1: i64[]
    ##   Body:
    ##     %1: i64[] = add(%x1, %c1)
    ##   Outputs:
    ##     %1: i64[]

Yielding stays within the value’s own category, so an R integer meeting
an `f32` is an error rather than a promotion – crossing a category is
the job of the `nv_*` layer.

In the second case, it assumes its default data type:

``` r

trace_fn(\(x) {
  prim_exp(x)
}, list(nv_aval("double", c())))
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: f32[] <- double
    ##   Body:
    ##     %1: f32[] = exp(%x1)
    ##   Outputs:
    ##     %1: f32[]

When the same `RData` input is used at several data types, it is
supplied at the narrowest one that holds them all, and each use site
converts down from it. Below the input is uploaded as `i64`; the `i8`
and `i16` uses convert from it, via `i32` because an R integer is not
built below 32 bits.

``` r

trace_fn(\(x) {
  prim_add(x, nv_scalar(1L, "i8"))
  prim_add(x, nv_scalar(1L, "i16"))
  prim_add(x, nv_scalar(1L, "i64"))
}, list(nv_aval("integer", c())))
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: i64[] <- integer
    ##   Constants:
    ##     %c1: i8[]
    ##     %c2: i16[]
    ##     %c3: i64[]
    ##   Body:
    ##     %1: i32[] = convert [dtype = i32] (%x1)
    ##     %2: i8[] = convert [dtype = i8] (%1)
    ##     %3: i8[] = add(%2, %c1)
    ##     %4: i16[] = convert [dtype = i16] (%1)
    ##     %5: i16[] = add(%4, %c2)
    ##     %6: i64[] = add(%x1, %c3)
    ##   Outputs:
    ##     %6: i64[]

This design tries to balance correctness with hardware compatibility.
Another approach would be to always represent R doubles as `f64`, which
is their natural representation. The problem with this approach is that:

1.  modern accelerators run much faster in `f32` than `f64`, and
2.  some accelerators (such as Metal) do not support `f64` at all.

Therefore, one of the underlying ideas is to only introduce `f64` values
when someone actually requested this data type.

``` r

trace_fn(\(x) {
  prim_add(x, nv_scalar(1, "f64"))
}, list(nv_aval("double", c())))
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: f64[] <- double
    ##   Constants:
    ##     %c1: f64[]
    ##   Body:
    ##     %1: f64[] = add(%x1, %c1)
    ##   Outputs:
    ##     %1: f64[]

Otherwise, the `double` input is fed as an `f32` to the pjrt program:

``` r

trace_fn(\(x) {
  prim_add(x, nv_scalar(1, "f32"))
}, list(nv_aval("double", c())))
```

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: f32[] <- double
    ##   Constants:
    ##     %c1: f32[]
    ##   Body:
    ##     %1: f32[] = add(%x1, %c1)
    ##   Outputs:
    ##     %1: f32[]

There is one special case, however: operations that explicitly request a
data type, such as
[`prim_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_convert.md)
and the
[`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
constructor. If
[`prim_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_convert.md)
were to follow the usual rule of committing its R inputs to their
default data type, then `prim_convert(large_double, "i32")` would first
convert the R `double` to an `f32` (the default data type for floats)
and then to an `i32`, which would result in a loss of precision. In
order to prevent this,
[`prim_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_convert.md)
materializes its input at its natural representation.

``` r

trace_fn(\(x) {
  prim_convert(x, "i32")
}, list(nv_aval("double", c())))
```

    ## Warning: Converting an R double to "i32" brings "f64" into the program.
    ## ✖ An R double cannot be built at "i32" directly, so it is built at "f64" and
    ##   the program converts -- and nothing else here asked for "f64".
    ## ℹ To keep it out, convert in its own category first: `nv_convert(nv_convert(x,
    ##   "f32"), "i32")`. The result differs for values its data type cannot hold
    ##   exactly.

    ## <AnvlGraph>
    ##   Inputs:
    ##     %x1: f64[] <- double
    ##   Body:
    ##     %1: i32[] = convert [dtype = i32] (%x1)
    ##   Outputs:
    ##     %1: i32[]

Because of its possibly problematic implications, a warning message is
thrown when this happens. In the future, we might also implement a
better solution to this problem. One idea would be to convert the
`double` input directly to an `i32` before passing it to the compiled
program.

This is why API functions and primitives should always canonicalize the
inputs right at the beginning, so this problem rarely happens.

## Device Inference in `jit()`

Device handling in
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) is quite
complicated. Some things that are important to be aware of:

1.  We don’t know the inferred device just from looking at the input, as
    we might have something like:
    `jit(\(x) x + nv_scalar(1, device = "cuda"))` where we might only
    learn about the device during tracing. This means the data is only
    converted at the end.

2.  There are different backends. There might be a function like
    `jit(\(dev) nv_scalar(1, device = dev), backend = "auto")`. But with
    the current implementation of
    [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md), the
    tracing is handled by the backend’s `jit` method, so we need to
    determine the backend from the input arguments. Therefore, the
    `device = device_arg("dev")` needs to be specified:

    ``` r

    f <- jit(\(dev) nv_scalar(1, device = dev), backend = "auto", device = device_arg("dev"))
    f(nv_device("cpu", "pjrt"))
    ```

    If we made the main
    [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)
    function trace eagerly, we could determine the backend during
    tracing, but this is not really needed yet.

3.  `device = device_arg()` is only accepted together with
    `backend = NULL` or `backend = "auto"`. A concrete backend combined
    with
    [`device_arg()`](https://r-xla.github.io/anvl/dev/reference/device_arg.md)
    is rejected, because with a concrete backend the device can simply
    be passed via a static argument.

### Nested Inputs and Outputs

TODO

## Dichotomy of anvl functions

Here, we will dig deeper into the dichotomy of {anvl} functions such as
`prim_add`. In the *Get Started* vignette, we have learned that these
functions can either be called directly on `AnvlArray`s to transform
data, or used within
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) blocks to
build up programs. Here, we will explain what this actually does and why
this is possible.

The core problem this dichotomy solves is that it is a mental burden to
always keep two versions of an {anvl} function:

1.  The [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
    version that can be used to transform arrays.
2.  The
    non-[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
    one that can be used to build up programs.

With our implementation, the following is possible:

``` r

library(anvl)
prim_add(nv_scalar(1), nv_scalar(2))
```

    ## AnvlArray
    ##  3
    ## [ CPUf32{} ]

``` r

times_2 <- jit(function(x) {
  nv_mul(x, 2)
})

times_4 <- jit(function(x) {
  times_2(times_2(x))
})

times_2(nv_scalar(2))
```

    ## AnvlArray
    ##  4
    ## [ CPUf32{} ]

``` r

times_4(nv_scalar(2))
```

    ## AnvlArray
    ##  8
    ## [ CPUf32{} ]

Otherwise, we would need the following:

``` r

times_2_r <- function(x) {
  nv_mul(x, 2)
}
times_2_jit <- jit(times_2_r)
times_4 <- jit(function(x) {
  times_2_r(times_2_r(x))
})
```

This is rather cumbersome, as there are always two versions of a
function and the first solution is preferable. Internally, we have
implemented this by wrapping every primitive function in
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) and making
a [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
function behave differently depending on whether we are in another
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) call or
not.

If we are in a
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) call and
call into a `jit(f)`, then `f` is evaluated inline and re-traced.
Otherwise, the standard jit path is followed.

However, for the {anvl} API this now means that special care needs to be
taken that everything works in jit-mode and in eager-mode. The most
important points are:

1.  Canonicalize inputs at the start using
    [`as_anvl_array()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
    /
    [`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md).
2.  Propagate the device from the inputs:
    1.  For functions with dynamic inputs: use `nv_*_like` for constant
        creation and pass input operands
    2.  For functions without dynamic inputs, add `device` arg and pass
        it to constant creators.
