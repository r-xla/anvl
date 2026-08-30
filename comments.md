## 1. gradient:

> jit(\(x) gradient(identity)(x))(1)
$x
AnvlArray
 1
[ CPUf32{} ]

>


vs

> jit(\() gradient(identity)(1))()
Error in `compute_requirements()` at anvl/R/reverse.R:109:3:
! Cannot compute gradient with respect to `x`.
✖ It was passed as a plain R value
ℹ Pass it as an <AnvlArray>.
Run `rlang::last_trace()` to see where the error occurred.
>


--> What should we do here?
During tracing, we kind of want to treat `RData` as an R value,
but here the analogy breaks.
But if something is an `RData` object, we know it is not static.
Just for the `jit(\() gradient(identity)(1))()` case
we need to be conservative and reject 1 as we don't know whether it should
be interpreted as static or not.

Maybe we can give `gradient()` an optional static argument so we can make the followin g work:

```r
f <- function(x) {
  x + 2
}
gradient(f, static = character())(1)
```

or maybe we should just remove the input check on the wrt for inputs that are literals / arrays, because for those we can't tell what it is.


Conclusion:
- I think it's fine the way it is.
- Should docuement `gradient()` accordingly.

## 2. RData

When do we really need RData?
Aren't RData objects not always without concrete data?


trace_fn(\(x) {print(x); print(as_anvl_array(1)); x + nv_scalar(3, "f64")}, list(x = nv_aval("double", c()))) -> l

GraphBox(GraphRData(RData(double, ()))) 
GraphBox(GraphLiteral(1, f32, ())) 


--> actual data we can just have as literals

Role A — RData(NULL, shape, r_type): irreducible. An argument of a jitted/traced function. There is no data at trace time (the compile cache keys on R type + shape only), and the dtype can't be picked locally: it's whatever the whole body asks for, resolved once at the end by finalize_rdata_inputs() → RDataInput(resolved, ...), plus prim_converts for the other requested dtypes. A literal cannot stand in for this. This is your x in the example.

Role B — RData(<data>, shape): a deferral slot, never a value. You're right that it always ends up as a literal. It is created by to_abstract() (array.R:1342) and boxed by new_rdata_box(), and it always resolves to a GraphLiteral (scalar) or a registered constant (R array) via materialize_rdata(). It never survives:

- all five maybe_box_arrayish() call sites materialize or commit in the same expression — graph.R:1121, graph.R:1275, array.R:288, array.R:369, primitives.R:2341;
- maybe_box_input(mode = "inline") (graph.R:663) explicitly unwraps a concrete rdata box back to aval$data — "hand the body the R value itself";
- a GraphRData node reaches a finished graph only through desc$inputs, which role B never enters.

--> We should remove Role B; the type is currently overloaded I think.
