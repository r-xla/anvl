# Subsetting

`AnvlArray`s are subset with `[` and updated with `[<-`, much like R
arrays. Most differences to base R come from two properties of compiled
code: the shape of every result must be known before the program runs
(see the [Static Shape
Restriction](https://r-xla.github.io/anvl/dev/articles/static_shapes.md)
article), and the compiled program cannot throw errors. This article
goes through what that means for selecting and updating elements.

``` r

library(anvl)
x <- nv_array(1:10)
x
```

    ## AnvlArray
    ##   1
    ##   2
    ##   3
    ##   4
    ##   5
    ##   6
    ##   7
    ##   8
    ##   9
    ##  10
    ## [ CPUi32{10} ]

## Selecting Elements

A single index selects one element and drops the axis, just like in R:

``` r

x[2]
```

    ## AnvlArray
    ##  2
    ## [ CPUi32{} ]

To select several elements, pass the indices as an R array.
[`arr()`](https://r-xla.github.io/anvl/dev/reference/arr.md) is a
shorthand for this, e.g. `arr(2, 4, 6)` is the same as
`array(c(2, 4, 6))`:

``` r

x[arr(2, 4, 6)]
```

    ## AnvlArray
    ##  2
    ##  4
    ##  6
    ## [ CPUi32{3} ]

A plain vector such as `c(2, 4, 6)` is not accepted, because it would be
ambiguous for a single index: should `x[c(2)]` drop the axis, like
`x[2]`, or keep it? With arrays, the answer is always clear, so {anvl}
needs no `drop` argument: `x[2]` drops the axis and `x[arr(2)]` keeps
it.

``` r

x[arr(2)]
```

    ## AnvlArray
    ##  2
    ## [ CPUi32{1} ]

Ranges work as in R, including ranges that count down, and an empty
subscript selects everything:

``` r

x[2:5]
```

    ## AnvlArray
    ##  2
    ##  3
    ##  4
    ##  5
    ## [ CPUi32{4} ]

``` r

x[5:2]
```

    ## AnvlArray
    ##  5
    ##  4
    ##  3
    ##  2
    ## [ CPUi32{4} ]

``` r

x[]
```

    ## AnvlArray
    ##   1
    ##   2
    ##   3
    ##   4
    ##   5
    ##   6
    ##   7
    ##   8
    ##   9
    ##  10
    ## [ CPUi32{10} ]

For an `AnvlArray` with several axes, there is one subscript per axis,
and they combine as in R. Axes that are left out at the end select all
of their elements.

``` r

m <- nv_matrix(1:12, nrow = 3, byrow = TRUE)
m
```

    ## AnvlArray
    ##   1  2  3  4
    ##   5  6  7  8
    ##   9 10 11 12
    ## [ CPUi32{3,4} ]

``` r

m[1, ] # the first row
```

    ## AnvlArray
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPUi32{4} ]

``` r

m[arr(1, 3), 2:3] # rows 1 and 3, columns 2 and 3
```

    ## AnvlArray
    ##   2  3
    ##  10 11
    ## [ CPUi32{2,2} ]

``` r

m[2] # also the second row, as the column subscript is left out
```

    ## AnvlArray
    ##  5
    ##  6
    ##  7
    ##  8
    ## [ CPUi32{4} ]

## Indices Given as `AnvlArray`s

So far, all indices were R values. We call such a subset *static*,
because it does not depend on any `AnvlArray`: its values are known in
advance, so {anvl} can check them before anything runs, e.g. whether an
index is out of range (see [indices outside of the
`AnvlArray`](#indices-outside-of-the-anvlarray) below). An index can
also be an `AnvlArray`, e.g. one that is itself the result of a
computation. Such a subset is *dynamic*: its values are only known once
the computation runs, so these checks are not possible. Below, the
position of the largest element is used to select it:

``` r

v <- nv_array(c(3L, 9L, 4L, 1L))
v[nv_which_max(v)]
```

    ## AnvlArray
    ##  9
    ## [ CPUi32{} ]

As with R values, a scalar index drops the axis, while an `AnvlArray`
with one axis keeps it and can select several elements:

``` r

x[nv_array(2L)]
```

    ## AnvlArray
    ##  2
    ## [ CPUi32{1} ]

``` r

x[nv_array(c(2L, 4L, 6L))]
```

    ## AnvlArray
    ##  2
    ##  4
    ##  6
    ## [ CPUi32{3} ]

Because the shape of the result must be known in advance, not every kind
of subscript can be dynamic:

| Subscript       | Static (R value) | Dynamic (`AnvlArray`) |
|-----------------|------------------|-----------------------|
| Single index    | Yes              | Yes                   |
| Several indices | Yes              | Yes                   |
| Range           | Yes              | No                    |
| Logical mask    | No               | No                    |

The size of a range `a:b`, and therefore the shape of the result, would
not be known in advance if `a` or `b` were dynamic. For the same reason,
logical masks such as `x[x > 5]` are not supported at all. The [masking
pattern](https://r-xla.github.io/anvl/dev/articles/static_shapes.html#the-masking-pattern)
section of the Static Shape Restriction article shows how to get the
same results without them. Negative indices, which exclude elements in
R, are not supported either.

## Indices Outside of the `AnvlArray`

In a static subset, an index that is outside of the `AnvlArray` is an
error:

``` r

x[11]
```

    ## Error in `parse_subset_spec()`:
    ## ! The index 11 is out of bounds for axis 1.
    ## ✖ Axis 1 has size 10, so indices must be between 1 and 10.

In a dynamic subset, the index only has a value once the computation
runs, when errors can no longer be thrown (see [compiled code cannot
throw
errors](https://r-xla.github.io/anvl/dev/articles/next_steps.html#compiled-code-cannot-throw-errors)).
Instead, the index is moved to the nearest valid one, so it is worth
being careful with such indices to avoid bugs:

``` r

x[nv_array(c(0L, 20L))]
```

    ## AnvlArray
    ##   1
    ##  10
    ## [ CPUi32{2} ]

## Updating Subsets

Subset assignment uses the same subscripts as selection. The new value
must either have the shape of the selected subset, or be a scalar, which
is then used for every selected element:

``` r

m[, 3] <- nv_array(-(1:3))
m
```

    ## AnvlArray
    ##   1  2 -1  4
    ##   5  6 -2  8
    ##   9 10 -3 12
    ## [ CPUi32{3,4} ]

``` r

m[1, ] <- 0L
m
```

    ## AnvlArray
    ##   0  0  0  0
    ##   5  6 -2  8
    ##   9 10 -3 12
    ## [ CPUi32{3,4} ]

The value must also fit the data type of the `AnvlArray`. This is
different from base R, which changes the type of the object that is
assigned into when the value does not fit, e.g. an integer vector
becomes a double vector:

``` r

r_int <- 1:3
r_int[1] <- 0.5
r_int
```

    ## [1] 0.5 2.0 3.0

``` r

typeof(r_int)
```

    ## [1] "double"

{anvl} never changes the data type of the `AnvlArray` that is being
updated, so the same assignment is an error:

``` r

y <- nv_array(1:3)
y[1] <- 0.5
```

    ## Error:
    ## ! Cannot bring `value` to data type "i32".
    ## ✖ It is an R double, which is only ever built at a data type of its own
    ##   category: a double becomes a float, an integer an integer, a logical a
    ##   "bool".
    ## ℹ Write it in the target's category (e.g. `0L` for an integer data type), or
    ##   convert it with `nv_convert()`.

To store non-integer values, convert the `AnvlArray` to a float data
type first, e.g. with `nv_convert(y, "f32")`.

Writes to dynamic indices outside of the `AnvlArray` are silently
ignored, for the same reason as when selecting (see [indices outside of
the `AnvlArray`](#indices-outside-of-the-anvlarray)). Below, only the
elements 1 and 3 are updated:

``` r

y <- nv_array(1:5)
y[nv_array(c(1L, 100L, 3L))] <- nv_array(c(-1L, -2L, -3L))
y
```

    ## AnvlArray
    ##  -1
    ##   2
    ##  -3
    ##   4
    ##   5
    ## [ CPUi32{5} ]

When the same element is written several times, it is not specified
which of the values ends up in it, and this can differ between devices,
e.g. between the CPU and a GPU:

``` r

y <- nv_array(1:5)
y[arr(1L, 1L, 1L)] <- nv_array(c(10L, 20L, 30L))
y
```

    ## AnvlArray
    ##  30
    ##   2
    ##   3
    ##   4
    ##   5
    ## [ CPUi32{5} ]

### In-place Updates

Outside of [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md),
`y[i] <- value` creates a new `AnvlArray`, even where R would update `y`
in place. This is because {anvl} cannot know whether another R variable
still refers to the same `AnvlArray`, which would then change as well.
To avoid the copy, pass `inplace = TRUE` among the subscripts, which
writes the update into the memory of `y`:

``` r

y <- nv_array(1:5)
y[2:3, inplace = TRUE] <- 0L
y
```

    ## AnvlArray
    ##  1
    ##  0
    ##  0
    ##  4
    ##  5
    ## [ CPUi32{5} ]

[`nv_subset_assign()`](https://r-xla.github.io/anvl/dev/reference/nv_subset_assign.md)
takes the same argument:
`nv_subset_assign(y, 2:3, value = 0L, inplace = TRUE)`.

This consumes the original `AnvlArray`: its memory is
[donated](https://r-xla.github.io/anvl/dev/articles/efficiency.html#donation)
to the result, so any other R variable that referred to it can no longer
be used afterwards.

``` r

y <- nv_array(1:5)
z <- y
y[1, inplace = TRUE] <- -1L
z
```

    ## AnvlArray

    ## Error:
    ## ! called on deleted or donated buffer

Inside a jitted function, `inplace = TRUE` is an error. There, `z` would
stay valid, so the same code would behave differently in eager and in
jit mode. The compiler avoids unnecessary copies inside
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) anyway, so
`inplace` is not needed there.
