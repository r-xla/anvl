# Subsetting

In this vignette, you will learn how to subset arrays in {anvl} and how
to update subsets. Because array shapes in {anvl} programs are static
(see the [Static Shape
Restriction](https://r-xla.github.io/anvl/dev/articles/static_shapes.md)
vignette), only certain subsetting operations are supported and they
come with some surprises.

We start by listing possible subsets and whether they support dynamic
values (arrays that are specified during runtime) or only static values
(e.g., R literals).

| Subset           | Dynamic | Static |
|------------------|---------|--------|
| Single Index     | Yes     | Yes    |
| Multiple Indices | Yes     | Yes    |
| Range            | No      | Yes    |
| Mask             | No      | No     |

Ranges cannot have dynamic values, because then the size of the subset
would be unknown (what’s the size of `a:b` where `a` and `b` are
unknown?). Boolean masks are not supported, because the output shape
depends on the data, which is not known at compile time. For workarounds
(e.g. `sum(x[x > 0])` or `x[mask] <- update`), see the [masking
pattern](https://r-xla.github.io/anvl/dev/articles/static_shapes.html#the-masking-pattern)
section of the Static Shape Restriction vignette. Negative indexing
(e.g., `x[-1]` to exclude elements) is currently also not supported. For
static values, this will throw an error. For dynamic values, negative
indices are treated as out-of-bounds and clamped to the valid range (see
[Out-of-bounds Handling](#out-of-bounds-handling) below). If you are
missing a feature, please open an issue on GitHub.

We will start with subsetting and then move on to subset-assignment.

## Subsetting

### Subsetting 1D arrays

Let’s start with some simple examples of selecting individual elements
from a 1-dimensional array. The index can be either static or dynamic
and we can drop or keep the axis:

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

- Static & Drop:

  ``` r

  x[2]
  ```

      ## AnvlArray
      ##  2
      ## [ CPUi32{} ]

- Static & Keep:

  Here we use
  [`arr()`](https://r-xla.github.io/anvl/dev/reference/arr.md), a
  convenience helper that builds an R array without having to wrap the
  values in [`c()`](https://rdrr.io/r/base/c.html) (so `arr(2L)` is
  equivalent to `array(2L)`, and `arr(1, 2, 3)` to `array(c(1, 2, 3))`).

  ``` r

  x[arr(2L)]
  ```

      ## AnvlArray
      ##  2
      ## [ CPUi32{1} ]

- Dynamic & Drop:

  ``` r

  x[nv_scalar(2L)]
  ```

      ## AnvlArray
      ##  2
      ## [ CPUi32{} ]

- Dynamic & Keep:

  Below, we perform almost the same operation as above, except that we
  use an array of shape `(1)` instead of a scalar with shape `()`. The
  difference is that subsetting with the former will preserve the axis,
  while the latter will drop it, as we have seen above. This ensures
  that the dimensionality of the result is the same for any 1D subset
  specification, and does not suddenly “simplify” the result to 0D.

  ``` r

  x[nv_array(2L)]
  ```

      ## AnvlArray
      ##  2
      ## [ CPUi32{1} ]

Next, we subset multiple elements, where we only have to distinguish
between static and dynamic indices.

- Static

  ``` r

  x[arr(2, 4, 6)]
  ```

      ## AnvlArray
      ##  2
      ##  4
      ##  6
      ## [ CPUi32{3} ]

- Dynamic

  ``` r

  x[nv_array(c(2L, 4L, 6L))]
  ```

      ## AnvlArray
      ##  2
      ##  4
      ##  6
      ## [ CPUi32{3} ]

We use [`arr()`](https://r-xla.github.io/anvl/dev/reference/arr.md) (or
equivalently [`array()`](https://rdrr.io/r/base/array.html)) instead of
a bare R vector, because otherwise the case where we use a length-1
vector would be ambiguous (do we drop or keep the axis?). This allows us
to do without a `drop` parameter.

We can also use a range that can be specified either canonically via
`a:b` or using
[`nv_seq()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md).

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

x[nv_seq(2, 5)]
```

    ## AnvlArray
    ##  2
    ##  3
    ##  4
    ##  5
    ## [ CPUi32{4} ]

Note that the `a:b` syntax works via Non-Standard Evaluation (NSE), so
we can distinguish it from the actual vector `2:5`. Internally, it is
translated to `nv_seq(a, b)`.

A range that counts down selects in reverse, as it does in base R.

``` r

x[5:2]
```

    ## AnvlArray
    ##  5
    ##  4
    ##  3
    ##  2
    ## [ CPUi32{4} ]

It is also possible to select the whole range by omitting the
specification altogether.

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

### Subsetting higher-dimensional arrays

We start by creating a 2-dimensional array.

``` r

x <- nv_matrix(1:12, nrow = 3, byrow = TRUE)
x
```

    ## AnvlArray
    ##   1  2  3  4
    ##   5  6  7  8
    ##   9 10 11 12
    ## [ CPUi32{3,4} ]

Combining subsets just works like one would expect.

``` r

x[1, ]
```

    ## AnvlArray
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPUi32{4} ]

``` r

x[1, 2]
```

    ## AnvlArray
    ##  2
    ## [ CPUi32{} ]

``` r

x[arr(1), 2:3]
```

    ## AnvlArray
    ##  2 3
    ## [ CPUi32{1,2} ]

``` r

x[arr(1, 3), 2:3]
```

    ## AnvlArray
    ##   2  3
    ##  10 11
    ## [ CPUi32{2,2} ]

``` r

x[1:2, 2:3]
```

    ## AnvlArray
    ##  2 3
    ##  6 7
    ## [ CPUi32{2,2} ]

``` r

x[1, 2:3]
```

    ## AnvlArray
    ##  2
    ##  3
    ## [ CPUi32{2} ]

``` r

x[arr(2, 2), ]
```

    ## AnvlArray
    ##  5 6 7 8
    ##  5 6 7 8
    ## [ CPUi32{2,4} ]

``` r

x[arr(2, 2)]
```

    ## AnvlArray
    ##  5 6 7 8
    ##  5 6 7 8
    ## [ CPUi32{2,4} ]

### Out-of-bounds Handling

If one specifies out-of-bounds indices, we can only throw an error if
the indices are static (and therefore known at compile time). The PJRT
backend that {anvl} compiles to does not throw errors for out-of-bounds
dynamic indices, but instead clamps them to the valid range:

``` r

x[nv_array(-1L), nv_array(100L)]
```

    ## AnvlArray
    ##  4
    ## [ CPUi32{1,1} ]

``` r

x[nv_array(1L), nv_array(4L)]
```

    ## AnvlArray
    ##  4
    ## [ CPUi32{1,1} ]

Therefore, you need to be careful when using dynamic indexing in order
to avoid bugs.

## Updating Subsets

Updating subsets supports the same syntax as subsetting. The value to
write must either have the shape of the subset, or be a scalar.

``` r

x
```

    ## AnvlArray
    ##   1  2  3  4
    ##   5  6  7  8
    ##   9 10 11 12
    ## [ CPUi32{3,4} ]

``` r

x[, 3] <- nv_array(-(1:3))
x
```

    ## AnvlArray
    ##   1  2 -1  4
    ##   5  6 -2  8
    ##   9 10 -3 12
    ## [ CPUi32{3,4} ]

``` r

x <- nv_matrix(1:12, nrow = 3, byrow = TRUE)
x[, 3] <- -99L
x
```

    ## AnvlArray
    ##    1   2 -99   4
    ##    5   6 -99   8
    ##    9  10 -99  12
    ## [ CPUi32{3,4} ]

Also, it must have a data type that is convertible to the data type of
the array.

``` r

x <- nv_matrix(1:12, nrow = 3, byrow = TRUE)
x[, 3] <- nv_array(c(1.5, 2.5, 3.5))
```

    ## Error:
    ## ! Cannot bring `value` to data type "i32".
    ## ✖ "f32" is not promotable to "i32".
    ## ℹ Convert it explicitly with `nv_convert()`.

``` r

x
```

    ## AnvlArray
    ##   1  2  3  4
    ##   5  6  7  8
    ##   9 10 11 12
    ## [ CPUi32{3,4} ]

### Out-of-bounds Handling

Similar to subsetting, out-of-bounds indices can only be checked for
static values. For dynamic indices, out-of-bounds writes are simply
ignored:

``` r

x <- nv_array(1:5)
x[nv_array(c(1L, 100L, 3L))] <- nv_array(c(-1L, -2L, -3L))
x
```

    ## AnvlArray
    ##  -1
    ##   2
    ##  -3
    ##   4
    ##   5
    ## [ CPUi32{5} ]

Here, the write to index 100 is silently ignored, while indices 1 and 3
are updated.

### Duplicate Indices

When writing to the same element multiple times, there is no guarantee
which value will be written. Specifically, this might differ between
backends (CPU vs. GPU).

``` r

x <- nv_array(1:5)
x[arr(1L, 1L, 1L)] <- nv_array(c(10L, 20L, 30L))
x
```

    ## AnvlArray
    ##  30
    ##   2
    ##   3
    ##   4
    ##   5
    ## [ CPUi32{5} ]

### In-place Updates

In eager mode, `x[i] <- val` creates a new array by default, even when R
could update `x` in place. This is because {anvl} cannot know whether
another R variable still refers to the same array, which would otherwise
be modified as well. To avoid the copy, pass `inplace = TRUE` among the
subscripts, which writes the update into the memory of `x`:

``` r

x <- nv_array(1:5)
x[2:3, inplace = TRUE] <- 0L
x
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
`nv_subset_assign(x, 2:3, value = 0L, inplace = TRUE)`.

This consumes the original array: its memory is
[donated](https://r-xla.github.io/anvl/dev/articles/efficiency.html#donation)
to the result, so every other R variable that referred to it can no
longer be used afterwards.

``` r

x <- nv_array(1:5)
y <- x
x[1, inplace = TRUE] <- -1L
y
```

    ## AnvlArray

    ## Error:
    ## ! called on deleted or donated buffer

Inside a jitted function, `inplace = TRUE` is an error. There, `y` would
stay valid, so the same code would behave differently in eager and in
jit mode. The compiler avoids unnecessary copies inside
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) anyway, so
`inplace` is not needed there.
