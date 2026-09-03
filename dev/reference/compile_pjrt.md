# Trace, lower, and compile a function to an XLA executable

Takes a function, traces it into a computational graph, lowers it to
StableHLO, and compiles it to a PJRT executable. Returns the compiled
executable along with metadata needed for execution.

## Usage

``` r
compile_pjrt(
  f,
  args_flat,
  in_tree,
  donate = character(),
  device = NULL,
  arg_devices = list(),
  fallback_device = NULL
)
```

## Arguments

- f:

  (`function`)  
  Function to compile.

- args_flat:

  (`list`)  
  Flat list of abstract input values.

- in_tree:

  (`Node`)  
  Tree structure of the inputs.

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated.

- device:

  (`NULL` \| `character(1)`)  
  Target device (e.g. `"cpu"`, `"cuda"`). If `NULL`, inferred from
  `arg_devices` and traced arrays.

- arg_devices:

  (`list`)  
  Devices of the concrete (non-static) input arguments, extracted before
  converting to abstract values. Used together with traced devices for
  device inference when `device` is `NULL`.

- fallback_device:

  (`NULL` \| device)  
  The device to compile for when `device` is `NULL` and nothing in the
  graph names one. pjrt's dispatcher supplies the device it keyed the
  entry on, so the program and its cache key agree. `NULL` (a caller
  with no dispatcher in front of it) falls back to
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md).

## Value

A `list` with elements:

- `exec`: The compiled PJRT executable.

- `out_tree`: The output tree structure.

- `const_arrays`: Constants needed at execution time.

- `out_avals`: One `list(dtype, shape)` per output leaf; pjrt's
  dispatcher builds the output wrappers from these.

- `input_dtypes`: One entry per input: the dtype an input built from
  bare R data is uploaded at, and `NA` for an array input, which is
  supplied as it is. The R data has no dtype of its own, so the program
  is the only thing that knows what it is uploaded as – pjrt's
  dispatcher therefore requires an entry for every bare R input and
  rejects a dtype declared for an array one. `NULL` for a call whose
  inputs are all arrays.
