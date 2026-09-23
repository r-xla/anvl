# anvl: Accelerated Array Computing and Automatic Differentiation

Accelerated array computing and code transformations for R. Numerical
programs operating on multi-dimensional arrays can be just-in-time
compiled to optimized executables via 'XLA' – the same compiler that
powers 'JAX' and 'TensorFlow' – and run on CPU or NVIDIA GPU from the
same source. Also provides reverse-mode automatic differentiation,
returning the gradient of a function as another R function.

## Options

- `anvl.backend` (`character(1)`, default `"pjrt"`): the backend every
  operation runs on – `"pjrt"` or `"quickr"`. Arrays are allocated with
  it and jitted functions are compiled for it. Also see
  [`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md),
  [`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
  and
  [`with_backend()`](https://r-xla.github.io/anvl/dev/reference/with_backend.md).

- `anvl.default_dtypes` (named
  [`character()`](https://rdrr.io/r/base/character.html) \| named
  [`list()`](https://rdrr.io/r/base/list.html)): the data types an R
  double and integer materialize at when it cannot be inferred from
  another operand. See
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  for more details.

- `anvl.default_device` (`character(1)` \| device object): the device a
  call that names none allocates on. Also see
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md),
  [`local_default_device()`](https://r-xla.github.io/anvl/dev/reference/local_default_device.md)
  and
  [`with_default_device()`](https://r-xla.github.io/anvl/dev/reference/local_default_device.md).

## Environment variables

- `ANVL_DEFAULT_DTYPES`: `category=dtype` pairs such as
  `"float=f64,int=i64"`, used for every backend when the
  `anvl.default_dtypes` option is not set.

- `ANVL_DEFAULT_DEVICE`: a device identifier such as `"cuda"` or
  `"cpu:1"`, used when the `anvl.default_device` option is not set.

Both are read once, when anvl is loaded.

The remaining ones, all prefixed `ANVL_TEST`, affect only anvl's own
test suite, not the package:

- `ANVL_TEST`: `tests/testthat.R` runs the tests only when this is
  `"1"`, so `R CMD check` in a shell without it runs none of them.

- `ANVL_TEST_SKIP_QUICKR`: when set to anything non-empty, the tests
  that need the quickr backend are skipped – they are comparatively
  slow.

## Third-Party Licenses

The `anvl` package itself is MIT-licensed. The CUDA backend dynamically
loads NVIDIA software which is not bundled with `anvl`, but downloaded
from NVIDIA's official redistributable channels by the CUDA toolkit R
package (e.g. `pjrt.cuda`) at install time. Its use is governed by the
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), with the
exception of cuDNN, which is covered by the [NVIDIA cuDNN
SLA](https://docs.nvidia.com/deeplearning/cudnn/sla/index.html), and
NCCL, which is covered by its [own
license](https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt). By
installing or using the CUDA backend you accept those terms.

## See also

Useful links:

- <https://r-xla.github.io/anvl/>

- <https://github.com/r-xla/anvl>

- Report bugs at <https://github.com/r-xla/anvl/issues>

## Author

**Maintainer**: Sebastian Fischer <seb.fischer@tutamail.com>
([ORCID](https://orcid.org/0000-0002-9609-3197))

Authors:

- Sebastian Fischer <seb.fischer@tutamail.com>
  ([ORCID](https://orcid.org/0000-0002-9609-3197))

- Daniel Falbel <daniel@posit.co>
  ([ORCID](https://orcid.org/0009-0006-0143-2392))

- Tomasz Kalinowski <tomasz@posit.co>

- Nikolai German <niko.german@gmail.com>
  ([ORCID](https://orcid.org/0009-0001-7394-8367))

Other contributors:

- Louis Aslett <louis.aslett@durham.ac.uk>
  ([ORCID](https://orcid.org/0000-0003-2211-233X)) \[contributor\]

- Hugh Graham <hugh@belian.earth>
  ([ORCID](https://orcid.org/0000-0001-9451-5010)) \[contributor\]
