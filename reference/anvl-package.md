# anvl: Framework for R code transformations

Code transformation framework for R.

## Third-Party Licenses

The `anvl` package itself is MIT-licensed. The CUDA backend dynamically
loads NVIDIA software which is not bundled with `anvl`, but downloaded
from NVIDIA's official redistributable channels by the CUDA toolkit R
package (e.g. `cuda12.8`) at install time. Its use is governed by the
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

- Daniel Falbel <daniel@posit.co>
  ([ORCID](https://orcid.org/0009-0006-0143-2392))

- Tomasz Kalinowski <tomasz@posit.co>

- Nikolai German <niko.german@gmail.com>
  ([ORCID](https://orcid.org/0009-0001-7394-8367))
