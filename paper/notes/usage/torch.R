# Usage example: torch
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - R bindings to LibTorch (the C++ core of PyTorch), no Python. Downloads LibTorch on first use.
# - Eager: each operation is dispatched from R immediately. No whole-program compiler in R (no
#   `torch.compile()`); `jit_trace()` produces TorchScript, mainly for deployment.
# - AD via `autograd`: tensors with `requires_grad = TRUE` record a tape during the forward pass,
#   `$backward()` walks it.
# - GPU: `device = "cuda"` (NVIDIA) or `"mps"` (Apple).

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

library(torch)

Xt <- torch_tensor(X, dtype = torch_float64())
yt <- torch_tensor(as.double(y), dtype = torch_float64())
b <- torch_tensor(beta0, dtype = torch_float64(), requires_grad = TRUE)

eta <- torch_mv(Xt, b)
u <- -torch_sum(yt * eta - nnf_softplus(eta)) + 0.5 * torch_sum(b^2)
u$backward()
as.numeric(u)
as.numeric(b$grad)
