# Source torch-based integration tests if torch is available
if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "test-integration-torch.R", package = "anvl"), local = TRUE)
}
