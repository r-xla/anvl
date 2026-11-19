# Use the renv library of renv.lock, except in the Docker image of envs/Dockerfile, which installs
# the packages of renv.lock into the library of R. The container is detected via /.dockerenv or, for
# runtimes that do not create that file (e.g., Enroot), via the mount point /mnt/data/paper.
if (!(file.exists("/.dockerenv") || file.exists("/mnt/data/paper"))) {
  source("renv/activate.R")
}
