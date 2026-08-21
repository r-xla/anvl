# run_cpp_tests() requires xml2 to parse the Catch output; skip when it (a
# testthat Suggests dependency) is unavailable, e.g. in noSuggests CI runs.
skip_if_not_installed("xml2")

run_cpp_tests("anvl")
