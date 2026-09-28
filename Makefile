.PHONY: format format-cpp format-r

format: format-cpp format-r

format-cpp:
	@echo "Formatting C/C++ sources with clang-format"
	@find src -maxdepth 1 \( -name '*.cpp' -o -name '*.h' \) ! -name 'RcppExports.cpp' -print0 \
	  | xargs -0 sh -c 'set -- "$$@"; [ "$$#" -gt 0 ] && clang-format -i "$$@"' _

format-r:
	@echo "Formatting R sources with air"
	@find R tests/testthat inst/extra-tests -maxdepth 1 -name '*.R' ! -name 'RcppExports.R' -print0 \
	  | xargs -0 sh -c 'set -- "$$@"; [ "$$#" -gt 0 ] && air format "$$@"' _
