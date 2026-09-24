#!/usr/bin/env bash
# Compiles RJwrapper.tex (written by rjtools) to RJwrapper.pdf.
# Older TeX distributions need xparse loaded explicitly for \NewDocumentCommand.
set -euo pipefail
cd "$(dirname "$0")"
grep -q '\\usepackage{xparse}' RJwrapper.tex || sed -i 's/\\usepackage{RJournal}/\\usepackage{RJournal}\n\\usepackage{xparse}/' RJwrapper.tex
pdflatex -interaction=nonstopmode RJwrapper.tex > /dev/null || true
bibtex RJwrapper > /dev/null || true
pdflatex -interaction=nonstopmode RJwrapper.tex > /dev/null || true
pdflatex -interaction=nonstopmode RJwrapper.tex > /dev/null || true
grep -E '^!|Warning: Citation|undefined' RJwrapper.log | sort -u || true
