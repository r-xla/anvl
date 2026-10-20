#!/usr/bin/env bash
## ---------------------------------------------------------------------------
## Downloads and builds TestU01 (and, with --practrand, PractRand) into
## `_deps/`, and compiles the harness `testu01` next to this script.
##
##   ./build.sh                 TestU01 + the harness
##   ./build.sh --practrand     also PractRand's `RNG_test`
##
## Needs curl, unzip, a C compiler and make (g++ for PractRand). Re-running it
## skips the steps whose output already exists; delete `_deps/` to start over.
## ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="$HERE/_deps"
PREFIX="$DEPS/testu01-install"
JOBS="$(nproc 2>/dev/null || echo 4)"

# The authors' GitHub repository of TestU01 1.2.3; the zip on their web site
# is the fallback (it answered 404 when this was written).
TESTU01_URLS=(
  "https://github.com/umontreal-simul/TestU01-2009/archive/refs/heads/master.zip"
  "https://simul.iro.umontreal.ca/testu01/TestU01.zip"
)
PRACTRAND_URL="https://downloads.sourceforge.net/project/pracrand/PractRand-pre0.95.zip"

practrand=0
for arg in "$@"; do
  case "$arg" in
    --practrand) practrand=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$DEPS"

# -- TestU01 ------------------------------------------------------------------
if [[ ! -f "$PREFIX/lib/libtestu01.a" ]]; then
  if [[ ! -f "$DEPS/TestU01.zip" ]]; then
    for url in "${TESTU01_URLS[@]}"; do
      echo "downloading $url" >&2
      if curl -fsSL -o "$DEPS/TestU01.zip.part" "$url"; then
        mv "$DEPS/TestU01.zip.part" "$DEPS/TestU01.zip"
        break
      fi
    done
    [[ -f "$DEPS/TestU01.zip" ]] || { echo "could not download TestU01" >&2; exit 1; }
  fi
  rm -rf "$DEPS/testu01-src"
  mkdir -p "$DEPS/testu01-src"
  unzip -q "$DEPS/TestU01.zip" -d "$DEPS/testu01-src"
  src="$(dirname "$(find "$DEPS/testu01-src" -maxdepth 2 -name configure | head -n 1)")"
  echo "building TestU01 in $src" >&2
  # TestU01 (2009) predates GCC 10's -fno-common default and C2x.
  (
    cd "$src"
    CFLAGS="-O2 -fcommon -std=gnu99 -w" ./configure --prefix="$PREFIX" \
      --disable-shared --enable-static > "$DEPS/testu01-configure.log" 2>&1
    make -j"$JOBS" > "$DEPS/testu01-make.log" 2>&1
    make install > "$DEPS/testu01-install.log" 2>&1
  ) || { echo "TestU01 build failed, see $DEPS/testu01-*.log" >&2; exit 1; }
fi

echo "compiling the harness" >&2
cc -O2 -std=gnu99 -Wall -I"$PREFIX/include" -o "$HERE/testu01" "$HERE/testu01.c" \
  "$PREFIX/lib/libtestu01.a" "$PREFIX/lib/libprobdist.a" "$PREFIX/lib/libmylib.a" -lm

# -- PractRand ----------------------------------------------------------------
if [[ "$practrand" == 1 && ! -x "$HERE/RNG_test" ]]; then
  if [[ ! -f "$DEPS/PractRand.zip" ]]; then
    echo "downloading $PRACTRAND_URL" >&2
    curl -fsSL -o "$DEPS/PractRand.zip.part" "$PRACTRAND_URL"
    mv "$DEPS/PractRand.zip.part" "$DEPS/PractRand.zip"
  fi
  rm -rf "$DEPS/practrand-src"
  mkdir -p "$DEPS/practrand-src"
  unzip -q "$DEPS/PractRand.zip" -d "$DEPS/practrand-src"
  src="$(dirname "$(find "$DEPS/practrand-src" -maxdepth 3 -type d -name tools | head -n 1)")"
  echo "building PractRand in $src" >&2
  (
    cd "$src"
    mkdir -p obj
    for f in src/*.cpp src/RNGs/*.cpp src/RNGs/other/*.cpp; do
      g++ -std=c++14 -O3 -w -Iinclude -pthread -c "$f" -o "obj/$(echo "$f" | tr / _).o" &
      while (( $(jobs -r | wc -l) >= JOBS )); do wait -n; done
    done
    wait
    ar rcs libPractRand.a obj/*.o
    g++ -std=c++14 -O3 -w -Iinclude -pthread -o "$HERE/RNG_test" tools/RNG_test.cpp libPractRand.a
  ) > "$DEPS/practrand-build.log" 2>&1 || { echo "PractRand build failed, see $DEPS/practrand-build.log" >&2; exit 1; }
fi

echo "done" >&2
