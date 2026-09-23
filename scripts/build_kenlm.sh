#!/bin/bash
# Builds KenLM's lmplz and build_binary into tools/kenlm/bin (used by `sheptun build-lm`).
# Needs cmake and boost from Homebrew: brew install cmake boost
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/tools/kenlm/src"
BIN="$ROOT/tools/kenlm/bin"

[[ -d "$SRC" ]] || git clone --depth 1 https://github.com/kpu/kenlm.git "$SRC"

# Boost >= 1.69 ships boost_system header-only; Homebrew's 1.90 no longer has its cmake config
sed -i '' '/^[[:space:]]*system[[:space:]]*$/d' "$SRC/CMakeLists.txt"

mkdir -p "$SRC/build" "$BIN"
LOG="$SRC/build/build.log"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=6 >"$LOG" 2>&1
make -C "$SRC/build" -j"$(sysctl -n hw.ncpu)" lmplz build_binary >>"$LOG" 2>&1 \
    || { echo "Сборка KenLM упала, лог: $LOG" >&2; exit 1; }
cp "$SRC/build/bin/lmplz" "$SRC/build/bin/build_binary" "$BIN/"
echo "KenLM: $BIN"
