#!/bin/bash
# Generate the Mila API docs directly from the canonical Doxyfile -- no CMake configure,
# no CUDA toolkit, no Graphviz. Requires only Doxygen 1.17+ on PATH. HTML output lands
# in build/docs/html (the same artifact the docs CI publishes).
set -e
cd "$(dirname "$0")/../.."   # repository root (Doxyfile paths are repo-root-relative)
mkdir -p build/docs          # Doxygen does not create nested OUTPUT_DIRECTORY parents
doxygen Mila/Docs/Doxyfile
echo "Docs generated at build/docs/html/index.html"
