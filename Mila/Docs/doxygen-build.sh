#!/bin/bash
mkdir -p build
cd build
cmake .. -G Ninja
cmake --build . --target docs
