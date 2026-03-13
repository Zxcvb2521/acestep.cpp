#!/bin/bash

rm -rf build
mkdir build
cd build

cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON
cmake --build . --config Release -j "$(nproc)"
