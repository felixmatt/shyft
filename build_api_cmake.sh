#!/bin/bash

set -e
exec 3>&1 4>&2

mkdir -p build
cd build
cmake ..
make -j 2
make install
