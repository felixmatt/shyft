#!/bin/bash

set -e
exec 3>&1 4>&2
bash build_support/build_dependencies.sh
WORKSPACE=$(readlink --canonicalize --no-newline `dirname ${0}`/..)
export LD_LIBRARY_PATH=${WORKSPACE}/shyft_dependencies/lib:$LD_LIBRARY_PATH
export PATH=${WORKSPACE}/miniconda/bin:$PATH
mkdir -p build
cd build
cmake ..
make -j 4
make install
