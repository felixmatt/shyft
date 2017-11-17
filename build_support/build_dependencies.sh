#!/bin/bash
export WORKSPACE=$(readlink --canonicalize --no-newline `dirname ${0}`/../..)
# to align the cmake support:
export SHYFT_DEPENDENCIES_DIR=${WORKSPACE}/shyft_dependencies
armadillo_name=armadillo-8.200.2
dlib_name=dlib-19.7
boost_ver=1_65_1
cmake_common="-DCMAKE_INSTALL_MESSAGE=NEVER"
echo ---------------
echo Update/build shyft-dependencies
echo WORKSPACE..............: ${WORKSPACE}
echo SHYFT_DEPENDENCIES_DIR.: ${SHYFT_DEPENDENCIES_DIR}
echo PACKAGES...............: miniconda w/shyft_env, doctest, boost_${boost_ver}, ${armadillo_name}, ${dlib_name} 

# the current versions we are building
mkdir -p ${SHYFT_DEPENDENCIES_DIR}
cd ${SHYFT_DEPENDENCIES_DIR}

if [ ! -d ${armadillo_name} ]; then 
    echo Building ${armadillo_name}
    if [ ! -f ${armadillo_name}.tar.xz ]; then 
        wget  http://sourceforge.net/projects/arma/files/${armadillo_name}.tar.xz
    fi;
    tar -xf ${armadillo_name}.tar.xz
    pushd ${armadillo_name}
    cmake -DCMAKE_INSTALL_PREFIX=${SHYFT_DEPENDENCIES_DIR} -DCMAKE_INSTALL_LIBDIR=lib ${cmake_common}
    make install
    popd
fi;
echo Done ${armadillo_name}

if [ ! -d ${dlib_name} ]; then
    echo Building ${dlib_name}
    if [ ! -f ${dlib_name}.tar.bz2 ]; then
        wget http://dlib.net/files/${dlib_name}.tar.bz2
    fi;
    tar -xf ${dlib_name}.tar.bz2
    pushd ${dlib_name}
    mkdir -p build
    dlib_cfg="-DDLIB_PNG_SUPPORT=0 -DDLIB_GIF_SUPPORT=0 -DDLIB_LINK_WITH_SQLITE3=0 -DDLIB_NO_GUI_SUPPORT=1 -DDLIB_JPEG_SUPPORT=0 -DDLIB_USE_BLAS=0 -DDLIB_USE_LAPACK=0"
    cd build && cmake .. -DCMAKE_INSTALL_PREFIX=${SHYFT_DEPENDENCIES_DIR} -DCMAKE_INSTALL_LIBDIR=lib ${cmake_common} ${dlib_cfg} && cmake --build . --config Release --target dlib && make install
    popd
fi;
echo Done ${dlib_name}

if [ ! -d doctest ]; then
    echo Building doctest
    git clone https://github.com/onqtam/doctest
    pushd doctest
    cmake -DCMAKE_INSTALL_PREFIX=${SHYFT_DEPENDENCIES_DIR} ${cmake_common}
    make install
    popd
fi;
echo Done doctest

cd ${WORKSPACE}
# we cache minconda on travis, so the directory exists, have to check for bin dir:
if [ ! -d miniconda/bin ]; then
    echo Building miniconda
    if [ -d miniconda ]; then
        rm -rf miniconda
    fi;
    if [ ! -f miniconda.sh ]; then
        wget  -O miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi;	
    bash miniconda.sh -b -p ${WORKSPACE}/miniconda
    export PATH="${WORKSPACE}/miniconda/bin:$PATH"
    conda config --set always_yes yes --set changeps1 no
    conda update conda
    conda install numpy
    conda create -c conda-forge -n shyft_env python=3.6 pyyaml numpy libgfortran netcdf4 gdal matplotlib requests nose coverage pip shapely  pyproj
    ln -s ${WORKSPACE}/miniconda/include/python3.6m ${WORKSPACE}/miniconda/include/python3.6
    ln -s ${WORKSPACE}/miniconda/envs/shyft_env/include/python3.6m ${WORKSPACE}/miniconda/envs/shyft_env/include/python3.6 
fi;
echo Done minconda


export PATH="${WORKSPACE}/miniconda/bin:$PATH"
cd ${SHYFT_DEPENDENCIES_DIR}
if [ ! -d boost_${boost_ver} ]; then
    echo Building boost_${boost_ver}
    if [ ! -f boost_${boost_ver}.tar.gz ]; then
        wget -O boost_${boost_ver}.tar.gz http://sourceforge.net/projects/boost/files/boost/${boost_ver//_/.}/boost_${boost_ver}.tar.gz
    fi;
    tar -xf boost_${boost_ver}.tar.gz
    pushd boost_${boost_ver}
    ./bootstrap.sh --prefix=${SHYFT_DEPENDENCIES_DIR}
    boost_packages="--with-system --with-filesystem --with-date_time --with-python --with-serialization"
    ./b2 -j2 -d0 link=shared variant=release threading=multi ${boost_packages}
    ./b2 -j2 -d0 install threading=multi link=shared ${boost_packages}
    popd
fi;
echo  Done boost_${boost_ver}

cd ${WORKSPACE}
if [ -d shyft-data ]; then 
    pushd shyft-data
    git pull >/dev/null
    popd
else 
    git clone https://github.com/statkraft/shyft-data
fi;
echo Done shyft-data
echo Update shyft/shyft/lib with all 3rd party .so so that rpath will work for python extensions
mkdir -p ${WORKSPACE}/shyft/shyft/lib
install  --preserve-timestamps --target=${WORKSPACE}/shyft/shyft/lib ${SHYFT_DEPENDENCIES_DIR}/lib/*.so.*

