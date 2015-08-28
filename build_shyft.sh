#!/bin/bash

# This is a build script for Ubuntu 14.04 and higher. Most dependencies for SHyFT are
# available in the repositories, so apt-get install will provide what is needed.
# We do not provide alternatives for basic build tools (e.g. make, cmake, etc.) but
# for armadillo you can set the flag to build it from source below. The preferred
# method is simply to use::
#    sudo apt-get install libarmadillo4 libarmadillo4-dev
#
# Other dependencies that are header only dependencies are downloaded and installed
# relative to the shyft directory (one level below). The required structure for
# building shyft is to have a ~/projects/shyft directory so that the boost, dlib, and
# cxxtest header dependencies will be installed to ~/projects. This cannot be changed
# as this structure is expected in the make files.
#
# run this script as::
#    $./build_enki.sh
#
# to build and install shyft to the ~/projects/shyft/bin/Release directory
#
#    $./build_enki.sh clean
#
# will clean all build files
#
# 2014.12.20
comment=1
clean=$1
BUILD_ARMADILLO=1

if [ $comment -eq 1 ]; then
  echo """Build script for SHyFT on Ubuntu 14.04 or higher. Untested on other distributions."""
fi

export LD_LIBRARY_PATH=`pwd`/bin/Release
export LIBRARY_PATH=$LD_LIBRARY_PATH

# will install all dependencies in one directory below shyft
shyft_dir=`pwd`
cd ..
PROJ_DIR=`pwd`

# armadillo dependency
if [ ! $clean ] && [ $BUILD_ARMADILLO -eq 1 ]; then

	# if required, we have to build and install libarmadillo4
	# it is available in Ubuntu 14.04+
	CMAKE_PREFIX=$ENKI_DIR/bin/Release
        if [ ! -d "$PROJ_DIR/armadillo-4.550.2" ]; then
	    wget http://sourceforge.net/projects/arma/files/armadillo-4.550.2.tar.gz
	    tar -xf armadillo-4.550.2.tar.gz
        fi
	cd armadillo-4.550.2/
	./configure
	cmake -DCMAKE_INSTALL_PREFIX=$CMAKE_PREFIX .
	make
	make install
	cd ..
	# we link lib armadillo into the build directories for shyft
	cd $CMAKE_PREFIX
        mkdir ../Debug
	ln -s lib/libarmadillo.so libarmadillo.so.4
	ln -s lib/libarmadillo.so libarmadillo.so
	ln -s libarmadillo.so ../Debug/.
	ln -s libarmadillo.so.4 ../Debug/.
	cd -
fi



## Header Dependencies
if [ $comment -eq 1 ]; then
  echo """For the header dependencies we do a simple test if the directory exists. 
          Doesn't guarantee success... """
fi

# boost dependency
DEP_DIR=$PROJ_DIR/boost
if [ ! $clean ] && [ ! -d "$DEP_DIR" ]; then
	wget http://kent.dl.sourceforge.net/project/boost/boost/1.57.0/boost_1_57_0.tar.gz
	tar -xf boost_1_57_0.tar.gz
	ln -s boost_1_57_0 boost
fi
export CPLUS_INCLUDE_PATH=$DEP_DIR

# dlib dependency
DEP_DIR=$PROJ_DIR/dlib
if [ ! $clean ] && [ ! -d "$DEP_DIR" ]; then
	wget http://downloads.sourceforge.net/project/dclib/dlib/v18.11/dlib-18.11.tar.bz2
	tar xjf dlib-18.11.tar.bz2
	ln -sf dlib-18.11 dlib
fi
#export CPLUS_INCLUDE_PATH=$DEP_DIR

# cxxtest installation
DEP_DIR=$PROJ_DIR/cxxtest-4.2
if [ ! $clean ] && [ ! -d "$DEP_DIR" ]; then
	mkdir cxxtest && cd cxxtest
	wget http://downloads.sourceforge.net/project/cxxtest/cxxtest/4.4/cxxtest-4.4.tar.gz
	tar zxf cxxtest-4.4.tar.gz
	ln -s cxxtest-4.4 cxxtest-4.2
	cd ..
fi
#export CPLUS_INCLUDE_PATH=$DEP_DIR



# build shyft
cd $ENKI_DIR
make -f SHyFT.workspace.mak $1

# move Python API to the shyft package
echo "Moving Python API into the shyft package"
mv -v bin/Release/api.py* shyft
mv -v bin/Release/_api* shyft
mv -v bin/Release/libenkicore* shyft

# not needed, module looks at .py location first, and there are no .so
