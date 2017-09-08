README
======

|Branch      |Status   |Docs   |
|------------|---------|---------|
|master       | [![Build Status](https://travis-ci.org/statkraft/shyft.svg?branch=master)](https://travis-ci.org/statkraft/shyft) | [![Doc Development](https://img.shields.io/badge/docs-latest-blue.svg)](http://shyft.readthedocs.io/en/latest/) |

Shyft is an OpenSource hydrological toolbox developed by [Statkraft](http://www.statkraft.com). It is optimized for highly efficient modeling of hydrologic processes following the paradigm of distributed, lumped parameter models -- though recent developments have introduced more physically based / process-level methods.



The code is based on an early [initiative for distributed hydrological simulation](http://www.sintef.no/sintef-energi/xergi/xergi-2004/nr-1---april/betre-tilsigsprognoser-med-meir-informasjon/) , called [ENKI](https://bitbucket.org/enkiopensource/enki) funded by Statkraft and developed at Sintef by Sjur Kolberg with contributions from Kolbjorn Engeland and Oddbjorn Bruland.

IMPORTANT: While Shyft is being developed to support Linux and Windows platforms, it should be noted that the instructions contained in this README are geared toward linux systems. A Wiki for [Shyft](https://github.com/statkraft/shyft/wiki) includes details for how to get prebuilt binaries, a few other build recipes, and how to contribute.

REQUIREMENTS
============

For compiling and running Shyft, you will need:

* A C++1y compiler (gcc-5 or higher)
* The BLAS and LAPACK libraries (development packages)
* A Python3 (3.4 or higher) interpreter
* The NumPy package (>= 1.8.0)
* The netCDF4 package (>= 1.2.1)
* The CMake building tool (2.8.7 or higher)

In addition, a series of Python packages are needed mainly for running the tests. These can be easily installed via:

```bash
$ pip install -r requirements.txt
```

or, if you are using conda (see below):

```bash
$ cat requirements.txt | xargs conda install
```


CLONING
=========
Shyft is distributed in three separate code repositories. This repository, `shyft` provides the main code base. A second repository (required for tests) is located at [shyft-data](https://github.com/statkraft/shyft-data). A third repository [shyft-doc](https://github.com/statkraft/shyft-doc) is available containing example notebooks and tutorials. The three repositories assume they have been checked out in parallel into a `shyft_workspace` directory:

```bash
mkdir shyft_workspace && cd shyft_workspace
export SHYFT_WORKSPACE=`pwd`
git clone https://github.com/statkraft/shyft.git
git clone https://github.com/statkraft/shyft-data.git
git clone https://github.com/statkraft/shyft-doc.git
```

PYTHON SET UP
=============
A general recommendation is to use conda from [Continuum Analytics](http://conda.pydata.org/docs/get-started.html). Below we instruct to create an Anaconda environment. If you are running Shyft, there are several other resources this will provide for plotting and visualization of results -- including jupyter for running the tutorial notebooks. If you prefer a leaner solution, simply use the requirements.txt file included with the repository and a miniconda environment.

Unless you are building from scratch using one of the provided build scripts or would prefer to use an isolated miniconda environment for Shyft, we recommend (and only provide instructions for) setting up a [conda environment](http://conda.pydata.org/docs/using/envs.html#create-an-environment):

```bash
conda create --name shyft python=3.5 anaconda
```

A few other customizations to the environment will help with your workflow. First, define a few `env_vars` for the new environment. The easiest way to do this is to find the directory where you've created your environment (probably `$HOME/.conda/envs/shyft`). Inside that directory create the following files:

```bash
cd $HOME/.conda/envs/shyft
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```

Then edit the `$HOME/.conda/envs/shyft/etc/conda/activate.d/env_vars.sh` file to include the following environment variables:

```
#!/bin/bash
SHYFT_WORKSPACE=/path/to_directory_into_which_you_cloned/shyft-repositories

#SHYFT
export SHYFT_DEPENDENCIES_DIR=$SHYFT_WORKSPACE/shyft-dependencies
export LD_LIBRARY_PATH=$SHYFT_DEPENDENCIES_DIR/local/lib
export PYTHONPATH=$SHYFT_WORKSPACE/shyft
```

Next edit the `$HOME/.conda/envs/shyft/etc/conda/deactivate.d/env_vars.sh` file to unset the environment variables:

```
#!/bin/bash

#SHYFT
unset SHYFT_DEPENDENCIES_DIR
unset PYTHONPATH
```

Now, to build activate the shyft environment and cd to your `$SHYFT_WORKSPACE` directory:

```bash
source activate shyft
cd $SHYFT_WORKSPACE/shyft
```

And you should be ready to build, install, and run Shyft!

BUILDING
========

NOTE: the build/compile instructions below have been mainly tested on Linux platforms. Shyft can also be compiled (and it is actively maintained) for Windows, but the building instructions are not covered here (yet).

NOTE: the dependency regarding a modern compiler generally means gcc-5 is required to build Shyft.


You can compile Shyft by using the typical procedure for Python packages. We use environment variables to control the build. The `SHYFT_DEPENDENCIES_DIR` defines where the dependencies will be built (or exist). When you call `setup.py` the script will call cmake. If the dependencies exist in the aforementioned directory, they will be used. Otherwise, they will be downloaded and built into that directory as part of the build process. If not set, cmake will create a directory `shyft-dependencies` in the `shyft` repository directory. A suggestion is to set the `shyft-dependencies` directory to your `shyft-workspace`. If you have set these as part of your `conda environment` per the instructions above, and assuming you are active in that environment, then simply:
 
 ```bash
 pip install -r requirements.txt
 python setup.py build_ext --inplace 
 
 ```


NOTE: If you haven't set `env_vars` as part of your conda environment, then you need to do the following:

```bash
# assumes you are still in the shyft_workspace directory containing
# the git repositories
export SHYFT_WORKSPACE=`pwd`
mkdir shyft-dependencies
export SHYFT_DEPENDENCIES_DIR=$SHYFT_WORKSPACE/shyft-dependencies
cd shyft #the shyft repository
python setup.py build_ext --inplace
```


QUICK TEST
==========
It is recommended to at least run a few of the tests after building. This will ensure your paths and environment variables are set correctly.

The quickest and easiest test to run is:

```bash
python -c "from shyft import api"
```

If this raises:
`ImportError: libboost_python3.so.1.61.0: cannot open shared object file: No such file or directory`

Then you don't have your `LD_LIBRARY_PATH` set correctly. This should point to:

```bash
export LD_LIBRARY_PATH=$SHYFT_DEPENDENCIES_DIR/local/lib
```

To run further tests, see the TESTING section below. 

INSTALLING
==========

If the tests above run, then you can simply install Shyft using:

```bash
cd $SHYFT_WORKSPACE/shyft
python setup.py install
```

Just be aware of the dependency of the LD_LIBRARY_PATH so that the libboost libraries are found.

Now, you should be set to start working with the [shyft-doc](https://github.com/statkraft/shyft-doc) notebooks and learning Shyft!


COMPILING MANUALLY VIA CMAKE
============================

Although (at least on Linux) the `setup.py` method above uses the
CMake building tool behind the scenes, you can also compile it
manually (in fact, if you plan to develop Shyft, this may be recommended because you will be able to run
the integrated C++ tests).  The steps are the usual ones:

```bash
$ export SHYFT_SOURCES=$SHYFT_WORKSPACE  # absolute path required!
$ cd $SHYFT_SOURCES
$ mkdir build
$ cd build
$ export SHYFT_DEPENDENCIES_DIR=$SHYFT_SOURCES/.. # directory_to_keep_dependencies,  absolute path
$ cmake ..      # configuration step; or "ccmake .." for curses interface
$ make -j 4     # do the actual compilation of C++ sources (using 4 processes)
$ make install  # copy Python extensions somewhere in $SHYFT_SOURCES
```

We have the beast compiled by now.  For testing:

```bash
$ export LD_LIBRARY_PATH=$SHYFT_DEPENDENCIES_DIR/local/lib
$ make test     # run the C++ tests
$ export PYTHONPATH=$SHYFT_SOURCES
$ nosetests ..  # run the Python tests
```

If all the tests pass, then you have an instance of Shyft that is
fully functional.  In case this directory is going to act as a
long-term installation it is recommended to persist your
`$LD_LIBRARY_PATH` and `$PYTHONPATH` environment variables (in `~/.bashrc`
or using the conda `env_vars` described above).


TESTING
=======

The way to test Shyft is by running:

```bash
$ nosetests
```
from the root shyft repository directory.

The test suite is comprehensive, and in addition to unit-tests covering c++ parts and python parts, it also covers integration tests with netcdf and geo-services.

Shyft tests are meant to be run from the sources directory. As a start, you can run the python api test suite by:

```bash
cd $SHYFT_WORKSPACE/shyft/shyft/tests/api
nosetests
```

## Comprehensive Tests

To conduct further testing and to run direct C++ tests, you need to be sure you have the `shyft-data` repository as a sibling of the `shyft` repository directory.

To run some of the C++ core tests you can try the following:

```bash
cd $SHYFT_WORKSPACE/shyft/build/test
make test
```


AUTHORS
=======

Shyft is developed by Statkraft, and the two main initial authors to
the C++ core were Sigbjørn Helset <Sigbjorn.Helset@statkraft.com> and
Ola Skavhaug <ola@xal.no>.

Orchestration and the Python wrappers were originally developed by
John F. Burkhart <john.burkhart@statkraft.com>

Several of the methods implemented are rewrites of corresponding code in
[ENKI](https://bitbucket.org/enkiopensource/enki)

THANKS
======

Contributors and current project participants include:
 * Sigbjørn Helset <Sigbjorn.Helset@statkraft.com>
 * Ola Skavhaug <ola@xal.no>
 * John Burkhart <John.Burkhart@statkraft.com>
 * Yisak Sultan Abdella <YisakSultan.Abdella@statkraft.com>
 * Felix Matt <f.n.matt@geo.uio.no>
 * Francesc Alted <faltet@gmail.com>



COPYING / LICENSE
=================
Shyft is released under LGPL V.3
See LICENCE
