README
======

|Branch      |Status   |
|------------|---------|
|master       | [![Build Status](https://travis-ci.org/statkraft/shyft.svg?branch=master)](https://travis-ci.org/statkraft/shyft) |

SHyFT is an OpenSource hydrological toolbox developed by
[Statkraft](http://www.statkraft.com).

It is optimized for highly efficient modeling of hydrologic processes
following the paradigm of distributed, lumped parameter models -- though
recent developments have introduced more physically based / process-level
methods.

The code is based on an early initiative for distributed hydrological
simulation, called [ENKI](https://bitbucket.org/enkiopensource/enki)
developed at Sintef by Sjur Kolberg with contributions from Kolbjorn
Engeland and Oddbjorn Bruland.

REQUIREMENTS
============

For compiling and running SHyFT, you will need:

* A C++11 compiler
* The BLAS and LAPACK libraries (development packages)
* A Python3 (3.4 or higher) interpreter
* The SWIG wrapping tool (>= 3.0.5)
* The NumPy package (>= 1.8.0)
* The netCDF4 package (>= 1.2.1)
* The CMake building tool (2.8.7 or higher)

In addition, a series of Python packages are needed mainly for running
the tests.  These can be easily installed via:

```bash
$ pip install -r requeriments.txt
```

or, if you are using conda:

```bash
$ cat requirements.txt | xargs conda install
```

COMPILING
=========

NOTE: the compiling instructions below have been mainly tested on
Linux platforms.  SHyFT can also be compiled (and it is actively
maintained) for Windows, but the building instructions are not covered
here (yet).

You can compile SHyFT by using the typical procedure for Python packages:

```bash
$ python setup.py build
```

from the root directory.

Although SHyFT tests are meant to be run from the sources directory
(e.g. it expects the shyft-data repo to be cloned locally next to the
shyft repo sources), you can also install it with:

```bash
$ python setup.py install
```

Although you won't be able to run the tests except in a very
restricted scenario (i.e. `shyft-data` should be a sibling of the
current working directory), this won't prevent you to use SHyFT on top
of your own datasets.


COMPILING MANUALLY VIA CMAKE
============================

Although (at least on Linux) the `setup.py` method above uses the
CMake building tool behind the scenes, you can also compile it
manually (in fact, this is recommended because you will be able to run
the integrated C++ tests).  The steps are the usual ones:

```bash
$ export SHYFT_SOURCES=shyft_root_directory  # absolute path required!
$ cd $SHYFT_SOURCES
$ mkdir build
$ cd build
$ export SHYFT_DEPENDENCIES_DIR=directory_to_keep_dependencies  # absolute path
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

If all the tests pass, then you have an instance of SHyFT that is
fully functional.  In case this directory is going to act as a
long-term installation it is recommended to persist your
$LD_LIBRARY_PATH and $PYTHONPATH environment variables (in ~/.bashrc
or similar).


TESTING
=======

The way to test SHyFT is by running:

```bash
$ nosetests
```
from the root directory.

The test suite is not very comprehensive yet, but at least would provide
indications that your installation is sane.

INSTALLING
==========

Once you tested you SHyFT package you can install it in your system via::

```bash
$ python setup.py install
```

AUTHORS
=======

SHyFT is developed by Statkraft, and the two main initial authors to
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
SHyFT is released under LGPL V.3
See LICENCE
