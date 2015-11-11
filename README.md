README
======

|Branch      |Status   |
|------------|---------|
|master       | [![Build Status](https://travis-ci.org/FrancescAlted/shft.svg?branch=master)](https://travis-ci.org/FrancescAlted/shyft) |

SHyFT is an OpenSource hydrological toolbox developed by
[Statkraft](http://www.statkraft.com).

It is optimized for highly efficient modeling of hydrologic processes
following the paradigm of distributed, lumped parameter models -- though
recent developments have introduced more physically based / process-level
methods.

The code is based on an early initiative for distributed hydrological
simulation, called [ENKI](https://bitbucket.org/enkiopensource/enki)
developed at Sintef by Sjur Kolberg with contributions from Kolbjorn Engeland
and Oddbjorn Bruland.

REQUIREMENTS
============

For compiling and running SHyFT, you will need a C++11 compiler, 
Python3 (3.4 or higher), plus a series of packages that can be installed via:

```bash
$ pip install -r requeriments.txt
```

or, if you are using conda:

```bash
$ cat requirements.txt | xargs conda install
```

COMPILING
=========

You can compile SHyFT by using the typical procedure for Python packages:

```bash
$ python setup.py build_ext --inplace
```

from the root directory.

COMPILING VIA CMAKE
=====================

You can also compile SHyFT with CMake building tool which is available
for the most of the platforms out there.  The steps are the usual ones:

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
