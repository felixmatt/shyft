README	
====================

|Branch      |Status   |
|------------|---------|
|cmake       | [![Build Status](https://travis-ci.org/FrancescAlted/shft.svg?branch=master)](https://travis-ci.org/FrancescAlted/shyft) |

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

COMPILING
=====================
You can compile SHyFT by using the typical procedure for Python packages::
```bash
$ python setup.py build_ext --inplace
```
from the root directory.

TESTING
====================
The way to test SHyFT is by running::
```bash
$ nosetests
```
from the root directory (your will need the numpy and pytest packages).

The test suite is not very comprehensive yet, but at least would provide
indications that your installation is sane.

INSTALLING
====================
Once you tested you SHyFT package you can install it in your system via::
```bash
$ python setup.py install
```
AUTHORS
====================
SHyFT is developed by Statkraft, and the two main initial authors to the C++ core were
Sigbjørn Helset <Sigbjorn.Helset@statkraft.com> and Ola Skavhaug <ola@xal.no>. 

Orchestration and the Python wrappers were originally developed by
John F. Burkhart <john.burkhart@statkraft.com>

Several of the methods implemented are rewrites of corresponding code in
[ENKI](https://bitbucket.org/enkiopensource/enki)

THANKS
====================

Contributors and current project participants include:
 * Sigbjørn Helset <Sigbjorn.Helset@statkraft.com>
 * Ola Skavhaug <ola@xal.no>
 * John Burkhart <John.Burkhart@statkraft.com>
 * Yisak Sultan Abdella <YisakSultan.Abdella@statkraft.com>
 * Felix Matt <f.n.matt@geo.uio.no>
 * Francesc Alted <faltet@gmail.com>



COPYING / LICENSE
====================
SHyFT is released under LGPL V.3
See LICENCE
