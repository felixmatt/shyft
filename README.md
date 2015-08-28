README	
====================
SHyFT is an OpenSource hydrological toolbox developed by
[Statkraft](http://www.statkraft.com). 

The code is based on an early initiative for distributed hydrological
simulation, called [ENKI](http://www.opensource-enki.org/>).

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
SHyFT is developed by Statkraft, and the two main initial authors were
Sigbjørn Helset <Sigbjorn.Helset@statkraft.com> and Ola Skavhaug <ola@xal.no>.

Several of the methods implemented are rewrites of corresponding code in
[ENKI](http://www.opensource-enki.org/>)

THANKS
====================

Contributors and project participants include:
 * John Burkhart <John.Burkhart@statkraft.com>
 * Felix Matt <f.n.matt@geo.uio.no>
 * Francesc Alted <faltet@gmail.com>
 * Yisak Sultan Abdella <YisakSultan.Abdella@statkraft.com>


COPYING / LICENSE
====================
SHyFT is released under LGPL V.3
See LICENCE
