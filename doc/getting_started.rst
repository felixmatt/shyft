Getting Started
===============

Compiling
---------

You can compile SHyFT by using the typical procedure for Python packages::

  $ python setup.py build_ext --inplace

from the root directory.


Testing
-------

The way to test SHyFT is by running::

  $ nosetests

from the root directory (your will need the numpy and nose packages).

The test suite is not very comprehensive yet, but at least would provide indications that your installation is sane.


Installing
----------

Once you tested you SHyFT package you can install it in your system via::

    $ python setup.py install


Running a small example
-----------------------

Once everything is setup, you can run a small example with::

  $ shyft_runner -c doc/example/netcdf/configuration.yaml -s Himalayas


If you get the result of a simulation, you are done with setting up SHyFT!  Please proceed with the next chapter
so as to see how to calibrate SHyFT and setup your own configurations.
