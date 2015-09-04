from __future__ import print_function
from __future__ import absolute_import

import os
import os.path


this_dir = __path__[0]
__version__ = "development"
try:
    from .version import __version__
except:
    pass

if "SHYFTDATA" in os.environ:
    shyftdata_dir = os.environ["SHYFTDATA"]
else:
    # If SHYFTDATA environment variable is not here, then use a decent guess
    shyftdata_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "shyft-data")


def print_versions():
    """Print all the versions for packages that SHyFT relies on."""
    import numpy
    import netCDF4
    import sys

    print("-=" * 38)
    print("SHyFT version: %s" % __version__)
    print("NumPy version:     %s" % numpy.__version__)
    print("netCDF4 version:     %s" % netCDF4.__version__)
    print("Python version:    %s" % sys.version)
    if os.name == "posix":
        (sysname, nodename, release, version_, machine) = os.uname()
        print("Platform:          %s-%s" % (sys.platform, machine))
    print("Byte-ordering:     %s" % sys.byteorder)
    print("-=" * 38)


def run_tests():
    import nose
    nose.main()
