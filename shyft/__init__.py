from __future__ import print_function
from __future__ import absolute_import

import os

this_dir = __path__[0]
__version__ = "development"
try:
    from .version import __version__
except:
    pass


def _get_hg_description(path_):
    """ Get the output of hg summary when executed in a given path. """
    import subprocess

    # make an absolute path if required, for example when running in a clone
    if not os.path.isabs(path_):
        loc_path = os.path.abspath(os.path.curdir)
        path_ = os.path.join(loc_path, path_)
    # look up the commit using subprocess and hg summary
    try:
        # redirect stderr to stdout to make sure the hg error message in case
        # we are not in a hg repo doesn't appear on the screen and confuse the
        # user.
        out = subprocess.check_output(
            ["hg", "summary"], cwd=path_, stderr=subprocess.STDOUT).strip()
        out = "".join(['\n  ' + l for l in out.split('\n')])
        return out
    except OSError:  # in case hg wasn't found
        pass
    except subprocess.CalledProcessError:  # not in hg repo
        pass


_hg_description = _get_hg_description(this_dir)


def print_versions():
    """Print all the versions for packages that pflexible relies on."""
    import numpy
    import sys

    print("-=" * 38)
    print("shyft version: %s" % __version__)
    if _hg_description:
        print("shyft hg summary:    %s" % _hg_description)
    print("NumPy version:     %s" % numpy.__version__)
    print("Python version:    %s" % sys.version)
    if os.name == "posix":
        (sysname, nodename, release, version_, machine) = os.uname()
        print("Platform:          %s-%s" % (sys.platform, machine))
    print("Byte-ordering:     %s" % sys.byteorder)
    print("-=" * 38)


def run_tests():
    import nose
    nose.main()
