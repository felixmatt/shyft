from __future__ import print_function

import os.path
from setuptools import setup, find_packages
import subprocess


# Build shyft using the build_script
if (not os.path.exists("shyft/api.py") or
    not os.path.exists("shyft/_api.so")):
    try:
        print(subprocess.check_output(
            "sh build_api.sh", shell=True))
    except:
        print("Problems compiling shyft, try building with the build_api.sh script manually...")
    else:
        if not os.path.exists("shyft/_api.so"):
            print("### ERRORS ### \n\n"
                  "Check build messages for errors. Seems Release\n "
                  "version of shyft/_api.so does not exist.")

if (os.path.exists("shyft/_api.so") and
    os.path.exists("shyft/api.py")):
    print("### SUCCESS BUILDING SHyFT ### \n\n"
          "Looks like shyft is built correctly.\n "
          "Be sure to:: \n export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH \n".format(
          os.path.join(os.path.abspath(os.path.curdir), "shyft")))


# SHyFT version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('shyft/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

setup(
    name='shyft',
    version=VERSION,
    author='Statkraft',
    author_email='shyft@statkraft.com',
    url='http://www.opensource-enki.com',
    description='A modular hydrologic framework',
    license='LGPL v3',
    packages=find_packages(),
    package_data={'shyft': ['_api.so', 'tests/netcdf/*']},
    entry_points={
        'console_scripts': [
            #################### orchestration #######################
            'shyft_runner = orchestration2.shyft_runner:main',
        ]
    },
    requires=["numpy", "nose", "netCDF4"]
)
