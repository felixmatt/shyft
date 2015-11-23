
import os.path
import glob
import platform
from setuptools import setup, find_packages
import subprocess


# Build shyft using the build_script
if not glob.glob("shyft/api/___init__*"):
    try:
        if "Linux" in platform.platform():
            # For Linux, use the cmake approach for compiling the extensions
            print(subprocess.check_output(
                "sh build_api_cmake.sh", shell=True))
        else:
            print(subprocess.check_output(
                "sh build_api.sh", shell=True))
    except:
        print("Problems compiling shyft, try building with the build_api.sh "
              "or build_api_cmake.sh (Linux only) script manually...")

if glob.glob("shyft/api/___init__*"):
    print("### SUCCESS BUILDING SHyFT ### \n\n"
          "Looks like shyft has been built correctly.\n ")
else:
    print("### ERRORS ### \n\n"
          "Check build messages for errors. Seems that\n "
          "extensions do not appear in its directory.")


# SHyFT version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('shyft/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

setup(
    name='shyft',
    version=VERSION,
    author='Statkraft',
    author_email='shyft@statkraft.com',
    url='https://github.com/statkraft/shyft',
    description='An OpenSource hydrological toolbox',
    license='LGPL v3',
    packages=find_packages(),
    package_data={'shyft': ['api/*.so', 'tests/netcdf/*']},
    entry_points={
        'console_scripts': [
            #################### orchestration #######################
            'shyft_runner = orchestration2.shyft_runner:main',
        ]
    },
    requires=["numpy", "nose", "netCDF4"]
)
