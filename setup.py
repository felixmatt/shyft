import os
from os import path
import sys
import shutil
import glob
import platform
import subprocess
from setuptools import setup, find_packages


print('Building SHyFT')

# VERSION should be set in a previous build step (ex: TeamCity)
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('shyft/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

ext_s = '.pyd' if  'Windows' in platform.platform() else '.so'

ext_names=['shyft/api/_api'+ext_s,
           'shyft/api/pt_gs_k/_pt_gs_k'+ext_s,
           'shyft/api/pt_hs_k/_pt_hs_k'+ext_s,
           'shyft/api/pt_ss_k/_pt_ss_k'+ext_s,
           'shyft/api/hbv_stack/_hbv_stack'+ext_s ]
           
needs_build_ext = not all([path.exists(ext_name) for ext_name in ext_names])

if needs_build_ext:
    print('One or more extension modules needs build, attempting auto build')
    if "Windows" in platform.platform():
        msbuild_2015 = r'C:\Program Files (x86)\MSBuild\14.0\Bin\amd64\MSBuild.exe' if 'MSBUILD_2015_PATH' not in os.environ else os.environ['MSBUILD_2015_PATH']
        msbuild_2017 = r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\amd64\MSBuild.exe' if 'MSBUILD_2017_PATH' not in os.environ else os.environ['MSBUILD_2017_PATH']
        if path.exists(msbuild_2017):
            msbuild = msbuild_2017
            cmd = [msbuild, '/p:Configuration=Release', '/p:Platform=x64','/p:PlatformToolset=v141', '/p:WindowsTargetPlatformVersion=10.0.14393.0', '/m']
        elif path.exists(msbuild_2015):
            msbuild = msbuild_2015
            cmd = [msbuild, '/p:Configuration=Release', '/p:Platform=x64', '/m']
        else:
            print("Sorry, but this setup only supports ms c++ installed to standard locations")
            print(" you can set MSBUILD_2015_PATH or MSBUILD_2017_PATH specific to your installation and restart.")
            exit()
        
        if '--rebuild' in sys.argv:
            cmd.append('/t:Rebuild')
            sys.argv.remove('--rebuild')
        
        p = subprocess.Popen(cmd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

        for line in iter(p.stdout.readline, ''):
           print(line.rstrip())
        
        p.wait()
        if p.returncode != 0:
            print('\nMSBuild FAILED.')
            exit()
        
    elif "Linux" in platform.platform():
        try:
        # For Linux, use the cmake approach for compiling the extensions
            print(subprocess.check_output("sh build_api_cmake.sh", shell=True))
        except:
            print("Problems compiling shyft, try building with the build_api.sh "
                  "or build_api_cmake.sh (Linux only) script manually...")
            exit()
    else:
        print("Only windows and Linux supported")
        exit()
else:
    print('Extension modules are already built in place')

# Copy libraries needed to run Shyft
if "Windows" in platform.platform():
    lib_dir = os.getenv('SHYFT_DEPENDENCIES', '.')
    boost_dll = path.join(lib_dir, 'boost', 'stage', 'lib', '*.dll')
    blas_dir = path.join(lib_dir, 'blaslapack')
    files = glob.glob(boost_dll)
    files += [path.join(blas_dir, 'lapack_win64_MT.dll'), path.join(blas_dir, 'blas_win64_MT.dll')]
    files = [f for f in files if '-gd-' not in path.basename(f)]
    dest_dir = path.join(path.dirname(path.realpath(__file__)), 'shyft', 'lib')
    if not path.isdir(dest_dir):
        os.mkdir(dest_dir)
    for f in files:
        shutil.copy2(f, path.join(dest_dir, path.basename(f)))

requires = ["numpy", "nose", "netcdf4", "pyyaml","six", "pyproj", "shapely" ]
setup(
    name='shyft',
    version=VERSION,
    author='Statkraft',
    author_email='shyft@statkraft.com',
    url='https://github.com/statkraft/shyft',
    description='An OpenSource hydrological toolbox',
    license='LGPL v3',
    packages=find_packages(),
    package_data={'shyft': ['api/*.so', 'api/*.pyd', 'api/pt_gs_k/*.pyd', 'api/pt_gs_k/*.so', 'api/pt_hs_k/*.pyd', 
        'api/pt_hs_k/*.so', 'api/pt_ss_k/*.pyd', 'api/pt_ss_k/*.so', 'api/hbv_stack/*.pyd', 'api/hbv_stack/*.so', 
        'tests/netcdf/*', 'lib/*.dll']},
    entry_points={},
    requires= requires,
    install_requires=requires,
    )