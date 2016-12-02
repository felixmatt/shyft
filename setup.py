import os
from os import path
import sys
import shutil
import glob
import platform
import subprocess
from setuptools import setup, find_packages


print('Building SHyFT')

if "Windows" in platform.platform():
    msbuild = r'C:\Program Files (x86)\MSBuild\14.0\Bin\amd64\MSBuild.exe'
    cmd = [msbuild, '/p:Configuration=Release', '/p:Platform=x64', '/m']
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

# VERSION should be set in a previous build step (ex: TeamCity)
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('shyft/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

# Copy libraries needed to run Shyft
if "Windows" in platform.platform():
    lib_dir = os.getenv('SHYFT_DEPENDENCIES', '.')
    boost_dll = path.join(lib_dir, 'boost', 'stage', 'lib', '*.dll')
    blas_dir = path.join(lib_dir, 'blaslapack')
    files = glob.glob(boost_dll)
    files += [path.join(blas_dir, 'lapack_win64_MT.dll'), path.join(blas_dir, 'blas_win64_MT.dll')]
    files = [f for f in files if '-gd-' not in path.basename(f)]
    dest_dir = path.join(path.dirname(path.realpath(__file__)), 'shyft', 'lib')
    os.mkdir(dest_dir)
    for f in files:
        shutil.copy2(f, path.join(dest_dir, path.basename(f)))


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
    requires=["numpy", "nose", "netcdf4"]
)
