import os.path
import glob
import platform
from setuptools import setup, find_packages
import subprocess

# Build shyft using the build_script
# Create package using 'python setup.py bdist_conda'

print('Building SHyFT')

if "Windows" in platform.platform():
    msbuild = r'C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe'
    cmd = [msbuild, '/p:Configuration=Release', '/p:Platform=x64'] # '/t:Rebuild'
    
    p = subprocess.Popen(cmd,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    for line in iter(p.stdout.readline, ''):
       print(line.rstrip())
    
elif not glob.glob("shyft/api/___init__*"):
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
              
    finally:
        if glob.glob("shyft/api/___init__*"):
            print("### SUCCESS BUILDING SHyFT ### \n\n"
                "Looks like shyft has been built correctly.\n ")
        else:
            print("### ERRORS ### \n\n"
                "Check build messages for errors. Seems that\n "
                "extensions do not appear in its directory.")

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
    package_data={'shyft': ['api/*.so', 'api/*.pyd', 'tests/netcdf/*']},
    entry_points={
        'console_scripts': [
            #################### orchestration #######################
            'shyft_runner = orchestration2.shyft_runner:main',
        ]
    },
    requires=["numpy", "nose", "netCDF4"]
)
