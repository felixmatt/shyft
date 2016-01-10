#!/bin/bash
# minimialistic script to build shyft python api, 
#given that the 
#  compile dependencies are in place
#  swig >=3.07
#  python 3.x (anaconda tested, other possible as long as we do have numpy for compile)
#  boost 1.60
#  dlib
#  armadillo (including blas/lapack, other options easily availabe)
# notice that you can override 
#    py_include/py_lib defaults to (-I/opt/anaconda/include/python3.4m -I/opt/anaconda/lib/python3.4/site-packages/numpy/core/include)/(-L/opt/anaconda/lib)
#    armadillo_defs/armadillo_libs defaults to (-DARMA_DONT_USE_WRAPPER)/(-lblas -llapack)  alternative if you utilize armadillo (/-larmadillo)
#    boost_include/boost_libs defaults to ()/()
#    shyft_march  defaults to (-march=native))
#  env-var before invoking the script
# Also notice that since we use march=native, this code is meant to run on the cpu you are compiling with.

set -e
exec 3>&1 4>&2

py_include="${py_include:--I/opt/anaconda/include/python3.4m -I/opt/anaconda/lib/python3.4/site-packages/numpy/core/include}"
py_lib="${py_lib:--L/opt/anaconda/lib}"

armadillo_libs="${armadillo_libs:--lblas -llapack}"
armadillo_defs="${armadillo_defs:--DARMA_DONT_USE_WRAPPER -DARMA_DONT_PRINT_ERRORS -DARMA_NO_DEBUG}"

boost_libs="${boost_libs:-}"
boost_include="${boost_include:-}"

shyft_march="${shyft_march:--march=native}"

gcc_opt="-shared -L/usr/local/lib64  -s -pthread -fPIC -std=c++1y -Wno-deprecated-declarations -O3 $shyft_march -I.. -I../../dlib $armadillo_defs $boost_include $py_include"
shyft_common_source="../core/utctime_utilities.cpp ../core/sceua_optimizer.cpp ../core/dream_optimizer.cpp ../api/api.cpp"

cd api/python 
echo "Generating Python interface using swig (this could take some 1..10 minutes if re-generation is needed)"
make -j 4 -f Makefile.swig_run
echo "Done with swig api generation"
cd ../../shyft 
echo "  Compile&Link __init.so (api) started"
g++ $gcc_opt  -o api/___init__.so    ../api/python/__init___wrap.cxx    $shyft_common_source  $armadillo_libs $boost_libs $py_lib &
echo "  Compile&Link _pt_gs_k.so started"
g++ $gcc_opt  -o api/_pt_gs_k.so   ../api/python/pt_gs_k_wrap.cxx   $shyft_common_source  $armadillo_libs $boost_libs $py_lib  &
echo "  Compile&Link _pt_ss_k.so started"
g++ $gcc_opt -o api/_pt_ss_k.so    ../api/python/pt_ss_k_wrap.cxx    $shyft_common_source  $armadillo_libs $boost_libs $py_lib &
echo "  Compile&Link _pt_hs_k.so started"
g++ $gcc_opt -o api/_pt_hs_k.so    ../api/python/pt_hs_k_wrap.cxx    $shyft_common_source  $armadillo_libs $boost_libs $py_lib &
echo -n "Waiting for the background compilations to complete..(could take some minutes)"
wait
echo "."
echo "All finished"
cd ..
exec 3>&- 4>&-

