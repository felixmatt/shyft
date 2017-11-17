#!/bin/bash
# minimialistic script to build shyft python api, 
#given that the 
#  compile dependencies are in place
#  python 3.x (anaconda tested, other possible as long as we do have numpy for compile)
#  boost 1.60
#  dlib
#  armadillo (including blas/lapack, other options easily availabe)
# notice that you can override 
#    py_include/py_lib defaults to (-I/usr/include/python3.6m -I/usr/lib/python3.6/site-packages/numpy/core/include)/(-L/usr/lib)
#    armadillo_defs/armadillo_libs defaults to (-DARMA_DONT_USE_WRAPPER)/(-lblas -llapack)  alternative if you utilize armadillo (/-larmadillo)
#    boost_include/boost_libs defaults to ()/()
#    shyft_march  defaults to (-march=native))
#  env-var before invoking the script
# Also notice that since we use march=native, this code is meant to run on the cpu you are compiling with.

set -e
exec 3>&1 4>&2

py_include="${py_include:--I../../miniconda/include/python3.6m -I../../miniconda/lib/python3.6/site-packages/numpy/core/include}"
py_lib="${py_lib:--L../../miniconda/lib -lpython3.6m}"

armadillo_libs="${armadillo_libs:--lblas -llapack}"
armadillo_defs="${armadillo_defs:--DARMA_DONT_USE_WRAPPER -DARMA_DONT_PRINT_ERRORS -DARMA_NO_DEBUG}"

boost_libs="${boost_libs:--lboost_python3 -lboost_serialization -lboost_filesystem -lboost_system}"
boost_include="${boost_include:-}"

dlib_lib="${dlib_lib:--ldlib}"
dlib_include="${dlib_include:--I../../shyft_dependencies/include}"

shyft_march="${shyft_march:--march=native}"

gcc_opt="-shared -L../../shyft_dependencies/lib -L/usr/local/lib64 -L/usr/lib64 -L/usr/lib  -pipe -s -pthread -fPIC -std=c++1y -Wno-deprecated-declarations -O3 $shyft_march -I.. -I../../shyft_dependencies/include $dlib_include $armadillo_defs $boost_include $py_include"
r1_opt="-Wl,-rpath=\$ORIGIN/../lib"
r2_opt="-Wl,-rpath=\$ORIGIN/../../lib"

shyft_common_source="../core/utctime_utilities.cpp ../core/sceua_optimizer.cpp ../core/dream_optimizer.cpp ../api/api.cpp ../api/time_series.cpp ../core/core_serialization.cpp ../api/api_serialization.cpp"

cd shyft 
shyft_api_source=`ls ../api/boostpython/api*.cpp`
echo "  Compile&Link _api.so started"
g++ ${r1_opt} $gcc_opt  -o api/_api.so $shyft_api_source                                 $shyft_common_source $dlib_lib $boost_libs $py_lib $armadillo_libs &
echo "  Compile&Link _pt_gs_k.so started"
g++ ${r2_opt} $gcc_opt  -o api/pt_gs_k/_pt_gs_k.so      ../api/boostpython/pt_gs_k.cpp   $shyft_common_source $dlib_lib $boost_libs $py_lib $armadillo_libs &
echo "  Compile&Link _pt_ss_k.so started"
g++ ${r2_opt} $gcc_opt -o api/pt_ss_k/_pt_ss_k.so       ../api/boostpython/pt_ss_k.cpp   $shyft_common_source $dlib_lib $boost_libs $py_lib $armadillo_libs &
echo "  Compile&Link _pt_hs_k.so started"
g++ ${r2_opt} $gcc_opt -o api/pt_hs_k/_pt_hs_k.so       ../api/boostpython/pt_hs_k.cpp   $shyft_common_source $dlib_lib $boost_libs $py_lib $armadillo_libs &
echo "  Compile&Link _hbv_stack.so started"
g++ ${r2_opt} $gcc_opt -o api/hbv_stack/_hbv_stack.so ../api/boostpython/hbv_stack.cpp   $shyft_common_source $dlib_lib $boost_libs $py_lib $armadillo_libs &

echo -n "Waiting for the background compilations to complete..(could take some minutes)"
wait
echo "."
echo "All finished"
cd ..
exec 3>&- 4>&-

