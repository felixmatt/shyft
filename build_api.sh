#!/bin/sh

set -e
tf="%e"
cd api/python 
echo -n "Generating Python interface using swig"
usecs=`time -f $tf -q make -f Makefile.swig_run 2>&1 > /dev/null`
echo ", took $usecs seconds"
t_secs=$usecs

cd ../../shyft 
echo -n "Compiling and linking _api.so, the c++ Python wrapper code for api"
usecs=`time -f $tf -q g++ -shared -L/usr/lib/python2.7 -L/usr/local/lib64 -o api/___init__.so -pthread -s -fPIC -std=c++11 -DARMA_DONT_USE_WRAPPER -I/usr/include/python2.7 -I.. -I../../dlib -O3 -lblas -llapack ../api/python/__init___wrap.cxx  ../core/utctime_utilities.cpp ../core/sceua_optimizer.cpp ../core/dream_optimizer.cpp ../api/api.cpp -lblas -llapack 2>&1 > /dev/null`
echo ", took $usecs seconds"
t_secs=`echo "$t_secs + $usecs" | bc`

echo -n "compiling and linking _pt_gs_k.so, the c++ Python wrapper code for the pt_gs_k method stack"
usecs=`time -f $tf -q g++ -shared -L/usr/lib/python2.7 -L/usr/local/lib64 -o api/_pt_gs_k.so -pthread -s -fPIC -std=c++11 -DARMA_DONT_USE_WRAPPER -I/usr/include/python2.7 -I.. -I../../dlib -O3 -lblas -llapack ../api/python/pt_gs_k_wrap.cxx  ../core/utctime_utilities.cpp ../core/sceua_optimizer.cpp ../core/dream_optimizer.cpp ../api/api.cpp -lblas -llapack 2>&1 > /dev/null`
echo ", took $usecs seconds"
t_secs=`echo "$t_secs + $usecs" | bc`

echo -n "compiling and linking _pt_ss_k.so, the c++ Python wrapper code for the pt_ss_k method stack"
usecs=`time -f $tf -q g++ -shared -L/usr/lib/python2.7 -L/usr/local/lib64 -o api/_pt_ss_k.so -pthread -s -fPIC -std=c++11 -DARMA_DONT_USE_WRAPPER -I/usr/include/python2.7 -I.. -I../../dlib -O3 -lblas -llapack ../api/python/pt_ss_k_wrap.cxx  ../core/utctime_utilities.cpp ../core/sceua_optimizer.cpp ../core/dream_optimizer.cpp ../api/api.cpp -lblas -llapack 2>&1 > /dev/null`
echo ", took $usecs seconds"
t_secs=`echo "$t_secs + $usecs" | bc`

mins=`echo "$t_secs / 60" | bc`
secs=`echo "$t_secs % 60" | bc`
echo Total time: ${mins}m ${secs}s

# Back where we started
cd ..
