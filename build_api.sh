echo "updating api.py and api_wrap.cxx using swig"
cd api/python && make -f Makefile.swig_run
echo "compiling and linking the c++ part of the api, the _api.so shared library"
cd ../../shyft && g++ -shared -L/usr/lib/python2.7 -L/usr/local/lib64 -o _api.so -pthread -s -fPIC -std=c++11 -DARMA_DONT_USE_WRAPPER -I/usr/include/python2.7 -I.. -I../../dlib -O3 -lblas -llapack ../api/python/api_wrap.cxx  ../core/utctime_utilities.cpp ../core/sceua_optimizer.cpp ../core/dream_optimizer.cpp ../api/api.cpp -lblas -llapack

