@echo off
@echo !
@echo ---------------------------------------------------------------------------------------------------
@echo !
@echo !  SHyFT MS Visual C++ build script
@echo !
@echo !   1. Before executing this script, start a command window, and start the ms vc mode compile like this
@echo !   cmd /k ""C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"" amd64
@echo !    then execute this script from the root directory of shyft
@echo !
@echo ! 2. Also note that you need to build boost 
@echo !
@echo !   in the ..\boost directory prior to invoking this command
@echo !   How to build boost for SHyFT is described on wiki: 
@echo !       https://github.com/statkraft/shyft/wiki/BuildCplusplus
@echo !   
@if exist D:\Anaconda\64 set PYTHONROOT=D:\Anaconda\64
@if exist C:\Anaconda set PYTHONROOT=C:\Anaconda
@if exist C:\Andaconda\64 set PYTHONROOT=C:\Anaconda\64
@echo ! Note: Anaconda 64 is found on this location: %PYTHONROOT%
@echo !
@echo Press enter to start build, or ctrl-C to stop
pause
pushd shyft

@echo Compiling sources
if not exist bin_c mkdir bin_c
if not exist bin_a mkdir bin_a

cl  /MP8 /Fobin_c\ /I".." /I"../.." /I"../../armadillo/include" /I"../../dlib" /I"../../boost" /I"%PYTHONROOT%\Include" /I"%PYTHONROOT%\lib\site-packages\numpy\core\include"  /D "ARMA_DONT_PRINT_ERRORS" /D "ARMA_USE_CXX11" /D "BOOSTSERIAL" /D "BOOST_THREAD_USE_DLL" /D "BOOST_LIB_DIAGNOSTIC=0" /D "BOOST_NUMERIC_ODEINT_CXX11" /D  "BOOST_ALL_DYN_LINK=1" /D "_WINDOWS" /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_WINDLL" /D "_MBCS" /D "_CRT_SECURE_NO_WARNINGS" /WX- /Zc:forScope /Gd /MD /EHsc /GS- /W1 /Gy /Zc:wchar_t /bigobj /Zi /Gm- /Ox /fp:precise /Fd"vc120.pdb"  /c   ../core/utctime_utilities.cpp  ../core/sceua_optimizer.cpp ../core/dream_optimizer.cpp ../api/timeseries.cpp 
cl  /MP8 /Fobin_a\ /I".." /I"../.." /I"../../armadillo/include" /I"../../dlib" /I"../../boost" /I"%PYTHONROOT%\Include" /I"%PYTHONROOT%\lib\site-packages\numpy\core\include"  /D "ARMA_DONT_PRINT_ERRORS" /D "ARMA_USE_CXX11" /D "BOOSTSERIAL" /D "BOOST_THREAD_USE_DLL" /D "BOOST_LIB_DIAGNOSTIC=0" /D "BOOST_NUMERIC_ODEINT_CXX11" /D  "BOOST_ALL_DYN_LINK=1" /D "_WINDOWS" /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_WINDLL" /D "_MBCS" /D "_CRT_SECURE_NO_WARNINGS" /WX- /Zc:forScope /Gd /MD /EHsc /GS- /W1 /Gy /Zc:wchar_t /bigobj /Zi /Gm- /Ox /fp:precise /Fd"vc120.pdb"  /c  ../api/boostpython/api.cpp ../api/boostpython/api_actual_evapotranspiration.cpp ../api/boostpython/api_cell_environment.cpp ../api/boostpython/api_gamma_snow.cpp ../api/boostpython/api_geo_cell_data.cpp ../api/boostpython/api_geo_point.cpp ../api/boostpython/api_hbv_snow.cpp ../api/boostpython/api_interpolation.cpp ../api/boostpython/api_kirchner.cpp ../api/boostpython/api_precipitation_correction.cpp ../api/boostpython/api_priestley_taylor.cpp ../api/boostpython/api_region_environment.cpp ../api/boostpython/api_skaugen.cpp ../api/boostpython/api_target_specification.cpp ../api/boostpython/api_time_axis.cpp ../api/boostpython/api_time_series.cpp ../api/boostpython/api_utctime.cpp ../api/boostpython/api_vectors.cpp ../api/boostpython/pt_gs_k.cpp ../api/boostpython/pt_hs_k.cpp ../api/boostpython/pt_ss_k.cpp 

@echo Linking api/_api.pyd
link /OUT:api\_api.pyd /libpath:..\..\shyft-data\blaslapack /libpath:..\..\boost\stage\lib /libpath:%PYTHONROOT%\libs /MANIFEST /NXCOMPAT /PDB:"_api.pdb" /DYNAMICBASE "blas_win64_MT.lib" "lapack_win64_MT.lib" "kernel32.lib" "user32.lib"  bin_c\utctime_utilities.obj  bin_c\sceua_optimizer.obj bin_c\dream_optimizer.obj bin_c\timeseries.obj bin_a\api.obj bin_a\api_vectors.obj bin_a\api_utctime.obj bin_a\api_actual_evapotranspiration.obj bin_a\api_cell_environment.obj bin_a\api_gamma_snow.obj bin_a\api_geo_cell_data.obj bin_a\api_geo_point.obj bin_a\api_hbv_snow.obj bin_a\api_interpolation.obj bin_a\api_kirchner.obj bin_a\api_precipitation_correction.obj bin_a\api_priestley_taylor.obj bin_a\api_region_environment.obj bin_a\api_skaugen.obj bin_a\api_target_specification.obj bin_a\api_time_axis.obj bin_a\api_time_series.obj /IMPLIB:"api\_api.lib" /DEBUG /DLL /MACHINE:X64 /OPT:REF /INCREMENTAL:NO /SUBSYSTEM:CONSOLE /MANIFESTUAC:"level='asInvoker' uiAccess='false'" /ManifestFile:"api\_api.pyd.intermediate.manifest" /OPT:ICF /ERRORREPORT:PROMPT /NOLOGO /TLBID:1 

@echo Linking pt_gs_k/_pt_gs_k.pyd
link /OUT:api\pt_gs_k\_pt_gs_k.pyd /libpath:..\..\shyft-data\blaslapack /libpath:..\..\boost\stage\lib /libpath:%PYTHONROOT%\libs /MANIFEST /NXCOMPAT /PDB:"api\pt_gs_k\_pt_gs_k.pdb" /DYNAMICBASE "blas_win64_MT.lib" "lapack_win64_MT.lib" "kernel32.lib" "user32.lib"  bin_a\pt_gs_k.obj  bin_c\timeseries.obj bin_c\utctime_utilities.obj bin_c\sceua_optimizer.obj bin_c\dream_optimizer.obj /IMPLIB:"api\pt_gs_k\_pt_gs_k.lib" /DEBUG /DLL /MACHINE:X64 /OPT:REF /INCREMENTAL:NO /SUBSYSTEM:CONSOLE /MANIFESTUAC:"level='asInvoker' uiAccess='false'" /ManifestFile:"api\pt_gs_k\_pt_gs_k.pyd.intermediate.manifest" /OPT:ICF /ERRORREPORT:PROMPT /NOLOGO /TLBID:1 

@echo Linking pt_ss_k/_pt_ss_k.pyd
link /OUT:api\pt_ss_k\_pt_ss_k.pyd /libpath:..\..\shyft-data\blaslapack /libpath:..\..\boost\stage\lib /libpath:%PYTHONROOT%\libs /MANIFEST /NXCOMPAT /PDB:"api\pt_ss_k\_pt_ss_k.pdb" /DYNAMICBASE "blas_win64_MT.lib" "lapack_win64_MT.lib" "kernel32.lib" "user32.lib"  bin_a\pt_ss_k.obj  bin_c\timeseries.obj bin_c\utctime_utilities.obj bin_c\sceua_optimizer.obj bin_c\dream_optimizer.obj /IMPLIB:"api\pt_ss_k\_pt_ss_k.lib" /DEBUG /DLL /MACHINE:X64 /OPT:REF /INCREMENTAL:NO /SUBSYSTEM:CONSOLE /MANIFESTUAC:"level='asInvoker' uiAccess='false'" /ManifestFile:"api\pt_ss_k\_pt_ss_k.pyd.intermediate.manifest" /OPT:ICF /ERRORREPORT:PROMPT /NOLOGO /TLBID:1 

@echo Linking pt_hs_k/_pt_hs_k.pyd
link /OUT:api\pt_hs_k\_pt_hs_k.pyd /libpath:..\..\shyft-data\blaslapack /libpath:..\..\boost\stage\lib /libpath:%PYTHONROOT%\libs /MANIFEST /NXCOMPAT /PDB:"api\pt_hs_k\_pt_hs_k.pdb" /DYNAMICBASE "blas_win64_MT.lib" "lapack_win64_MT.lib" "kernel32.lib" "user32.lib"  bin_a\pt_hs_k.obj  bin_c\timeseries.obj bin_c\utctime_utilities.obj bin_c\sceua_optimizer.obj bin_c\dream_optimizer.obj /IMPLIB:"api\pt_hs_k\_pt_hs_k.lib" /DEBUG /DLL /MACHINE:X64 /OPT:REF /INCREMENTAL:NO /SUBSYSTEM:CONSOLE /MANIFESTUAC:"level='asInvoker' uiAccess='false'" /ManifestFile:"api\pt_hs_k\_pt_hs_k.pyd.intermediate.manifest" /OPT:ICF /ERRORREPORT:PROMPT /NOLOGO /TLBID:1 

popd 
@echo Done (you can ignore all the Note: Use of this header(..) deprecated messages )
@echo !
@echo ! Notice that in order to have a fully working SHyFT api, you must have
@echo ! PATH pointing to 
@echo !    ..\boost\stage\lib that contains the boost runtime-dlls for boost
@echo !    ..\shyft-data\blaslapack that contains the blas_win64_MT.dll and lapack_win64_MT.dll 
@echo ! 
@echo ! Given this condtions, you can now verify the api running nosetest in shyft/shyft/tests/api directory
@echo !
