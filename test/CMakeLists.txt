# CMake configuration for tests

if(WIN32 AND MSVC)
  set(MAKE_COMMAND "nmake")
  set(MAKE_FLAGS "/F")
  set(MAKE_FILE "NMakeFile.testupdate")
else()
  set(MAKE_COMMAND "make")
  set(MAKE_FLAGS -f)
  set(MAKE_FILE "Makefile.testupdate")
endif()

# Generate the main.cpp file
message(STATUS "Generating main test unit for C++ test suite...")
set(CXXTESTGEN "TestGenerator=${SHYFT_DEPENDENCIES}/cxxtest/bin/cxxtestgen")
execute_process(COMMAND ${MAKE_COMMAND} ${MAKE_FLAGS} ${MAKE_FILE} ${CXXTESTGEN}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test
  RESULT_VARIABLE rv
  OUTPUT_VARIABLE make_output
  ERROR_VARIABLE  make_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

if(NOT rv EQUAL 0)
  message("test make output:" ${make_output})
  message("test make error:" ${make_error})
  message(FATAL_ERROR "Errors occurred.  Leaving now!")
  return()
endif()

# Sources
file(GLOB cpps *.cpp)
list(REMOVE_ITEM cpps "test_pch.cpp")
file(GLOB headers *.h)
set(sources ${cpps} ${headers})
set(target "test_shyft")

# Flags
add_definitions("-D__UNIT_TEST__ -DVERBOSE=0 -DCXXTEST_HAVE_EH -DCXXTEST_HAVE_STD")

add_executable(${target} ${sources})
# Additional libraries
target_link_libraries(${target} shyftcore shyftapi blas lapack boost_filesystem boost_system)

# This can be made more specific, but we would need the list of tests.
#add_test(${target} ${target})
#The current list:
add_test(inverse_distance_test ${target} inverse_distance_test)
add_test(time_axis_test ${target} time_axis_test)
add_test(region_model_test ${target} region_model_test )
add_test(priestley_taylor_test ${target} priestley_taylor_test)
add_test(gamma_snow_test ${target} gamma_snow_test)
add_test(kirchner_test ${target} kirchner_test)
add_test(bayesian_kriging_test ${target} bayesian_kriging_test)
add_test(utctime_utilities_test ${target} utctime_utilities_test)
add_test(pt_gs_k_test ${target} pt_gs_k_test)
add_test(actual_evapotranspiration_test ${target} actual_evapotranspiration_test)
add_test(calibration_test ${target} calibration_test)
add_test(hbv_snow_test  ${target} hbv_snow_test)
add_test(pt_hs_k_test ${target} pt_hs_k_test)
add_test(pt_ss_k_test ${target} pt_ss_k_test)
add_test(timeseries_test ${target} timeseries_test)
add_test(api_test ${target} api_test)
add_test(cell_builder_test ${target} cell_builder_test) 
add_test(skaugen_test ${target} skaugen_test)
add_test(sceua_test ${target} sceua_test)
add_test(gridpp_test ${target} gridpp_test)
add_test(kalman_test ${target} kalman_test)


