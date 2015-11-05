# CMake configuration for tests
# Sources
file(GLOB cpps *.cpp)
list(REMOVE_ITEM cpps "test_pch.cpp")
file(GLOB headers *.h)
set(sources ${cpps} ${headers})
set(target "test_shyft")

if(WIN32 AND MSVC)
  set(MAKE_COMMAND "nmake")
  set(MAKE_FLAGS "/F")
  set(MAKE_FILE "NMakeFile.testupdate")
else()
  set(MAKE_COMMAND "make")
  set(MAKE_FLAGS -j 4 -f)
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

# Flags
#set(C_FLAGS -Wall -pthread -std=c++1y -fexceptions -Winvalid-pch)
add_definitions("-DARMA_DONT_NO_DEBUG -DARMA_DONT_USE_WRAPPER -D__UNIT_TEST__ -DVERBOSE=0 -DCXXTEST_HAVE_EH -DCXXTEST_HAVE_STD")
include_directories(${CMAKE_SOURCE_DIR} ${SHYFT_DEPENDENCIES}/cxxtest ${SHYFT_DEPENDENCIES}/dlib)

add_executable(${target} ${sources})
# Additional libraries
target_link_libraries(${target} shyftcore blas lapack boost_filesystem boost_system)

# This can be made more specific, but we would need the list of tests.
add_test(${target} ${target})
