# CMake build system for SHyFT
# ============================
#
# This requires SWIG and NumPy installed as well as a C++11 compliant compiler.
#
# Available options:
#
#   BUILD_TESTING: default ON
#       build test programs and generates the "test" target
#   BUILD_PYTHON_EXTENSIONS: default ON
#       build Python extensions for SHYFT
#
# The next environment variables are honored:
#
#   SHYFT_DEPENDENCIES_DIR: default ${PROJECT_SOURCE_DIR}/shyft-dependencies
#


cmake_minimum_required(VERSION 2.8.7)
project(shyft)

# Get the full version for SHyFT
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION SHYFT_VERSION_STRING)

message("Configuring for SHyFT version: " ${SHYFT_VERSION_STRING})

# options
option(BUILD_TESTING
  "Build test programs for SHYFT C++ core library" ON)
option(BUILD_PYTHON_EXTENSIONS
  "Build Python extensions for SHYFT" ON)

set(SHYFT_DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified. Defaulting to '${SHYFT_DEFAULT_BUILD_TYPE}'.")
    set(CMAKE_BUILD_TYPE ${SHYFT_DEFAULT_BUILD_TYPE} CACHE STRING
        "Choose the type of build." FORCE)

    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# The dependencies directory
if(DEFINED ENV{SHYFT_DEPENDENCIES_DIR})
  set(SHYFT_DEPENDENCIES_DIR $ENV{SHYFT_DEPENDENCIES_DIR})
else()
  set(SHYFT_DEPENDENCIES_DIR "${PROJECT_SOURCE_DIR}/shyft-dependencies")
endif()
# Our code requires an absolute directory for the dependencies
get_filename_component(SHYFT_DEPENDENCIES ${SHYFT_DEPENDENCIES_DIR} ABSOLUTE)
message("SHYFT_DEPENDENCIES directory: " ${SHYFT_DEPENDENCIES})
message("You can override the above via the $SHYFT_DEPENDENCIES_DIR environment variable.")
# Create the dependencies directory
file(MAKE_DIRECTORY ${SHYFT_DEPENDENCIES}/api)

# The flags for compile the beast
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -std=c++1y -fexceptions -pthread -Winvalid-pch -L${SHYFT_DEPENDENCIES}/local/lib" CACHE STRING "CXX flags." FORCE)

# add defines that need to be consistent across sub-projects
add_definitions("-DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11 -DARMA_NO_DEBUG")

# The directories to be included
include_directories(${CMAKE_SOURCE_DIR} ${SHYFT_DEPENDENCIES}/cxxtest ${SHYFT_DEPENDENCIES}/dlib ${SHYFT_DEPENDENCIES}/armadillo/include ${SHYFT_DEPENDENCIES}/local/include)


# The versions for dependencies
set(DLIB_PREFIX "dlib")
set(DLIB_VERSION "18.18")
set(DLIB_TARBALL ${DLIB_PREFIX}-${DLIB_VERSION}.tar.bz2)
set(DLIB_URL http://sourceforge.net/projects/dclib/files/dlib/v${DLIB_VERSION}/${DLIB_TARBALL}/download)

set(ARMADILLO_PREFIX "armadillo")
set(ARMADILLO_VERSION "7.300.0")
set(ARMADILLO_TARBALL ${ARMADILLO_PREFIX}-${ARMADILLO_VERSION}.tar.gz)
set(ARMADILLO_URL http://sourceforge.net/projects/arma/files/${ARMADILLO_TARBALL})

set(BOOST_PREFIX "boost")
set(BOOST_VERSION "1.61.0")
string(REPLACE "." "_" BOOST__VERSION ${BOOST_VERSION})
set(BOOST_TARBALL ${BOOST_PREFIX}_${BOOST__VERSION}.tar.gz)
set(BOOST_URL http://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/${BOOST_TARBALL}/download)

set(CXXTEST_PREFIX "cxxtest")
set(CXXTEST_VERSION "4.4")
set(CXXTEST_TARBALL ${CXXTEST_PREFIX}-${CXXTEST_VERSION}.tar.gz)
set(CXXTEST_URL http://sourceforge.net/projects/cxxtest/files/cxxtest/${CXXTEST_VERSION}/${CXXTEST_TARBALL}/download)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include("${CMAKE_SOURCE_DIR}/cmake/DownloadAndExtractTarball.cmake")

message(STATUS "Downloading dependencies.  This may take a while...")

# dlib
DownloadAndExtractTarball(
  ${DLIB_PREFIX}
  ${DLIB_VERSION}
  ${DLIB_TARBALL}
  ${DLIB_URL}
  ${SHYFT_DEPENDENCIES})

# armadillo
DownloadAndExtractTarball(
  ${ARMADILLO_PREFIX}
  ${ARMADILLO_VERSION}
  ${ARMADILLO_TARBALL}
  ${ARMADILLO_URL}
  ${SHYFT_DEPENDENCIES})

# boost
DownloadAndExtractTarball(
  ${BOOST_PREFIX}
  ${BOOST_VERSION}
  ${BOOST_TARBALL}
  ${BOOST_URL}
  ${SHYFT_DEPENDENCIES})

# cxxtest
DownloadAndExtractTarball(
  ${CXXTEST_PREFIX}
  ${CXXTEST_VERSION}
  ${CXXTEST_TARBALL}
  ${CXXTEST_URL}
  ${SHYFT_DEPENDENCIES})


# Once we have the dependencies solved, let's go with cmake subdirectories

# C++ core and tests
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(core)
  add_subdirectory(api)
  add_subdirectory(test)
endif(BUILD_TESTING)

# Python extensions
if(BUILD_PYTHON_EXTENSIONS)
  add_subdirectory(shyft/api)
endif(BUILD_PYTHON_EXTENSIONS)
