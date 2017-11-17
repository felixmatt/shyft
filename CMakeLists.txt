# CMake build system for Shyft
# ============================
#
# This requires shyft_dependencies installed as well as a C++11/C++14 compliant compiler.
#
# Available options:
#
#   BUILD_TESTING: default ON
#       build test programs and generates the "test" target
#   BUILD_PYTHON_EXTENSIONS: default ON
#       build Python extensions for Shyft
#
# The next environment variables are honored:
#
#   SHYFT_DEPENDENCIES_DIR: default ${PROJECT_SOURCE_DIR}/shyft-dependencies
#


cmake_minimum_required(VERSION 3.2.0)

project(shyft)

# Get the full version for Shyft
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION SHYFT_VERSION_STRING)

message("Configuring for Shyft version: " ${SHYFT_VERSION_STRING})

#if ccache, use it, but require cmake 3.4.0..
#find_program(CCACHE_FOUND ccache)
#if(CCACHE_FOUND)
#    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
#    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
#endif(CCACHE_FOUND)

# options
option(BUILD_TESTING "Build test programs for SHYFT C++ core library" ON)
option(BUILD_PYTHON_EXTENSIONS "Build Python extensions for SHYFT" ON)
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
  set(SHYFT_DEPENDENCIES_DIR "${PROJECT_SOURCE_DIR}/../shyft_dependencies")
endif()
# Our code requires an absolute directory for the dependencies
get_filename_component(SHYFT_DEPENDENCIES ${SHYFT_DEPENDENCIES_DIR} ABSOLUTE)
message("SHYFT_DEPENDENCIES directory: " ${SHYFT_DEPENDENCIES})
message("You can override the above via the $SHYFT_DEPENDENCIES_DIR environment variable.")

# The flags for compile the beast
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -std=c++1y -fexceptions -pthread -Winvalid-pch -L${SHYFT_DEPENDENCIES}/lib" CACHE STRING "CXX flags." FORCE)

# add defines that need to be consistent across sub-projects
add_definitions("-DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11 -DARMA_NO_DEBUG ")

# The directories to be included
include_directories(${CMAKE_SOURCE_DIR} ${SHYFT_DEPENDENCIES}/include)

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
