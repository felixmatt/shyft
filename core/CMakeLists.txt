# CMake file for compiling the C++ core library

# library sources
FILE(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
FILE(GLOB cpps RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM cpps "core_pch.cpp")
set(SOURCES ${headers} ${cpps})

message(STATUS "Compiling boost library (only selected components)...")
execute_process(COMMAND ./bootstrap.sh --prefix=${SHYFT_DEPENDENCIES}/local --with-libraries=filesystem,system
  WORKING_DIRECTORY ${SHYFT_DEPENDENCIES}/boost
  RESULT_VARIABLE rv
  OUTPUT_VARIABLE output
  ERROR_VARIABLE  error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
if(NOT rv EQUAL 0)
  message("boost bootstrap output:" ${output})
  message("boost bootstart error:" ${error})
  message(FATAL_ERROR "Errors occurred.  Leaving now!")
  return()
endif()

execute_process(COMMAND ./b2 install
  WORKING_DIRECTORY ${SHYFT_DEPENDENCIES}/boost
  RESULT_VARIABLE rv
  OUTPUT_VARIABLE output
  ERROR_VARIABLE  error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
if(NOT rv EQUAL 0)
  message("boost b2 output:" ${output})
  message("boost b2 error:" ${error})
  message(FATAL_ERROR "Errors occurred.  Leaving now!")
  return()
endif()


message(STATUS "Generating Makefile for the SHyFT static library...")

set(CMAKE_CXX_FLAGS "-Wall -std=c++1y -fexceptions -pthread -Winvalid-pch -I${SHYFT_DEPENDENCIES}/armadillo/include -I${SHYFT_DEPENDENCIES}/local -L${SHYFT_DEPENDENCIES}/local/lib" CACHE STRING "CXX flags." FORCE)
add_definitions("-DARMA_DONT_USE_WRAPPER")

add_library(shyftcore STATIC ${SOURCES})
set_target_properties(shyftcore PROPERTIES OUTPUT_NAME shyftcore)
if (MSVC)
  set_target_properties(shyftcore PROPERTIES PREFIX lib)
endif()
target_link_libraries(shyftcore ${LIBS})
