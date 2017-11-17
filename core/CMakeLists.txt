# CMake file for compiling the C++ core library
# library sources
FILE(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
FILE(GLOB cpps RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM cpps "core_pch.cpp")
set(SOURCES ${headers} ${cpps})
message(STATUS "Generating Makefile for the SHyFT static library...")
add_library(shyftcore STATIC ${SOURCES} )
set_target_properties(shyftcore PROPERTIES OUTPUT_NAME shyftcore)
if (MSVC)
  set_target_properties(shyftcore PROPERTIES PREFIX lib)
endif()
target_link_libraries(shyftcore ${LIBS})
