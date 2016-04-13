# CMake file for compiling the C++ api library

# library sources
FILE(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
FILE(GLOB cpps RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(SOURCES ${headers} ${cpps})

message(STATUS "Generating Makefile for the SHyFT api static library...")

add_library(shyftapi STATIC ${SOURCES})
set_target_properties(shyftapi PROPERTIES OUTPUT_NAME shyftapi)
if (MSVC)
  set_target_properties(shyftapi PROPERTIES PREFIX lib)
endif()
target_link_libraries(shyftapi ${LIBS})
