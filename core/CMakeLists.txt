# CMake file for compiling the C++ core library
# library sources
set(SOURCES   dtss_client.cpp dtss.cpp experimental.cpp dream_optimizer.cpp sceua_optimizer.cpp 
    time_series_dd.cpp core_serialization.cpp time_series_serialization.cpp
    utctime_utilities.cpp expression_serialization.cpp
)
message(STATUS "Generating Makefile for the SHyFT static library...")
add_library(shyftcore STATIC ${SOURCES} )
set_target_properties(shyftcore PROPERTIES OUTPUT_NAME shyftcore)
if (MSVC)
  set_target_properties(shyftcore PROPERTIES PREFIX lib)
endif()
target_link_libraries(shyftcore ${LIBS})
