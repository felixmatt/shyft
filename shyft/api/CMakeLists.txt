# Compilation step for Python extensions
#-----------------------------------------------------------------------------
# Python install
find_package(PythonInterp REQUIRED)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c" "from distutils import sysconfig; print(sysconfig.get_python_lib())"
  RESULT_VARIABLE rv
  OUTPUT_VARIABLE lib_python_output
  ERROR_VARIABLE  lib_python_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(rv)
  message("Python lib output:" ${lib_python_output})
  message("Python lib error:" ${lib_python_error})
  message(FATAL_ERROR "Errors occurred.  Leaving now!")
  return()
else()
  string(LENGTH ${lib_python_output} len)
  math(EXPR pathend "${len} - 14")  # remove "/site-packages" at the end
  string(SUBSTRING ${lib_python_output} 0 ${pathend} lib_python)
endif()
message("lib_python: " ${lib_python})

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c" "from distutils import sysconfig; print(sysconfig.get_python_inc())"
  RESULT_VARIABLE rv
  OUTPUT_VARIABLE include_python
  ERROR_VARIABLE  include_python_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

if (rv)
  message("Python include:" ${include_python})
  message("Python import error:" ${include_python_error})
  message(FATAL_ERROR "Errors occurred.  Leaving now!")
  return()
endif()
message("include_python: " ${include_python})

execute_process(
 COMMAND ${PYTHON_EXECUTABLE} "-c" "import numpy as np; print(np.get_include())"
 RESULT_VARIABLE rv
 OUTPUT_VARIABLE include_numpy
 ERROR_VARIABLE  include_numpy_error
 OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (rv)
  message("NumPy include:" ${include_numpy})
  message("NumPy import error:" ${include_numpy_error})
  message(FATAL_ERROR "Errors occurred.  Leaving now!")
  return()
endif()
message("include_numpy: " ${include_numpy})

# Include files in core directory
include_directories(${PROJECT_SOURCE_DIR} ${SHYFT_DEPENDENCIES}/dlib
  ${CMAKE_SOURCE_DIR} ${include_python} ${SHYFT_DEPENDENCIES}/boost
  ${SHYFT_DEPENDENCIES}/armadillo/include ${include_numpy})

# Set the compiler flags (for gcc and clang)
set(C_FLAGS -O3 -shared -L${lib_python} -pthread -s -fPIC -std=c++1y -DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11)
#set(CMAKE_CXX_FLAGS ${C_FLAGS} CACHE STRING "CXX flags." FORCE)

# Set the common C++ files
set(CXX_FILES ${CMAKE_SOURCE_DIR}/core/utctime_utilities.cpp ${CMAKE_SOURCE_DIR}/core/sceua_optimizer.cpp ${CMAKE_SOURCE_DIR}/core/dream_optimizer.cpp ${CMAKE_SOURCE_DIR}/api/api.cpp)

# Create the API destination directory
file(MAKE_DIRECTORY ${SHYFT_DEPENDENCIES}/api)

# Additional libraries
set(LIBS "-lblas -llapack -lboost_python3")

# Compile extensions api extension
foreach(python_api_extension  "api" )
   file(GLOB cpps ${CMAKE_SOURCE_DIR}/api/boostpython/${python_api_extension}*.cpp )
  add_library(${python_api_extension} SHARED ${CXX_FILES} ${cpps} )
  set_target_properties(${python_api_extension} PROPERTIES OUTPUT_NAME ${python_api_extension})
  # Python extensions do not use the 'lib' prefix
  set_target_properties(${python_api_extension} PROPERTIES PREFIX "_")
  set_property(TARGET ${python_api_extension} APPEND PROPERTY COMPILE_DEFINITIONS SHYFT_EXTENSION)
  target_link_libraries(${python_api_extension} ${LIBS})
  install(TARGETS ${python_api_extension} DESTINATION ${CMAKE_SOURCE_DIR}/shyft/api)

endforeach(python_api_extension)

# then for each sub module
foreach(python_extension  "pt_gs_k" "pt_ss_k" "pt_hs_k")
  add_library(${python_extension} SHARED ${CXX_FILES} ${CMAKE_SOURCE_DIR}/api/boostpython/${python_extension}.cpp)
  set_target_properties(${python_extension} PROPERTIES OUTPUT_NAME ${python_extension})
  # Python extensions do not use the 'lib' prefix
  set_target_properties(${python_extension} PROPERTIES PREFIX "_")
  set_property(TARGET ${python_extension} APPEND PROPERTY COMPILE_DEFINITIONS SHYFT_EXTENSION)
  target_link_libraries(${python_extension} ${LIBS})
  install(TARGETS ${python_extension} DESTINATION ${CMAKE_SOURCE_DIR}/shyft/api/${python_extension})

endforeach(python_extension)
