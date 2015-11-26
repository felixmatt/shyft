# This is meant for SHyFT, although it would be worth to see if we can
# use this:
# https://github.com/neovim/neovim/blob/master/third-party/cmake/DownloadAndExtractFile.cmake

function(DownloadAndExtractTarball PREFIX VERSION TARBALL URL DOWNLOAD_DIR)
  file(MAKE_DIRECTORY ${DOWNLOAD_DIR})

  set(file ${DOWNLOAD_DIR}/${TARBALL})

  if(TIMEOUT)
    set(timeout_args TIMEOUT ${timeout})
    set(timeout_msg "${timeout} seconds")
  else()
    set(timeout_args "# no TIMEOUT")
    set(timeout_msg "none")
  endif()

  # Download the tarball
  if(NOT EXISTS "${file}")
    message(STATUS "downloading...
     src='${URL}'
     dst='${file}'
     timeout='${timeout_msg}'")
    file(DOWNLOAD
      ${URL}
      ${file}
      ${hash_args}
      STATUS status
      LOG log)
    list(GET status 0 status_code)
    list(GET status 1 status_string)
    if(NOT status_code EQUAL 0)
      message(FATAL_ERROR "error: downloading '${URL}' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
    else()
      message(STATUS "${TARBALL} downloaded successfully [OK]")
    endif()
    set(NEW_DOWNLOAD TRUE)
  else(NOT EXISTS "${file}")
    message(STATUS "${TARBALL} already downloaded [OK]")
    set(NEW_DOWNLOAD FALSE)
  endif(NOT EXISTS "${file}")

  set(SRC_DIR ${DOWNLOAD_DIR}/${PREFIX})

  # If the source dir exists and is not a new download, skip the extraction
  if(EXISTS "${SRC_DIR}" AND NOT NEW_DOWNLOAD)
    RETURN()
  endif()

  # Proceed with the extraction
  message(STATUS "downloading... done")
  message(STATUS "extracting...
     src='${TARBALL}'
     dst='${SRC_DIR}'")

  if(NOT EXISTS "${file}")
    message(FATAL_ERROR "error: file to extract does not exist: '${file}'")
  endif()

  # Prepare a space for extracting
  set(i 1234)
  while(EXISTS "${DOWNLOAD_DIR}/ex-${PREFIX}${i}")
    math(EXPR i "${i} + 1")
  endwhile()
  set(ut_dir "${DOWNLOAD_DIR}/ex-${PREFIX}${i}")
  file(MAKE_DIRECTORY "${ut_dir}")

  # Extract it
  message(STATUS "extracting... [tar xfz]")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz ${file}
    WORKING_DIRECTORY ${ut_dir}
    RESULT_VARIABLE rv)

  if(NOT rv EQUAL 0)
    message(STATUS "extracting... [error clean up]")
    file(REMOVE_RECURSE "${ut_dir}")
    message(FATAL_ERROR "error: extract of '${file}' failed")
  endif()

  # Analyze what came out of the tar file
  message(STATUS "extracting... [analysis]")
  file(GLOB contents "${ut_dir}/*")
  list(LENGTH contents n)
  if(NOT n EQUAL 1 OR NOT IS_DIRECTORY "${contents}")
    set(contents "${ut_dir}")
  endif()

  # Move "the one" directory to the final directory
  message(STATUS "extracting... [rename]")
  file(REMOVE_RECURSE ${SRC_DIR})
  get_filename_component(contents ${contents} ABSOLUTE)
  file(RENAME ${contents} ${SRC_DIR})

  # Clean up
  message(STATUS "extracting... [clean up]")
  file(REMOVE_RECURSE "${ut_dir}")

  message(STATUS "extracting... done")

endfunction()
