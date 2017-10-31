################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
# Author: Kent Knox

#if( FFTW_FIND_VERSION VERSION_LESS "3" )
#    message( FFTW_FIND_VERION is ${FFTW_FIND_VERSION})
#    message( FATAL_ERROR "FindFFTW can not configure versions less than FFTW 3.0.0" )
#endif( )

find_path(FFTW_INCLUDE_DIRS
    NAMES fftw3.h
    HINTS
        ${FFTW_ROOT}/include
        $ENV{FFTW_ROOT}/include
    PATHS
        /usr/include
        /usr/local/include
)
mark_as_advanced( FFTW_INCLUDE_DIRS )

# message( STATUS "FFTW_FIND_COMPONENTS: ${FFTW_FIND_COMPONENTS}" )
# message( STATUS "FFTW_FIND_REQUIRED_FLOAT: ${FFTW_FIND_REQUIRED_FLOAT}" )
# message( STATUS "FFTW_FIND_REQUIRED_DOUBLE: ${FFTW_FIND_REQUIRED_DOUBLE}" )

set( FFTW_LIBRARIES "" )
if( FFTW_FIND_REQUIRED_FLOAT OR FFTW_FIND_REQUIRED_SINGLE )
  find_library( FFTW_LIBRARIES_SINGLE
      NAMES fftw3f fftw3f-3 fftw3 fftw3-3
      HINTS
          ${FFTW_ROOT}/lib
          $ENV{FFTW_ROOT}/lib
      PATHS
          /usr/lib
          /usr/local/lib
      PATH_SUFFIXES
          x86_64-linux-gnu
      DOC "FFTW dynamic library single"
  )
  mark_as_advanced( FFTW_LIBRARIES_SINGLE )
  list( APPEND FFTW_LIBRARIES ${FFTW_LIBRARIES_SINGLE} )
endif( )

if( FFTW_FIND_REQUIRED_DOUBLE )
  find_library( FFTW_LIBRARIES_DOUBLE
      NAMES fftw3
      HINTS
          ${FFTW_ROOT}/lib
          $ENV{FFTW_ROOT}/lib
      PATHS
          /usr/lib
          /usr/local/lib
      PATH_SUFFIXES
          x86_64-linux-gnu
      DOC "FFTW dynamic library double"
  )
  mark_as_advanced( FFTW_LIBRARIES_DOUBLE )
  list( APPEND FFTW_LIBRARIES ${FFTW_LIBRARIES_DOUBLE} )
endif( )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( FFTW
    REQUIRED_VARS FFTW_INCLUDE_DIRS FFTW_LIBRARIES )

if( NOT FFTW_FOUND )
    message( STATUS "FindFFTW could not find all of the following fftw libraries" )
    message( STATUS "${FFTW_FIND_COMPONENTS}" )
else( )
    message(STATUS "FindFFTW configured variables:" )
    message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}" )
    message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}" )
endif()
