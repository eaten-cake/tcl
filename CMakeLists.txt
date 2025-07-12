cmake_minimum_required(VERSION 3.13)
project(extend_tvm C CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if(DEFINED ENV{TVM_PREBUILD_PATH})
  set(TVM_PREBUILD_PATH $ENV{TVM_PREBUILD_PATH})
  message(STATUS "Using TVM_PREBUILD_PATH from environment: ${TVM_PREBUILD_PATH}")
endif()
# set tvm prebuild path
# if TVM_PREBUILD_PATH is not defined, use the default path
if(DEFINED TVM_PREBUILD_PATH)
  message(STATUS "Using prebuilt TVM at ${TVM_PREBUILD_PATH}")
  # add library path
  add_library(tvm SHARED IMPORTED)
  set_target_properties(tvm PROPERTIES
      IMPORTED_LOCATION ${TVM_PREBUILD_PATH}/libtvm.so
      INTERFACE_INCLUDE_DIRECTORIES ${TVM_PREBUILD_PATH}/../include
  )
  add_library(tvm_runtime SHARED IMPORTED)
  set_target_properties(tvm_runtime PROPERTIES
      IMPORTED_LOCATION ${TVM_PREBUILD_PATH}/libtvm_runtime.so
      INTERFACE_INCLUDE_DIRECTORIES ${TVM_PREBUILD_PATH}/../include
  )
else()
  message(FATAL_ERROR "TVM_PREBUILD_PATH is not defined. Please set it to the path of your prebuilt TVM.")
endif()


# Set default build type to RelWithDebInfo if not provided
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  message(STATUS "Setting default build type to ${CMAKE_BUILD_TYPE}")
endif()


set(TVM_INCLUDE 
  ${TVM_PREBUILD_PATH}/../include
  ${TVM_PREBUILD_PATH}/../ffi/include
  ${TVM_PREBUILD_PATH}/../3rdparty/dlpack/include
  ${TVM_PREBUILD_PATH}/../3rdparty/dmlc-core/include
)


file(GLOB_RECURSE TCL_SRCS
  src/testing/*.cc
)
add_library(tcl_objs OBJECT ${TCL_SRCS})

target_include_directories(tcl_objs PRIVATE ${TVM_INCLUDE})
target_include_directories(tcl_objs PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Define compile-time macros
set(TCL_COMPILE_DEFS
  DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
  __STDC_FORMAT_MACROS=1
  PICOJSON_USE_INT64
)
target_compile_definitions(tcl_objs PRIVATE ${TCL_COMPILE_DEFS})

add_library(tcl SHARED $<TARGET_OBJECTS:tcl_objs>)
add_library(tcl_static STATIC $<TARGET_OBJECTS:tcl_objs>)
add_dependencies(tcl_static tvm_runtime)
set_target_properties(tcl_static PROPERTIES OUTPUT_NAME tcl)

target_link_libraries(tcl PUBLIC tvm_runtime)

add_library(tcl_module SHARED $<TARGET_OBJECTS:tcl_objs>)
target_link_libraries(tcl_module PUBLIC tvm)


# Install targets
if(TCL_INSTALL_STATIC_LIB)
  install(TARGETS tcl_static tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
  ) 
else()
  if(DEFINED TVM_PREBUILD_PATH)
    install(TARGETS tcl tcl_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  else()
    install(TARGETS tvm_runtime tcl tcl_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  endif()
endif()








