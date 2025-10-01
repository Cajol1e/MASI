# Install script for directory: D:/material/other/my_flexnn/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/material/other/my_flexnn/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/material/other/my_flexnn/out/build/x64-Debug/src/ncnnd.lib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "D:/material/other/my_flexnn/src/allocator.h"
    "D:/material/other/my_flexnn/src/benchmark.h"
    "D:/material/other/my_flexnn/src/blob.h"
    "D:/material/other/my_flexnn/src/c_api.h"
    "D:/material/other/my_flexnn/src/command.h"
    "D:/material/other/my_flexnn/src/cpu.h"
    "D:/material/other/my_flexnn/src/datareader.h"
    "D:/material/other/my_flexnn/src/gpu.h"
    "D:/material/other/my_flexnn/src/layer.h"
    "D:/material/other/my_flexnn/src/layer_shader_type.h"
    "D:/material/other/my_flexnn/src/layer_type.h"
    "D:/material/other/my_flexnn/src/mat.h"
    "D:/material/other/my_flexnn/src/modelbin.h"
    "D:/material/other/my_flexnn/src/net.h"
    "D:/material/other/my_flexnn/src/option.h"
    "D:/material/other/my_flexnn/src/paramdict.h"
    "D:/material/other/my_flexnn/src/pipeline.h"
    "D:/material/other/my_flexnn/src/pipelinecache.h"
    "D:/material/other/my_flexnn/src/simpleocv.h"
    "D:/material/other/my_flexnn/src/simpleomp.h"
    "D:/material/other/my_flexnn/src/simplestl.h"
    "D:/material/other/my_flexnn/src/vulkan_header_fix.h"
    "D:/material/other/my_flexnn/out/build/x64-Debug/src/ncnn_export.h"
    "D:/material/other/my_flexnn/out/build/x64-Debug/src/layer_shader_type_enum.h"
    "D:/material/other/my_flexnn/out/build/x64-Debug/src/layer_type_enum.h"
    "D:/material/other/my_flexnn/out/build/x64-Debug/src/platform.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "D:/material/other/my_flexnn/out/build/x64-Debug/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "D:/material/other/my_flexnn/out/build/x64-Debug/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "D:/material/other/my_flexnn/out/build/x64-Debug/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn-debug.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "D:/material/other/my_flexnn/out/build/x64-Debug/src/ncnnConfig.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "D:/material/other/my_flexnn/out/build/x64-Debug/src/ncnn.pc")
endif()

