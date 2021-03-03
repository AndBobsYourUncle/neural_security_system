# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.10)

project(NeuralSecuritySystem)

include(CheckCXXCompilerFlag)

#A function created to perform the same as in set(CMAKE_CXX...
function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
  if(flag_already_set EQUAL -1)
   check_cxx_compiler_flag("${flag}" flag_supported)
   if(flag_supported)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
   endif()
   unset(flag_supported CACHE)
  endif()
endfunction()

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
        set(ARCH intel64)
    elseif(ARCH STREQUAL "i386")
        set(ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

foreach(artifact IN ITEMS ARCHIVE COMPILE_PDB LIBRARY PDB RUNTIME)
    set("CMAKE_${artifact}_OUTPUT_DIRECTORY" "${CMAKE_CURRENT_BINARY_DIR}/${BIN_FOLDER}/$<CONFIG>")
endforeach()

if(UNIX)
    string(APPEND CMAKE_LIBRARY_OUTPUT_DIRECTORY "/lib")
    string(APPEND CMAKE_ARCHIVE_OUTPUT_DIRECTORY "/lib")
endif()

if (WIN32)
    if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings
    endif()
elseif(UNIX)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
    if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
    endif()
endif()

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
####################################

set (GFLAGS_IS_SUBPROJECT TRUE)
set (HAVE_SYS_STAT_H 1)
set (HAVE_INTTYPES_H 1)

add_subdirectory(${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/demos/thirdparty/gflags libs/thirdparty/gflags)

# add_subdirectory(/home/nicholas/paho.mqtt.cpp libs/mqtt)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

add_subdirectory(${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/demos/common libs/common)

include(CMakeParseArguments)

#
# is_add_project(NAME <target name>
#               SOURCES <source files>
#               [HEADERS <header files>]
#               [INCLUDE_DIRECTORIES <include dir>]
#               [DEPENDENCIES <dependencies>]
#               [OPENCV_DENENDENCIES <opencv modules>]
#               [EXCLUDE_CPPLINT]
#
macro(is_add_project)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES OPENCV_DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_PROJECT "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # Find OpenCV components if exist
    if(IE_PROJECT_OPENCV_DEPENDENCIES)
        find_package(OpenCV COMPONENTS ${IE_PROJECT_OPENCV_DEPENDENCIES} QUIET)
        if(NOT OpenCV_FOUND)
            message(WARNING "OPENCV is disabled or not found, " ${IE_PROJECT_NAME} " skipped")
            return()
        else()
            add_definitions(-DUSE_OPENCV)
        endif()
    endif()

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${IE_PROJECT_SOURCES})
    if(IE_PROJECT_HEADERS)
        source_group("include" FILES ${IE_PROJECT_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${IE_PROJECT_NAME} ${IE_PROJECT_SOURCES} ${IE_PROJECT_HEADERS})

    if(WIN32)
        set_target_properties(${IE_PROJECT_NAME} PROPERTIES COMPILE_PDB_NAME ${IE_PROJECT_NAME})
    endif()

    if(IE_PROJECT_INCLUDE_DIRECTORIES)
        target_include_directories(${IE_PROJECT_NAME} PRIVATE ${IE_PROJECT_INCLUDE_DIRECTORIES})
    endif()

    target_link_libraries(${IE_PROJECT_NAME} PRIVATE ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES}
                                                    ${IE_PROJECT_DEPENDENCIES} common gflags)

    if(UNIX)
        target_link_libraries(${IE_PROJECT_NAME} PRIVATE pthread)
    endif()

    # target_link_libraries(${IE_PROJECT_NAME} PRIVATE paho-mqtt3a)
    # target_link_libraries(${IE_PROJECT_NAME} PRIVATE paho-mqttpp3)
    # # target_link_libraries(${IE_PROJECT_NAME} PRIVATE ${PahoMqttC_LIBRARIES})
    # find_package(PahoMqttCpp REQUIRED)

    target_link_libraries(${IE_PROJECT_NAME} PRIVATE PahoMqttCpp::paho-mqttpp3)

    # create global target with all projects
    if(NOT TARGET ie_projects)
        add_custom_target(ie_projects ALL)
    endif()
    add_dependencies(ie_projects ${IE_PROJECT_NAME})

    if(COMMAND add_cpplint_target AND NOT IE_PROJECT_EXCLUDE_CPPLINT)
        add_cpplint_target(${IE_PROJECT_NAME}_cpplint FOR_TARGETS ${IE_PROJECT_NAME})
    endif()
endmacro()


# use this flag if you need to throw custom message in case if the IE package is not found.
if (IE_NOT_FOUND_MESSAGE)
    find_package(InferenceEngine 2.0 QUIET)
    if (NOT(InferenceEngine_FOUND))
        message(FATAL_ERROR ${IE_NOT_FOUND_MESSAGE})
    endif()
else()
    find_package(InferenceEngine 2.0 REQUIRED)
endif()

find_package(ngraph REQUIRED)

# enable_cxx_compiler_flag_if_supported("-O2")
# enable_cxx_compiler_flag_if_supported("D_NDEBUG")

find_package(OpenSSL REQUIRED)
find_package(PahoMqttCpp REQUIRED)

# find_package(PahoMqttCpp REQUIRED)

# find_library(paho-mqttpp3 NAMES libpaho-mqttpp3.a REQUIRED)
# find_library(paho-mqtt3a NAMES libpaho-mqtt3a.so REQUIRED)

# add_library(pahottpp STATIC IMPORTED)
# set_target_properties(pahottpp PROPERTIES IMPORTED_LOCATION ${paho-mqttpp3})

# add_library(pahomqtt3a SHARED IMPORTED)
# set_property(TARGET pahomqtt3a PROPERTY IMPORTED_LOCATION ${paho-mqtt3a})

FILE(GLOB SRC_FILES ./*.cpp)
FILE(GLOB H_FILES ./*.h)

is_add_project(NAME neural_security_system
              SOURCES ${SRC_FILES}
              HEADERS ${H_FILES}
              DEPENDENCIES monitors models pipelines
              OPENCV_DEPENDENCIES highgui videoio imgproc)
