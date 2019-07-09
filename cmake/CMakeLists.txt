cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (PROJ aicv_4th)
set (PROJ_DIR ../)
# set (BUILD_DIR ${PROJ_DIR}/build)
set (BUILD_DIR build)
set (SRC_DIR src)
set (VENDOR_DIR ${SRC_DIR}/vendor)

project(${PROJ} VERSION 0.0.1)

# build dir
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR})

# vendor
include_directories(${VENDOR_DIR})

# test
enable_testing()
set (SRC_FILES ${SRC_DIR}/test_main.cpp)
add_library(test_main OBJECT ${SRC_FILES})


# week2
set (WEEK2_DIR ${PROJ_DIR}/week2)
include_directories(${WEEK2_DIR})
aux_source_directory(${WEEK2_DIR} WEEK2_FILES)
add_executable("week2_test" "${WEEK2_FILES}" $<TARGET_OBJECTS:test_main>)
add_test(NAME "week2_test" COMMAND "week2_test")