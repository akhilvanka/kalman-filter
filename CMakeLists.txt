cmake_minimum_required(VERSION 3.16)
project(KalmanFilter VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

include_directories(include)

add_executable(ekf_demo  src/ekf.cpp src/main.cpp)
add_executable(ekf_tests src/ekf.cpp tests/test_ekf.cpp)

enable_testing()
add_test(NAME ekf_tests COMMAND ekf_tests)
