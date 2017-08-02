cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(core)

# This must be run using a conda environment
if("$ENV{CONDA_PREFIX}" STREQUAL "")
    message(FATAL_ERROR "This must be run within the MolSSI SSS conda prefix. "
                        "Delete the build directory and try again")
endif()

# CMake will sometimes find the conda python version
# (rather the the python inside the sss environment)
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python3)

# Find PyBind11
set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11: ${pybind11_CONFIG}")

pybind11_add_module(core core.cc)
set(CMAKE_CXX_FLAGS "-fopenmp")

find_package(TargetLAPACK REQUIRED)
target_link_libraries(core PRIVATE tgt::lapack)

pybind11_add_module(JK_builder_C MODULE JK_builder_C.cpp)
