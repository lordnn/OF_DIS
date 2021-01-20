cmake_minimum_required (VERSION 2.8) 
project (FastPatchOF) 

option(USE_OPENMP "OpenMP build" OFF)
set(USE_SIMD "SSE" CACHE STRING "SIMD optimization")
set_property(CACHE USE_SIMD PROPERTY STRINGS None Neon SSE)

if(MSVC)
else(MSVC)
    list(APPEND OptList -Wno-unknown-pragmas)
endif(MSVC)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${OptList} -Wall")  #-Wall
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${OptList} -Wall")  #-Wall

# OpenMP switcher
if(USE_OPENMP)
    find_package(OpenMP)
    message(STATUS "OpenMP enabled ${USE_OPENMP}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED) 
include_directories(${EIGEN3_INCLUDE_DIR})

# # ENABLE PARALLEL FLOW AGGREGATION, CAN LEAD TO DATA RACES, BUT IN PRACTICE HAS ONLY A WEAK NEGATIVE EFFECT ON THE RESULT, [affects only PatGridClass::AggregateFlowDense() ]
# # add_definitions(-DUSE_PARALLEL_ON_FLOWAGGR) 


set(CODEFILES run_dense.cpp oflow.cpp patch.cpp patchgrid.cpp refine_variational.cpp FDF1.0.1/image.cpp FDF1.0.1/opticalflow_aux.c FDF1.0.1/solver.c)

# GrayScale, Optical Flow
add_executable (run_OF_INT ${CODEFILES})
target_compile_features(run_OF_INT PRIVATE cxx_std_14)
set_target_properties (run_OF_INT PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
set_property(TARGET run_OF_INT APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=1") # use grey-valued image
TARGET_LINK_LIBRARIES(run_OF_INT ${OpenCV_LIBS})

# RGB, Optical Flow
add_executable (run_OF_RGB ${CODEFILES})
target_compile_features(run_OF_RGB PRIVATE cxx_std_14)
set_target_properties (run_OF_RGB PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
set_property(TARGET run_OF_RGB APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=3") # use RGB image
TARGET_LINK_LIBRARIES(run_OF_RGB ${OpenCV_LIBS})

# GrayScale, Depth from Stereo
add_executable (run_DE_INT ${CODEFILES})
target_compile_features(run_DE_INT PRIVATE cxx_std_14)
set_target_properties (run_DE_INT PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=2")
set_property(TARGET run_DE_INT APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=1")
TARGET_LINK_LIBRARIES(run_DE_INT ${OpenCV_LIBS})

# RGB, Depth from Stereo
add_executable (run_DE_RGB ${CODEFILES})
target_compile_features(run_DE_RGB PRIVATE cxx_std_14)
set_target_properties (run_DE_RGB PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=2")
set_property(TARGET run_DE_RGB APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=3")
TARGET_LINK_LIBRARIES(run_DE_RGB ${OpenCV_LIBS})


message(STATUS "SIMD=${USE_SIMD}")
if(${USE_SIMD} MATCHES "SSE")
    target_compile_definitions(run_OF_INT PRIVATE -DUSE_SSE)
    target_compile_definitions(run_OF_RGB PRIVATE -DUSE_SSE)
    target_compile_definitions(run_DE_INT PRIVATE -DUSE_SSE)
    target_compile_definitions(run_DE_RGB PRIVATE -DUSE_SSE)
    if(MINGW)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4")
    endif()
elseif(${USE_SIMD} MATCHES "Neon")
    target_compile_definitions(run_OF_INT PRIVATE -DUSE_NEON)
    target_compile_definitions(run_OF_RGB PRIVATE -DUSE_NEON)
    target_compile_definitions(run_DE_INT PRIVATE -DUSE_NEON)
    target_compile_definitions(run_DE_RGB PRIVATE -DUSE_NEON)
    if(ANDROID)
        if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
            add_definitions(-march=armv8-a)
        else()
            add_definitions(-mfpu=neon)
        endif()
    endif()
endif()
