######### CMake Version #####################
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
#############################################
set( CMAKE_DISABLE_SOURCE_CHANGES  ON )
set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )
#############################################

######### Options ###########################
option( VF_USE_CUDA    "Use CUDA to speed up certain parts of the code."              ON )
option( VF_BUILD_TEST  "Build small test project using the library."                  ON )
#############################################

######### Meta information about the project
set(META_PROJECT_NAME         "vectorfield")
set(META_PROJECT_DESCRIPTION  "vectorfield numerics library with cuda backend")
set(META_AUTHOR_ORGANIZATION  "")
set(META_AUTHOR_DOMAIN        "https://github.com/GPMueller")
set(META_AUTHOR_MAINTAINER    "gideon.mueller.91@outlook.com")
set(META_VERSION_MAJOR        "0")
set(META_VERSION_MINOR        "0")
set(META_VERSION_PATCH        "0")
set(META_VERSION_REVISION     "${GIT_REV}")
set(META_VERSION              "${META_VERSION_MAJOR}.${META_VERSION_MINOR}.${META_VERSION_PATCH}")
set(META_NAME_VERSION         "${META_PROJECT_NAME} v${META_VERSION} (${META_VERSION_REVISION})")
#############################################

######### CUDA decisions ####################
if (VF_USE_CUDA)
    MESSAGE( STATUS ">> --------------  USING CUDA --------------" )
    ### Set cuda toolkit path 
    if( (NOT CUDA_TOOLKIT_ROOT_DIR) )
        set( CUDA_TOOLKIT_ROOT_DIR /opt/cuda )
    endif()
    if (APPLE OR UNIX)
        set(CMAKE_C_COMPILER   ${CUDA_TOOLKIT_ROOT_DIR}/bin/gcc)
        set(CMAKE_CXX_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/g++)
    elseif (WIN32)
        MESSAGE( STATUS ">> We are on Windows... CUDA Path needs to be set..." )
    endif()
endif()
#############################################


######### Info ##############################
MESSAGE( STATUS ">> CMAKE_C_COMPILER:        " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER:      " ${CMAKE_CXX_COMPILER} )
#############################################


######### Project Name ######################
project( ${META_PROJECT_NAME} VERSION ${META_VERSION} )
#############################################


######### CUDA decisions ####################
if (VF_USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11 -arch=sm_30 --expt-relaxed-constexpr" )
endif()
#############################################



######### Files #############################
set(HEADER_FILES
    include/vectorfield.hpp
    include/managed_allocator.hpp
    include/kernel_vectormath.hpp
    include/kernel_manifold.hpp
)
source_group( "include" FILES ${HEADER_FILES} )

set(SOURCE_FILES
    src/kernel_vectormath.cu
    src/kernel_vectormath.cpp
    src/kernel_manifold.cu
    src/kernel_manifold.cpp
)
source_group( "src" FILES ${SOURCE_FILES} )
#############################################



######### libvectorfield ####################
### Build and Link Library
if ( VF_USE_CUDA )
    include_directories( ${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
    cuda_add_library( ${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES} )
    target_link_libraries( ${PROJECT_NAME} ${CUDA_LIBRARIES} )
else()
    add_library( ${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES} )
endif()
### Include Directories
target_include_directories( ${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories( ${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty)
### Properties
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)
### Install
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include DESTINATION include/vectorfield)
#############################################


######### Test executable ###################
set( TEST_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
### Test creation macro
macro( add_framework_test testName testSrc )
    # Executable
    add_executable( ${testName} test/main.cpp ${testSrc} )
    # Link Library
    target_link_libraries( ${testName} ${PROJECT_NAME} )
    # Properties
    set_property(TARGET ${testName} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TARGET ${testName} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${testName} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${testName} PROPERTY CXX_EXTENSIONS OFF)
    # Include Directories
    target_include_directories( ${testName} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    # Add the test
    add_test( NAME        ${testName}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMAND           ${testName} --use-colour=yes )
endmacro( add_framework_test testName testSrc )
### Create tests if needed
if ( VF_BUILD_TEST )
    ### Enable CTest testing
    enable_testing()

    ###### Test 1
    add_framework_test( vftest test/test.cpp )
endif()
#############################################