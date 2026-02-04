include(particleToGrid_add_test)

function(addMpiTest source exename testname ranks)
    add_executable(${exename} ${source})
    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
    target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main)

    if(ENABLE_H5PART)
        enableH5Part(${exename})
    endif()

    particleToGrid_add_test(${testname} EXECUTABLE ${exename} RANKS ${ranks})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

function(addParticleToGridMpiTest source exename testname ranks)
    list(APPEND source test_main.cpp)
    addMpiTest("${source}" ${exename} ${testname} ${ranks})
endfunction()

addParticleToGridMpiTest(mesh_tests.cpp mesh_mpi meshTest 4)
addParticleToGridMpiTest(interpolation_tests.cpp interpolation_mpi InterpolationTest 1)

# File I/O tests: need io library and data path
function(addFileIoTest source exename testname ranks)
    add_executable(${exename} ${source} test_main.cpp)
    target_include_directories(${exename} PRIVATE
        ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src ${PROJECT_SOURCE_DIR}/extern/io)
    target_compile_definitions(${exename} PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
    target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main io)
    if(ENABLE_H5PART)
        enableH5Part(${exename})
    endif()
    particleToGrid_add_test(${testname} EXECUTABLE ${exename} RANKS ${ranks})
endfunction()
addFileIoTest(file_io_tests.cpp file_io_mpi FileIOTest 1)

# HDF5 output test with multiple ranks (for parallel write/read validation)
addFileIoTest(file_io_tests.cpp file_io_mpi_multi FileIOTestMultiRank 4)