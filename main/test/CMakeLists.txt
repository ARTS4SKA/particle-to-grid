include(particleToGrid_add_test)
include(setup_GTest)

# Uses cstone's MPI-aware GTest driver (test_main.cpp + gtest-mpi-listener.hpp).
function(addParticleToGridTest exename source)
    add_executable(${exename} ${source} ${CSTONE_TEST_DIR}/integration_mpi/test_main.cpp)
    target_include_directories(${exename} PRIVATE
        ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${CSTONE_TEST_DIR}/integration_mpi
        ${PROJECT_SOURCE_DIR}/main/src ${PROJECT_SOURCE_DIR}/extern/io)
    target_compile_definitions(${exename} PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
    target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest io)
    # GTest is resolved from a conda environment whose lib dir ends up in the
    # RPATH; its bundled libssl.so.3 is missing symbols the system libcurl
    # needs. Prepend the system lib dir so it's found first.
    set_target_properties(${exename} PROPERTIES BUILD_RPATH "/usr/lib/x86_64-linux-gnu")
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

addParticleToGridTest(mesh_mpi mesh_tests.cpp)
addParticleToGridTest(interpolation_mpi interpolation_tests.cpp)
addParticleToGridTest(file_io_mpi file_io_tests.cpp)

particleToGrid_add_test(meshTest EXECUTABLE mesh_mpi RANKS 4)
particleToGrid_add_test(InterpolationTest EXECUTABLE interpolation_mpi RANKS 1)
particleToGrid_add_test(FileIOTest EXECUTABLE file_io_mpi RANKS 1)
particleToGrid_add_test(FileIOTestMultiRank EXECUTABLE file_io_mpi RANKS 4)
