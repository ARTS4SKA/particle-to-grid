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
addParticleToGridTest(roundtrip_mpi roundtrip_tests.cpp)

# CPU/GPU agreement for SPH deposition. Needs the CUDA backend, so it is built like the
# particle-to-grid-cuda executable: host sources with USE_CUDA plus mesh.cu.
if(CMAKE_CUDA_COMPILER)
    add_executable(sph_cuda_mpi sph_cuda_tests.cpp ${PROJECT_SOURCE_DIR}/main/src/mesh.cu
        ${CSTONE_TEST_DIR}/integration_mpi/test_main.cpp)
    target_include_directories(sph_cuda_mpi PRIVATE
        ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${CSTONE_TEST_DIR}/integration_mpi
        ${PROJECT_SOURCE_DIR}/main/src ${PROJECT_SOURCE_DIR}/extern/io)
    target_compile_definitions(sph_cuda_mpi PRIVATE USE_CUDA TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
    target_link_libraries(sph_cuda_mpi PRIVATE cstone_gpu io GTest::gtest ${MPI_CXX_LIBRARIES} CUDA::cudart)
    set_target_properties(sph_cuda_mpi PROPERTIES BUILD_RPATH "/usr/lib/x86_64-linux-gnu")
    if(ENABLE_H5PART)
        enableH5Part(sph_cuda_mpi)
    endif()
    particleToGrid_add_test(SphCudaTest EXECUTABLE sph_cuda_mpi RANKS 1)
    particleToGrid_add_test(SphCudaTestMultiRank EXECUTABLE sph_cuda_mpi RANKS 4)
endif()

particleToGrid_add_test(meshTest EXECUTABLE mesh_mpi RANKS 4)
particleToGrid_add_test(InterpolationTest EXECUTABLE interpolation_mpi RANKS 1)
particleToGrid_add_test(FileIOTest EXECUTABLE file_io_mpi RANKS 1)
particleToGrid_add_test(FileIOTestMultiRank EXECUTABLE file_io_mpi RANKS 4)
particleToGrid_add_test(RoundTripTest EXECUTABLE roundtrip_mpi RANKS 1)
particleToGrid_add_test(RoundTripTestMultiRank EXECUTABLE roundtrip_mpi RANKS 4)
