pybind11_add_module(particle_to_grid_py bindings.cpp)

# The import name must be the module name; the CMake target name differs to avoid clashing
# with the `particle-to-grid` executable target.
set_target_properties(particle_to_grid_py PROPERTIES OUTPUT_NAME particle_to_grid)

target_include_directories(particle_to_grid_py PRIVATE
    ${PROJECT_SOURCE_DIR}/main/src
    ${CSTONE_DIR}
    ${MPI_CXX_INCLUDE_PATH}
)
target_link_libraries(particle_to_grid_py PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_compile_options(particle_to_grid_py PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

if(BUILD_TESTING)
    # The tests need numpy, and the serial one needs pytest. Report rather than fail the build,
    # since the module itself is usable without them.
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy" RESULT_VARIABLE have_numpy ERROR_QUIET)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import pytest" RESULT_VARIABLE have_pytest ERROR_QUIET)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import h5py" RESULT_VARIABLE have_h5py ERROR_QUIET)

    # The freshly built module lands next to this file in the build tree.
    set(module_dir "$<TARGET_FILE_DIR:particle_to_grid_py>")

    if(have_numpy EQUAL 0 AND have_pytest EQUAL 0)
        add_test(NAME PythonBindingsTest
            COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py)
        set_tests_properties(PythonBindingsTest PROPERTIES ENVIRONMENT "PYTHONPATH=${module_dir}")
    else()
        message(STATUS "No PythonBindingsTest: ${Python_EXECUTABLE} needs numpy and pytest")
    endif()

    if(have_numpy EQUAL 0)
        add_test(NAME PythonBindingsTestMultiRank
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mpi_bindings.py)
        set_tests_properties(PythonBindingsTestMultiRank PROPERTIES ENVIRONMENT "PYTHONPATH=${module_dir}")
    else()
        message(STATUS "No PythonBindingsTestMultiRank: ${Python_EXECUTABLE} needs numpy")
    endif()

    # CDM checkpoint -> HDF5. Only the small grid runs here; the 1024 and 2048 cases need
    # P2G_HEAVY=1 (see the test's docstring). Skips itself when scripts/CDM_6Mpc_2048.00001
    # is absent, since that file is not in the repository.
    if(have_numpy EQUAL 0 AND have_pytest EQUAL 0 AND have_h5py EQUAL 0)
        add_test(NAME PythonCdmDensityTest
            COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cdm_density.py)
        set_tests_properties(PythonCdmDensityTest PROPERTIES
            ENVIRONMENT "PYTHONPATH=${module_dir};P2G_OUT_DIR=${CMAKE_CURRENT_BINARY_DIR}" TIMEOUT 600)
    else()
        message(STATUS "No PythonCdmDensityTest: ${Python_EXECUTABLE} needs numpy, pytest and h5py")
    endif()
endif()
