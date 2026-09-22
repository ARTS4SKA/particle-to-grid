function(particleToGrid_add_test name)
    cmake_parse_arguments(ARG "" "EXECUTABLE;RANKS" "" ${ARGN})
    if(NOT ARG_RANKS)
        set(ARG_RANKS 1)
    endif()
    add_test(NAME ${name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS} $<TARGET_FILE:${ARG_EXECUTABLE}>)
endfunction()
