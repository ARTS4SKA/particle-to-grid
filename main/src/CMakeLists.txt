function(enableGpuDirect exename)
    if(GPU_DIRECT)
        target_compile_definitions(${exename} PRIVATE USE_GPU_DIRECT)
    endif()
endfunction()

set(exename particle-to-grid)
add_executable(${exename} particle-to-grid.cpp)
message(STATUS "CSTONE_DIR=${CSTONE_DIR}")
target_include_directories(${exename} PUBLIC ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/extern/io ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(${exename} PRIVATE io OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CMAKE_CUDA_COMPILER)
    add_executable(${exename}-cuda particle-to-grid.cpp mesh.cu)
    target_include_directories(${exename}-cuda PUBLIC
        ${CMAKE_BINARY_DIR}/main/src
        ${CSTONE_DIR}
        ${PROJECT_SOURCE_DIR}/extern/io
        ${MPI_CXX_INCLUDE_PATH}
    )
    target_compile_definitions(${exename}-cuda PRIVATE USE_CUDA)
    target_link_libraries(${exename}-cuda PRIVATE cstone_gpu io OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart)
    
    enableGpuDirect(${exename}-cuda)
    install(TARGETS ${exename}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# elseif(CMAKE_HIP_COMPILER)
#     add_executable(${exename}-hip power_spectrum.cpp)
#     target_compile_definitions(${exename}-hip PRIVATE USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
#     target_link_libraries(${exename}-hip PRIVATE cstone_gpu io OpenMP::OpenMP_CXX
#         ${MPI_CXX_LIBRARIES} hip::host)
#     set_target_properties(${exename}-hip PROPERTIES LINKER_LANGUAGE CXX)
#     enableGpuDirect(${exename}-hip)
#     install(TARGETS ${exename}-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
