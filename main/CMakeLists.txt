add_subdirectory(src)

if(BUILD_TESTING)
    add_subdirectory(test)
endif(BUILD_TESTING)

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
