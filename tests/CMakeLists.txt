# test-backend-ops with the extra cases for this backend. The upstream source is
# copied from the submodule and patches/*.patch are applied at configure time,
# so the submodule stays untouched. A patch that no longer applies stops configure.
find_package(Git REQUIRED)

set(_src ${LLAMA_DIR}/tests/test-backend-ops.cpp)
set(_tmp ${CMAKE_CURRENT_BINARY_DIR}/patch-work)
set(_dst ${CMAKE_CURRENT_BINARY_DIR}/test-backend-ops-hex.cpp)
file(GLOB _patches ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
list(SORT _patches)

# git apply works outside a repository, on paths relative to the working directory.
file(REMOVE_RECURSE ${_tmp})
file(MAKE_DIRECTORY ${_tmp}/tests)
file(COPY_FILE ${_src} ${_tmp}/tests/test-backend-ops.cpp)
foreach(_p IN LISTS _patches)
    execute_process(COMMAND ${GIT_EXECUTABLE} apply --whitespace=nowarn ${_p}
        WORKING_DIRECTORY ${_tmp}
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
    if (NOT _rc EQUAL 0)
        message(FATAL_ERROR "tests: ${_p} does not apply to ${_src}:\n${_out}")
    endif()
endforeach()
file(COPY_FILE ${_tmp}/tests/test-backend-ops.cpp ${_dst} ONLY_IF_DIFFERENT)
file(REMOVE_RECURSE ${_tmp})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_src} ${_patches})

add_executable(test-backend-ops-hex ${_dst})
target_include_directories(test-backend-ops-hex PRIVATE ${LLAMA_DIR}/tests)
target_link_libraries(test-backend-ops-hex PRIVATE llama llama-common)
