# Source part of the Hexagon build ID: a hash over the backend sources, plus the
# git commit for people reading logs. The backend CMakeLists appends the skel
# configuration. Editing any backend source re-runs configure, so the ID follows.
file(GLOB_RECURSE _hex_sources LIST_DIRECTORIES false
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/*
    ${CMAKE_SOURCE_DIR}/ggml/include/ggml-hexagon.h)
list(SORT _hex_sources)
set(_hex_digests "")
foreach(_f IN LISTS _hex_sources)
    file(SHA256 ${_f} _d)
    string(APPEND _hex_digests "${_d}")
endforeach()
string(SHA256 _hex_hash "${_hex_digests}")
string(SUBSTRING "${_hex_hash}" 0 12 _hex_hash)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_hex_sources})

find_package(Git QUIET)
set(_hex_git "nogit")
if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev=9
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE _hex_git_out
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _hex_git_rc)
    if (_hex_git_rc EQUAL 0)
        set(_hex_git "${_hex_git_out}")
    endif()
endif()

set(GGML_HEXAGON_BUILD_SRC_ID "${_hex_git}.${_hex_hash}")
