# Like asr/, build a patched export; never change the pinned upstream worktree.
cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(fusa-sensevoice LANGUAGES C CXX)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()
get_filename_component(FUSA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(LLAMA_DIR "${FUSA_ROOT}/llama.cpp" CACHE PATH "Pinned llama.cpp checkout")
option(GGML_HEXAGON_SENSEVOICE_HTTP "Build optional persistent SenseVoice HTTP/WebSocket server" OFF)
include(ExternalProject)
find_package(Git REQUIRED)
set(_upstream "${FUSA_ROOT}/SenseVoice")
if(NOT EXISTS "${_upstream}/runtime/llama.cpp/funasr-sensevoice/funasr-sensevoice.cpp")
    message(FATAL_ERROR "SenseVoice submodule missing: git submodule update --init SenseVoice")
endif()
set(_src "${CMAKE_CURRENT_BINARY_DIR}/sensevoice-src")
set(_tmp "${CMAKE_CURRENT_BINARY_DIR}/patch-work")
file(GLOB _patches CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
list(SORT _patches)
file(REMOVE_RECURSE "${_tmp}")
file(MAKE_DIRECTORY "${_tmp}")
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_upstream}" archive --format=tar
    -o "${_tmp}.tar" HEAD runtime/llama.cpp LICENSE RESULT_VARIABLE _rc ERROR_VARIABLE _out)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "sensevoice: cannot export submodule: ${_out}")
endif()
file(ARCHIVE_EXTRACT INPUT "${_tmp}.tar" DESTINATION "${_tmp}")
file(REMOVE "${_tmp}.tar")
# Outside any repository, as in asr/: inside this checkout git apply would skip the paths.
foreach(_patch IN LISTS _patches)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
            "${GIT_EXECUTABLE}" apply --verbose --whitespace=nowarn "${_patch}"
        WORKING_DIRECTORY "${_tmp}" RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
    if(NOT _rc EQUAL 0 OR _out MATCHES "Skipped patch")
        message(FATAL_ERROR "sensevoice: ${_patch} does not apply: ${_out}")
    endif()
endforeach()
file(GLOB_RECURSE _files RELATIVE "${_tmp}" "${_tmp}/*")
foreach(_file IN LISTS _files)
    get_filename_component(_dir "${_src}/${_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dir}")
    file(COPY_FILE "${_tmp}/${_file}" "${_src}/${_file}" ONLY_IF_DIFFERENT)
endforeach()
file(REMOVE_RECURSE "${_tmp}")
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_upstream}" rev-parse --path-format=absolute --git-path HEAD
    OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_patches} "${_head}")

set(_target_args "")
if(CMAKE_TOOLCHAIN_FILE)
    get_filename_component(_toolchain "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    list(APPEND _target_args "-DCMAKE_TOOLCHAIN_FILE=${_toolchain}")
endif()
foreach(_var ANDROID_ABI ANDROID_PLATFORM CMAKE_ANDROID_ARCH_ABI CMAKE_ANDROID_API)
    if(DEFINED ${_var})
        list(APPEND _target_args "-D${_var}=${${_var}}")
    endif()
endforeach()
ExternalProject_Add(sensevoice-runtime
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/sensevoice-build"
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
               "-DFUSA_SOURCE_ROOT=${FUSA_ROOT}" "-DFUSA_BINARY_ROOT=${CMAKE_BINARY_DIR}"
               "-DSENSEVOICE_SOURCE_DIR=${_src}/runtime/llama.cpp"
               "-DFUSA_SENSEVOICE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
               "-DLLAMA_DIR=${LLAMA_DIR}" "-DFUSA_ROOT=${FUSA_ROOT}"
               "-DGGML_HEXAGON_SENSEVOICE_HTTP=${GGML_HEXAGON_SENSEVOICE_HTTP}" ${_target_args}
    BUILD_ALWAYS ON
    INSTALL_COMMAND "")
add_custom_target(sensevoice ALL DEPENDS sensevoice-runtime)
if(TARGET ggml-hexagon)
    add_dependencies(sensevoice ggml-hexagon)
endif()
