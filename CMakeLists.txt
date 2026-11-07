cmake_minimum_required(VERSION 3.21)
project(ggml-hexagon-fusa C CXX)

# llama.cpp is built unmodified, with every backend as a loadable module.
# Its own Hexagon backend stays off; ours is built below from ggml/src/ggml-hexagon.
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# Same layout as a standalone llama.cpp build: binaries and backend modules in bin/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(BUILD_SHARED_LIBS ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   ON  CACHE BOOL "" FORCE)
set(GGML_HEXAGON      OFF CACHE BOOL "" FORCE)

# Keep build-machine paths out of the binaries: __FILE__ in asserts and aborts,
# and the build-tree RPATH (everything lands in bin/, so $ORIGIN is enough).
add_compile_options(-ffile-prefix-map=${CMAKE_SOURCE_DIR}/= -ffile-prefix-map=${CMAKE_BINARY_DIR}/=)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# llama.cpp turns these off when it is not the top-level project.
option(LLAMA_BUILD_COMMON   "llama: build common utils library" ON)
option(LLAMA_BUILD_TOOLS    "llama: build tools"                ON)
option(LLAMA_BUILD_SERVER   "llama: build server"               ON)
option(LLAMA_BUILD_TESTS    "llama: build tests"                OFF)
option(LLAMA_BUILD_EXAMPLES "llama: build examples"             OFF)

add_subdirectory(${LLAMA_DIR} llama.cpp)

# v0.9.0 has FP8 tensor/CPU kernels but omits it from the common CLI's KV whitelist.
# Opt in for evaluation, without editing the pinned submodule. All four cache-type
# options (target/draft, K/V) use this same whitelist. No model/graph code is patched.
option(GGML_HEXAGON_EXPERIMENTAL_FP8_KV "Expose f8_e4m3 in llama.cpp's KV cache CLI" OFF)
if(GGML_HEXAGON_EXPERIMENTAL_FP8_KV)
    file(READ ${LLAMA_DIR}/common/arg.cpp _arg)
    set(_old "const std::vector<ggml_type> kv_cache_types = {\n")
    string(FIND "${_arg}" "${_old}" _pos)
    if(_pos LESS 0)
        message(FATAL_ERROR "FP8 KV: upstream CLI whitelist changed; review the adapter")
    endif()
    string(REPLACE "${_old}" "${_old}    GGML_TYPE_F8_E4M3,\n" _arg "${_arg}")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/arg-fp8-kv.cpp.tmp "${_arg}")
    configure_file(${CMAKE_CURRENT_BINARY_DIR}/arg-fp8-kv.cpp.tmp ${CMAKE_CURRENT_BINARY_DIR}/arg-fp8-kv.cpp COPYONLY)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/arg-fp8-kv.cpp.tmp)
    get_target_property(_common_sources llama-common SOURCES)
    list(FILTER _common_sources EXCLUDE REGEX "(^|/)arg\\.cpp$")
    set_property(TARGET llama-common PROPERTY SOURCES ${_common_sources} ${CMAKE_CURRENT_BINARY_DIR}/arg-fp8-kv.cpp)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLAMA_DIR}/common/arg.cpp)
    # llama-bench has its own parser rather than common/arg.cpp.
    if(TARGET llama-bench-impl)
        file(READ ${LLAMA_DIR}/tools/llama-bench/llama-bench.cpp _bench)
        set(_old_bench "static ggml_type ggml_type_from_name(const std::string & s) {\n")
        string(FIND "${_bench}" "${_old_bench}" _pos_bench)
        if(_pos_bench LESS 0)
            message(FATAL_ERROR "FP8 KV: upstream benchmark type parser changed; review the adapter")
        endif()
        string(REPLACE "${_old_bench}" "${_old_bench}    if (s == \"f8_e4m3\") return GGML_TYPE_F8_E4M3;\n" _bench "${_bench}")
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bench-fp8-kv.cpp.tmp "${_bench}")
        configure_file(${CMAKE_CURRENT_BINARY_DIR}/bench-fp8-kv.cpp.tmp ${CMAKE_CURRENT_BINARY_DIR}/bench-fp8-kv.cpp COPYONLY)
        file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/bench-fp8-kv.cpp.tmp)
        get_target_property(_bench_sources llama-bench-impl SOURCES)
        list(FILTER _bench_sources EXCLUDE REGEX "(^|/)llama-bench\\.cpp$")
        set_property(TARGET llama-bench-impl PROPERTY SOURCES ${_bench_sources} ${CMAKE_CURRENT_BINARY_DIR}/bench-fp8-kv.cpp)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLAMA_DIR}/tools/llama-bench/llama-bench.cpp)
    endif()
endif()

# The vendored backend CMakeLists calls ggml_add_backend_library() as it would
# inside ggml/src. Redefine it for a backend that lives outside the ggml tree:
# a module that links ggml-base and sees the ggml internal headers.
function(ggml_add_backend_library backend)
    add_library(${backend} MODULE ${ARGN})
    set_target_properties(${backend} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    target_compile_definitions(${backend} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD)
    target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
    target_include_directories(${backend} BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${LLAMA_DIR}/ggml/src)
    target_link_libraries(${backend} PRIVATE ggml-base)
    target_compile_features(${backend} PRIVATE c_std_11 cxx_std_17)  # as ggml/src does for its targets
    add_dependencies(ggml ${backend})
    install(TARGETS ${backend} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})

    # g++ before 13 has no _Float16 in C++; __fp16 is the same IEEE half.
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("int main() { return sizeof(_Float16) == 2 ? 0 : 1; }" GGML_HEXAGON_HAVE_CXX_FLOAT16)
    if (NOT GGML_HEXAGON_HAVE_CXX_FLOAT16)
        target_compile_definitions(${backend} PRIVATE _Float16=__fp16)
    endif()
endfunction()

option(GGML_HEXAGON_FUSA "ggml-hexagon: build for functional-safety DSP firmware and check the settings it needs" OFF)
set(GGML_HEXAGON_FUSA_SHELL "$ENV{GGML_HEXAGON_FUSA_SHELL}" CACHE FILEPATH
    "ggml-hexagon: DSP firmware shell (fastrpc_shell_*) to check the skel imports against; optional")
if (GGML_HEXAGON_FUSA)
    if (NOT GGML_HEXAGON_HTP_YIELD_BETWEEN_OPS OR NOT GGML_HEXAGON_HTP_YIELD_IN_CPY OR NOT GGML_HEXAGON_HTP_YIELD_IN_OPS)
        message(FATAL_ERROR "GGML_HEXAGON_FUSA needs GGML_HEXAGON_HTP_YIELD_BETWEEN_OPS, _IN_CPY and _IN_OPS")
    endif()
    if (NOT EXISTS "${GGML_HEXAGON_SWIV_UTILITY}")
        message(FATAL_ERROR "GGML_HEXAGON_FUSA needs GGML_HEXAGON_SWIV_UTILITY (swiv_build_utility.py); got '${GGML_HEXAGON_SWIV_UTILITY}'")
    endif()
endif()

include(cmake/build-id.cmake)
include(GNUInstallDirs)
add_subdirectory(ggml/src/ggml-hexagon)

if (GGML_HEXAGON_FUSA)
    # Leave ops that would hold the DSP past half of a 50 ms self-test period without a
    # yield point to the CPU (GGML_HEXAGON_OP_BUDGET_MS overrides at run time, 0 = off).
    target_compile_definitions(ggml-hexagon PRIVATE GGML_HEXAGON_OP_BUDGET_MS_DEFAULT=25)
endif()

if (LLAMA_BUILD_TESTS)
    add_subdirectory(tests)
endif()

option(GGML_HEXAGON_ASR "ggml-hexagon: build NeMo-Speech.cpp speech recognition (asr/) against this backend" OFF)
if (GGML_HEXAGON_ASR)
    add_subdirectory(asr)
endif()

if (GGML_HEXAGON_FUSA)
    # Check every skel after it is built: SWIV section, and imports against the firmware shell if given.
    foreach(V IN LISTS GGML_HEXAGON_HTP_ARCHS)
        add_custom_target(check-htp-${V} ALL
            COMMAND ${CMAKE_SOURCE_DIR}/scripts/check-skel.sh
                    ${CMAKE_BINARY_DIR}/ggml/src/ggml-hexagon/libggml-htp-${V}.so
                    ${GGML_HEXAGON_HTP_TOOLS_ROOT} "${GGML_HEXAGON_FUSA_SHELL}"
            DEPENDS htp-${V}
            COMMENT "checking libggml-htp-${V}.so"
            VERBATIM)
    endforeach()
endif()

option(GGML_HEXAGON_SENSEVOICE "ggml-hexagon: build the SenseVoice llama.cpp runtime" OFF)
if(GGML_HEXAGON_SENSEVOICE)
    add_subdirectory(sensevoice)
endif()
