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
    add_dependencies(ggml ${backend})
    install(TARGETS ${backend} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})

    # g++ before 13 has no _Float16 in C++; __fp16 is the same IEEE half.
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("int main() { return sizeof(_Float16) == 2 ? 0 : 1; }" GGML_HEXAGON_HAVE_CXX_FLOAT16)
    if (NOT GGML_HEXAGON_HAVE_CXX_FLOAT16)
        target_compile_definitions(${backend} PRIVATE _Float16=__fp16)
    endif()
endfunction()

include(GNUInstallDirs)
add_subdirectory(ggml/src/ggml-hexagon)
