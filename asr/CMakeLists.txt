# NeMo-Speech.cpp speech recognition on this backend. The upstream source is
# exported from the submodule and patches/*.patch are applied in the build
# directory, so the submodule stays untouched. A patch that no longer applies
# stops configure.
#
# NeMo is a separate CMake project (ExternalProject): it builds its own shared
# ggml, with the CPU backend linked in, from the same llama.cpp submodule, and
# loads libggml-hexagon.so from this build at run time. scripts/package.sh --asr
# collects the two.
option(GGML_HEXAGON_ASR_HTTP "Build the optional ASR-only NeMo HTTP/WebSocket server" OFF)
include(ExternalProject)
find_package(Git REQUIRED)

set(NEMO_DIR ${CMAKE_SOURCE_DIR}/NeMo-Speech.cpp)
if (NOT EXISTS ${NEMO_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "GGML_HEXAGON_ASR needs the NeMo-Speech.cpp submodule: git submodule update --init NeMo-Speech.cpp")
endif()

set(_src ${CMAKE_CURRENT_BINARY_DIR}/nemo-src)
set(_tmp ${CMAKE_CURRENT_BINARY_DIR}/patch-work)
file(GLOB _patches ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
list(SORT _patches)

# The committed submodule tree, not its working tree.
file(REMOVE_RECURSE ${_tmp})
file(MAKE_DIRECTORY ${_tmp})
execute_process(COMMAND ${GIT_EXECUTABLE} -C ${NEMO_DIR} archive --format=tar -o ${_tmp}.tar HEAD
    RESULT_VARIABLE _rc ERROR_VARIABLE _out)
if (NOT _rc EQUAL 0)
    message(FATAL_ERROR "asr: cannot export ${NEMO_DIR}:\n${_out}")
endif()
file(ARCHIVE_EXTRACT INPUT ${_tmp}.tar DESTINATION ${_tmp})
file(REMOVE ${_tmp}.tar)

# git apply works outside a repository, on paths relative to the working directory. Inside one
# (a build directory in this checkout) it skips the paths and still succeeds, so the search for
# a repository stops above the work directory, and a skipped patch is an error.
get_filename_component(_ceiling ${_tmp} DIRECTORY)
foreach(_p IN LISTS _patches)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${_ceiling}
            ${GIT_EXECUTABLE} apply --verbose --whitespace=nowarn ${_p}
        WORKING_DIRECTORY ${_tmp}
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
    if (NOT _rc EQUAL 0 OR _out MATCHES "Skipped patch")
        message(FATAL_ERROR "asr: ${_p} does not apply to ${NEMO_DIR}:\n${_out}")
    endif()
endforeach()

# Copy only what changed, so a reconfigure does not rebuild all of NeMo.
file(GLOB_RECURSE _files RELATIVE ${_tmp} ${_tmp}/*)
foreach(_f IN LISTS _files)
    get_filename_component(_d ${_src}/${_f} DIRECTORY)
    file(MAKE_DIRECTORY ${_d})
    file(COPY_FILE ${_tmp}/${_f} ${_src}/${_f} ONLY_IF_DIFFERENT)
endforeach()
file(REMOVE_RECURSE ${_tmp})

execute_process(COMMAND ${GIT_EXECUTABLE} -C ${NEMO_DIR} rev-parse --path-format=absolute --git-path HEAD
    OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_patches} ${_head})

if(GGML_HEXAGON_ASR_HTTP)
    include(${CMAKE_SOURCE_DIR}/cmake/AsrHttpDependency.cmake)
    fusa_http_dependency(_http_source)
    # git archive omits nested submodules. Copy the pinned dependency and license
    # into the exported source, never into the pristine NeMo worktree.
    file(COPY "${_http_source}/" DESTINATION "${_src}/third_party/cpp-httplib"
        PATTERN ".git" EXCLUDE)
endif()

# Both subprojects use this toolchain plus the path mapping of the main build, so
# no build-machine path ends up in the binaries (scripts/package.sh checks).
set(_toolchain ${CMAKE_CURRENT_BINARY_DIR}/toolchain.cmake)
set(_map "-ffile-prefix-map=${CMAKE_SOURCE_DIR}/= -ffile-prefix-map=${CMAKE_BINARY_DIR}/=")
set(_tc "")
if (CMAKE_TOOLCHAIN_FILE)
    get_filename_component(_tf ${CMAKE_TOOLCHAIN_FILE} ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
    string(APPEND _tc "include(\"${_tf}\")\n")
endif()
string(APPEND _tc "string(APPEND CMAKE_C_FLAGS_INIT \" ${_map}\")\nstring(APPEND CMAKE_CXX_FLAGS_INIT \" ${_map}\")\n")
file(CONFIGURE OUTPUT ${_toolchain} CONTENT "${_tc}")

# SentencePiece as a static archive for the target, at the commit NeMo's own
# scripts/build_sentencepiece_static.sh pins, unless one is given.
set(GGML_HEXAGON_ASR_SENTENCEPIECE_LIB "" CACHE FILEPATH
    "asr: static libsentencepiece.a for the target; empty builds it from source")
set(GGML_HEXAGON_ASR_SENTENCEPIECE_INCLUDE_DIR "" CACHE PATH
    "asr: directory with sentencepiece_processor.h, with GGML_HEXAGON_ASR_SENTENCEPIECE_LIB")
set(_deps "")
if (GGML_HEXAGON_ASR_SENTENCEPIECE_LIB)
    set(_spm_lib ${GGML_HEXAGON_ASR_SENTENCEPIECE_LIB})
    set(_spm_inc ${GGML_HEXAGON_ASR_SENTENCEPIECE_INCLUDE_DIR})
else()
    file(STRINGS ${_src}/scripts/build_sentencepiece_static.sh _spm_commit REGEX "^COMMIT=[0-9a-f]+$")
    string(REPLACE "COMMIT=" "" _spm_commit "${_spm_commit}")
    if (NOT _spm_commit)
        message(FATAL_ERROR "asr: no SentencePiece commit in scripts/build_sentencepiece_static.sh")
    endif()
    ExternalProject_Add(sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG        ${_spm_commit}
        PREFIX         ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_ARGS     -DCMAKE_TOOLCHAIN_FILE=${_toolchain}
                       -DCMAKE_BUILD_TYPE=Release
                       -DSPM_BUILD_TEST=OFF
                       -DSPM_ENABLE_SHARED=OFF
                       -DSPM_ENABLE_TCMALLOC=OFF
        BUILD_COMMAND  ${CMAKE_COMMAND} --build <BINARY_DIR> --target sentencepiece-static
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS <BINARY_DIR>/src/libsentencepiece.a)
    ExternalProject_Get_Property(sentencepiece SOURCE_DIR BINARY_DIR)
    set(_spm_lib ${BINARY_DIR}/src/libsentencepiece.a)
    set(_spm_inc ${SOURCE_DIR}/src)
    set(_deps sentencepiece)
endif()

# Speech recognition only, against the pristine ggml in llama.cpp. NeMo's
# NEMO_SPEECH_HEXAGON_ASR (from the patches) checks this combination.
ExternalProject_Add(nemo-speech
    SOURCE_DIR      ${_src}
    BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/nemo-build
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_ARGS      -DCMAKE_TOOLCHAIN_FILE=${_toolchain}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DBUILD_SHARED_LIBS=ON
                    -DGGML_BACKEND_DL=OFF
                    -DGGML_HEXAGON=OFF
                    -DGGML_NATIVE=OFF
                    -DGGML_CCACHE=OFF
                    -DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR=${LLAMA_DIR}
                    -DNEMO_SPEECH_GGML_PATCHED=OFF
                    -DNEMO_SPEECH_HEXAGON_ASR=ON
                    -DNEMO_SPEECH_BUILD_ASR=ON
                    -DNEMO_SPEECH_BUILD_DIAR=OFF
                    -DNEMO_SPEECH_BUILD_TTS=OFF
                    -DNEMO_SPEECH_BUILD_NMT=OFF
                    -DNEMO_SPEECH_BUILD_S2S=OFF
                    -DNEMO_SPEECH_BUILD_HTTP=${GGML_HEXAGON_ASR_HTTP}
                    -DNEMO_SPEECH_BUILD_GRPC=OFF
                    -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF
                    -DNEMO_SPEECH_BUILD_TESTS=ON
                    -DSENTENCEPIECE_STATIC_LIB=${_spm_lib}
                    -DSENTENCEPIECE_INCLUDE_DIR=${_spm_inc}
    BUILD_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target nemo_speech_cli nemo_speech_asr_c test_backend_storage
    BUILD_ALWAYS    ON
    INSTALL_COMMAND ""
    DEPENDS         ${_deps})

# Built with the default target, after the backend it loads.
add_custom_target(asr ALL DEPENDS nemo-speech)
add_dependencies(asr ggml-hexagon)
