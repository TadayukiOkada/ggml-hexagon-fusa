cmake_minimum_required(VERSION 3.22.2)
project(ggml-htp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake-qaic.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

if ((HEXAGON_HTP_YIELD_IN_CPY OR HEXAGON_HTP_YIELD_IN_OPS) AND NOT HEXAGON_HTP_YIELD_BETWEEN_OPS)
    message(FATAL_ERROR "HEXAGON_HTP_YIELD_IN_CPY and HEXAGON_HTP_YIELD_IN_OPS require HEXAGON_HTP_YIELD_BETWEEN_OPS")
endif()

set(HTP_LIB ggml-htp-${DSP_VERSION})

add_library(${HTP_LIB} SHARED
    main.c
    htp_iface_skel.c
    work-queue.c
    dma-queue.c
    hmx-queue.c
    htp-dcache.c
    matmul-ops.c
    flash-attn-ops.c
    gated-delta-net-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    glu-ops.c
    rope-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
    gather-rows.c
    repeat-ops.c
    ssm-conv.c
    cumsum-ops.c
    fill-ops.c
    concat-ops.c
    diag-ops.c
    solve-tri-ops.c
    pad-ops.c
    argsort-ops.c
    im2col-ops.c
    roll-ops.c
    pool-ops.c
    allreduce-ops.c
)

target_compile_definitions(${HTP_LIB} PRIVATE
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>
    $<$<BOOL:${HEXAGON_HTP_YIELD_BETWEEN_OPS}>:HTP_YIELD_BETWEEN_OPS=1>
    $<$<BOOL:${HEXAGON_HTP_YIELD_IN_CPY}>:HTP_YIELD_IN_CPY=1>
    $<$<BOOL:${HEXAGON_HTP_YIELD_IN_OPS}>:HTP_YIELD_IN_OPS=1>)

if (HEXAGON_BUILD_ID)
    target_compile_definitions(${HTP_LIB} PRIVATE HTP_BUILD_ID="${HEXAGON_BUILD_ID}")
endif()

# Keep build-machine paths out of the skel (__FILE__ in asserts).
target_compile_options(${HTP_LIB} PRIVATE
    -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=htp/
    -ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}/=)

# SDK 5.5.x has no HMX voltage corners MIN_SVS and SVS_L2. Use the nearest lower corner.
file(STRINGS ${HEXAGON_SDK_ROOT}/incs/HAP_power.h _hap_exp_vcorners REGEX "HAP_DCVS_EXP_VCORNER_(MIN_SVS|SVS_L2) *=")
if (NOT _hap_exp_vcorners MATCHES "HAP_DCVS_EXP_VCORNER_MIN_SVS")
    target_compile_definitions(${HTP_LIB} PRIVATE HAP_DCVS_EXP_VCORNER_MIN_SVS=HAP_DCVS_EXP_VCORNER_MIN)
endif()
if (NOT _hap_exp_vcorners MATCHES "HAP_DCVS_EXP_VCORNER_SVS_L2")
    target_compile_definitions(${HTP_LIB} PRIVATE HAP_DCVS_EXP_VCORNER_SVS_L2=HAP_DCVS_EXP_VCORNER_SVS_L1)
endif()

build_idl(htp_iface.idl ${HTP_LIB})

set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

# The linker records its command line, with build-machine paths, in .comment.
add_custom_command(TARGET ${HTP_LIB} POST_BUILD
    COMMAND ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-llvm-objcopy --remove-section=.comment $<TARGET_FILE:${HTP_LIB}>
    VERBATIM)

if (NOT Python3_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# HEXAGON_SWIV_UTILITY replaces htp-swiv.py with the firmware's own swiv_build_utility.py.
# It does not write in place, so go through a temporary file.
if (HEXAGON_SWIV_UTILITY)
    add_custom_command(TARGET ${HTP_LIB} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Adding SWIV section $<TARGET_FILE_NAME:${HTP_LIB}> with ${HEXAGON_SWIV_UTILITY}"
        COMMAND ${Python3_EXECUTABLE} ${HEXAGON_SWIV_UTILITY} -i $<TARGET_FILE:${HTP_LIB}> -o $<TARGET_FILE:${HTP_LIB}>.swiv
        COMMAND ${CMAKE_COMMAND} -E rename $<TARGET_FILE:${HTP_LIB}>.swiv $<TARGET_FILE:${HTP_LIB}>
        VERBATIM)
else()
    add_custom_command(TARGET ${HTP_LIB} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Adding SWIV section $<TARGET_FILE_NAME:${HTP_LIB}>"
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/htp-swiv.py -i $<TARGET_FILE:${HTP_LIB}> -o $<TARGET_FILE:${HTP_LIB}>
        VERBATIM)
endif()

install(TARGETS ${HTP_LIB})
