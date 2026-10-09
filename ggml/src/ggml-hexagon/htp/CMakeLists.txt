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

if (HEXAGON_HTP_YIELD_IN_CPY AND NOT HEXAGON_HTP_YIELD_BETWEEN_OPS)
    message(FATAL_ERROR "HEXAGON_HTP_YIELD_IN_CPY requires HEXAGON_HTP_YIELD_BETWEEN_OPS")
endif()

set(HTP_LIB ggml-htp-${DSP_VERSION})

add_library(${HTP_LIB} SHARED
    main.c
    htp_iface_skel.c
    work-queue.c
    dma-queue.c
    hmx-queue.c
    htp-tensor.c
    matmul-ops.c
    flash-attn-ops.c
    gated-delta-net-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    act-ops.c
    rope-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
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
    $<$<BOOL:${HEXAGON_HTP_YIELD_IN_CPY}>:HTP_YIELD_IN_CPY=1>)

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

# Embed the SWIV CRC in place, so install and signing see the verified image
if (HEXAGON_SWIV_UTILITY)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(TARGET ${HTP_LIB} POST_BUILD
        COMMAND ${Python3_EXECUTABLE} ${HEXAGON_SWIV_UTILITY} -i $<TARGET_FILE:${HTP_LIB}> -o $<TARGET_FILE:${HTP_LIB}>.swiv
        COMMAND ${CMAKE_COMMAND} -E rename $<TARGET_FILE:${HTP_LIB}>.swiv $<TARGET_FILE:${HTP_LIB}>
        COMMENT "embedding SWIV CRC into lib${HTP_LIB}.so"
        VERBATIM)
endif()

install(TARGETS ${HTP_LIB})
