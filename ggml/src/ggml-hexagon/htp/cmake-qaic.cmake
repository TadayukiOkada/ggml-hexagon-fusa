# SDK 6.x has the IDL compiler at ipc/fastrpc/qaic/bin/qaic, where build_idl() from hexagon_fun.cmake looks.
# SDK 5.5.x has one binary per host OS and creates bin/ in its own Makefile.
# Stage a copy in the build tree instead of writing into the SDK.
# Include after hexagon_fun.cmake.

if (NOT EXISTS "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin")
    if (CMAKE_HOST_WIN32)
        set(_qaic_host_dir "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/WinNT")
    else()
        set(_qaic_host_dir "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu20")
    endif()
    if (NOT IS_DIRECTORY "${_qaic_host_dir}")
        message(FATAL_ERROR "qaic not found under ${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic")
    endif()
    file(COPY "${_qaic_host_dir}/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/qaic/bin")
    set(proj-qaic_src "${CMAKE_CURRENT_BINARY_DIR}/qaic")
    unset(_qaic_host_dir)
endif()
