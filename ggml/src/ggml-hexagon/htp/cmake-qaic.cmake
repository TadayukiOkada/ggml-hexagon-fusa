# build_idl() from hexagon_fun.cmake looks for the IDL compiler at ipc/fastrpc/qaic/bin/qaic.
# The SDKs ship one binary per host OS (Ubuntu20/ in 5.5.x, Ubuntu/ in 6.x) and create bin/
# in their own Makefile, which an SDK may not have run.
# Stage a copy in the build tree instead of writing into the SDK.
# Include after hexagon_fun.cmake.

if (NOT EXISTS "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin")
    if (CMAKE_HOST_WIN32)
        set(_qaic_host_dir "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/WinNT")
    else()
        set(_qaic_host_dir "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu20")
        if (NOT IS_DIRECTORY "${_qaic_host_dir}")
            set(_qaic_host_dir "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu")
        endif()
    endif()
    if (NOT IS_DIRECTORY "${_qaic_host_dir}")
        message(FATAL_ERROR "qaic not found under ${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic")
    endif()
    file(COPY "${_qaic_host_dir}/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/qaic/bin")
    set(proj-qaic_src "${CMAKE_CURRENT_BINARY_DIR}/qaic")
    unset(_qaic_host_dir)
endif()
