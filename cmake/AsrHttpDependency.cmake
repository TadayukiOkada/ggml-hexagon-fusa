# Same cpp-httplib commit as the pinned NeMo submodule (includes WebSockets).
# Fetch only for optional HTTP builds; CLI-only builds remain offline-capable.
include(FetchContent)
function(fusa_http_dependency output)
    FetchContent_Declare(fusa_httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG 62d899feac3cf9215a55f2b43da250fdd98d2156
        GIT_PROGRESS OFF
        SOURCE_SUBDIR fusa-no-cmake)
    FetchContent_MakeAvailable(fusa_httplib)
    set(${output} "${fusa_httplib_SOURCE_DIR}" PARENT_SCOPE)
endfunction()
