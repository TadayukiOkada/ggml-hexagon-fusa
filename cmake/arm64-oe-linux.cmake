# Cross toolchain for ARM64 Linux images built with OpenEmbedded / Yocto.
# Use the image's own SDK so the binaries match its glibc and libstdc++.
#
# OE_TOOLCHAIN_ROOT is the SDK root with:
#   sysroots/x86_64-qtisdk-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-{gcc,g++}
#   sysroots/armv8a-oe-linux/
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if (NOT DEFINED ENV{OE_TOOLCHAIN_ROOT})
    message(FATAL_ERROR "OE_TOOLCHAIN_ROOT is not set")
endif()
file(TO_CMAKE_PATH "$ENV{OE_TOOLCHAIN_ROOT}" OE_ROOT)
set(OE_BIN "${OE_ROOT}/sysroots/x86_64-qtisdk-linux/usr/bin/aarch64-oe-linux")

set(CMAKE_C_COMPILER   "${OE_BIN}/aarch64-oe-linux-gcc")
set(CMAKE_CXX_COMPILER "${OE_BIN}/aarch64-oe-linux-g++")
set(CMAKE_SYSROOT      "${OE_ROOT}/sysroots/armv8a-oe-linux")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# The compiler check runs before CMAKE_SYSROOT applies, so pass it in the flags too.
set(CMAKE_C_FLAGS_INIT   "--sysroot=${CMAKE_SYSROOT} -march=armv8.2-a+fp16+dotprod")
set(CMAKE_CXX_FLAGS_INIT "--sysroot=${CMAKE_SYSROOT} -march=armv8.2-a+fp16+dotprod")
