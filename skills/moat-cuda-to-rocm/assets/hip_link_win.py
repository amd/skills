#!/usr/bin/env python3
"""
Link wrapper for CMake's HIP language on Windows, for ONE arrangement: the host
code is compiled with clang-cl and the target uses relocatable device code
(-fgpu-rdc).

In that arrangement CMake composes the link line for an MSVC linker (bare flags
such as /machine:x64 and /subsystem:console, import libraries spelled foo.lib),
while -fgpu-rdc needs the GNU-style driver to do the device link. The GNU-style
driver reads the bare flags and libraries as input files and the link fails. This
wrapper rewrites the argument list and then runs the real link:
  - turns bare MSVC /flags into -Wl,<flag>
  - turns bare foo.lib into "-Xlinker foo.lib", which lld-link resolves through
    the LIB environment variable
  - drops -fuse-ld=* (CMake's -fuse-ld=lld-link is rejected by the -fgpu-rdc
    device link; it is harmless without -fgpu-rdc)
  - turns "-Xlinker <flag>" pairs into -Wl,<flag> and -lfoo.lib into
    "-Xlinker foo.lib" (neither made a difference in any configuration tested)

A project that compiles everything with the GNU-style clang++ does not need this
script: without -fgpu-rdc CMake's own link rule works, and with it one line,
set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld"), is the whole fix.

Limits:
  - Response files are not read or rewritten. Once the link line is long enough
    for the build tool to move it into @file.rsp, the libraries in that file are
    not rewritten and the link fails.
  - CMAKE_HIP_LINK_EXECUTABLE covers executables only; a shared library needs the
    same treatment through CMAKE_HIP_CREATE_SHARED_LIBRARY.

Usage (as CMAKE_HIP_LINK_EXECUTABLE, on Windows only):
  python <this_script> <clang_exe> [args...]

Standard library only.
"""
import subprocess
import sys


def transform(args):
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        # CMake injects -fuse-ld=lld-link (Windows-Clang platform default)
        # into <LINK_FLAGS>; it conflicts with --hip-link. Drop it.
        if a.startswith('-fuse-ld='):
            i += 1
            continue
        # -Xlinker <flag> pairs from CMake: convert to -Wl,<flag> which is
        # handled correctly in --hip-link mode.
        if a == '-Xlinker' and i + 1 < len(args):
            out.append('-Wl,' + args[i + 1])
            i += 2
            continue
        # -lXXX.lib: GCC-driver lib syntax for Windows import libs. lld-link
        # ignores -lXXX.lib. Convert to -Xlinker XXX.lib so lld-link finds
        # the lib in the LIB env path.
        if a.startswith('-l') and a.endswith('.lib'):
            lib_name = a[2:]  # strip -l prefix
            out += ['-Xlinker', lib_name]
            i += 1
            continue
        # Bare MSVC-style flags (start with / and no file extension)
        base = a.split(':')[0]
        if a.startswith('/') and '.' not in base:
            out.append('-Wl,' + a)
        # Bare .lib names (no directory separators): wrap for GCC driver
        elif a.endswith('.lib') and '/' not in a and '\\' not in a:
            out += ['-Xlinker', a]
        else:
            out.append(a)
        i += 1
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: hip_link_win.py <clang_exe> [args...]")
    clang = sys.argv[1]
    transformed = transform(sys.argv[2:])
    sys.exit(subprocess.run([clang] + transformed).returncode)


if __name__ == '__main__':
    main()
