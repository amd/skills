#!/usr/bin/env python3
"""
Link wrapper for CMake's HIP language on Windows.

On Windows, CMake's HIP link rule runs clang in --hip-link mode with the GNU-style
driver but passes it MSVC-style arguments: -fuse-ld=lld-link, bare linker flags
such as /subsystem:console, and import libraries spelled -lfoo.lib or foo.lib. The
GNU-style driver takes the bare flags for input files and does not hand the
libraries to lld-link in a form it resolves, so the link fails. This wrapper
rewrites the argument list and then runs the real link:
  - drops -fuse-ld=* (it conflicts with --hip-link)
  - turns "-Xlinker <flag>" pairs and bare MSVC /flags into -Wl,<flag>
  - turns -lfoo.lib and bare foo.lib into "-Xlinker foo.lib", which lld-link
    resolves through the LIB environment variable
Response files are not read or rewritten.

Usage (as CMAKE_HIP_LINK_EXECUTABLE, on Windows only):
  python <this_script> <clang_exe> [args...]

This is unrelated to relocatable device code (-fgpu-rdc): any project that links
HIP objects through CMake on Windows needs it. Standard library only.
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
