# HipCxx23Shim.cmake -- work around the HIP <cmath> / C++23 overload conflict
# WITHOUT pinning the language standard, and without breaking on a fixed toolchain.
#
# THE BUG. C++23 (P0533R9) makes the standard math functions constexpr. HIP treats a
# constexpr function as implicitly __host__ __device__, so clang's
# __clang_cuda_math_forward_declares.h -- which declares isfinite/isinf/isnan/isnormal
# __device__-only -- can no longer overload them:
#
#   error: __device__ function 'isfinite' cannot overload __host__ __device__ function 'isfinite'
#
# Fixed upstream by llvm/llvm-project#201563 (merged 2026-06-09), which includes the
# forward-declares BEFORE <cmath> in __clang_hip_runtime_wrapper.h. Toolchains predating
# that fix still fail. Today this is MSVC-only because MSVC's STL is the only one that has
# implemented P0533R9; when libstdc++ does, Linux fails identically, which is why nothing
# here is guarded on WIN32.
#
# WHAT THIS DOES. The driver force-includes __clang_hip_runtime_wrapper.h through the normal
# include search, so a -I directory takes precedence over the compiler's resource dir. When
# (and only when) the installed toolchain is affected, this generates a copy of THE INSTALLED
# wrapper with the two includes reordered, and hands back a directory to put on the include
# path. Language standard untouched, no _HAS_CXX23 override, so host and device translation
# units keep identical language AND standard-library configuration -- unlike -std=c++20 or
# -D_HAS_CXX23=0, both of which desynchronise the STL across the boundary and risk ODR.
#
# FORWARD COMPATIBILITY, which is the whole point:
#   * The shim is generated from the INSTALLED header at configure time, never checked in,
#     so it cannot shadow a newer toolchain with a stale copy.
#   * If the installed header already has the fix, this returns empty and adds nothing.
#   * If a probe compile succeeds without the shim, this returns empty and adds nothing
#     (covers a libstdc++ host, or any toolchain where the conflict does not arise).
#   * After generating, it RE-PROBES with the shim. If the shim does not actually fix the
#     build it is discarded and a warning is issued, rather than silently shadowing a header
#     with something that does not help.
# So on ROCm 10.1 and later this is inert: it detects the fix and no-ops.
#
# USAGE:
#   include(HipCxx23Shim)
#   hip_cxx23_shim(SHIM_DIR)
#   if(SHIM_DIR)
#     target_include_directories(my_hip_target BEFORE PRIVATE "${SHIM_DIR}")
#   endif()
#
# The directory must come BEFORE the compiler's own include paths; use BEFORE.

function(hip_cxx23_shim OUT_VAR)
  set(${OUT_VAR} "" PARENT_SCOPE)

  set(_compiler "${CMAKE_HIP_COMPILER}")
  if(NOT _compiler)
    set(_compiler "${CMAKE_CXX_COMPILER}")
  endif()
  if(NOT _compiler)
    message(STATUS "hip_cxx23_shim: no HIP or CXX compiler known; skipping")
    return()
  endif()

  set(_std "${CMAKE_HIP_STANDARD}")
  if(NOT _std)
    set(_std "${CMAKE_CXX_STANDARD}")
  endif()
  if(NOT _std)
    set(_std 23)
  endif()

  set(_work "${CMAKE_BINARY_DIR}/hip-cxx23-shim")
  file(MAKE_DIRECTORY "${_work}")
  set(_probe "${_work}/probe.hip")
  file(WRITE "${_probe}"
    "#include <hip/hip_runtime.h>\n#include <cmath>\n__global__ void k(float* p){ *p = 1.0f; }\n")

  set(_arch_flag "")
  if(CMAKE_HIP_ARCHITECTURES)
    list(GET CMAKE_HIP_ARCHITECTURES 0 _arch0)
    set(_arch_flag "--offload-arch=${_arch0}")
  endif()

  # 1. Does it already build? Then there is nothing to do.
  execute_process(
    COMMAND "${_compiler}" -std=c++${_std} ${_arch_flag} -x hip -c "${_probe}"
            -o "${_work}/probe_plain.o"
    RESULT_VARIABLE _plain_rc OUTPUT_QUIET ERROR_VARIABLE _plain_err)
  if(_plain_rc EQUAL 0)
    message(STATUS "hip_cxx23_shim: toolchain builds HIP at C++${_std}; no shim needed")
    return()
  endif()

  # Only act on THIS diagnostic. Any other failure is not ours to paper over.
  if(NOT _plain_err MATCHES "cannot overload __host__ __device__")
    message(STATUS
      "hip_cxx23_shim: HIP probe failed for an unrelated reason; not shimming.")
    return()
  endif()

  # 2. Locate the installed wrapper via the compiler's own resource dir.
  execute_process(COMMAND "${_compiler}" -print-resource-dir
                  OUTPUT_VARIABLE _resdir OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _rc ERROR_QUIET)
  if(NOT _rc EQUAL 0)
    message(WARNING "hip_cxx23_shim: could not query -print-resource-dir; not shimming")
    return()
  endif()
  file(TO_CMAKE_PATH "${_resdir}" _resdir)
  set(_wrapper "${_resdir}/include/__clang_hip_runtime_wrapper.h")
  if(NOT EXISTS "${_wrapper}")
    message(WARNING "hip_cxx23_shim: ${_wrapper} not found; not shimming")
    return()
  endif()

  # 3. Reorder, but only if it is actually in the broken order.
  file(STRINGS "${_wrapper}" _lines)
  set(_fwd_line 0)
  set(_cmath_line 0)
  set(_i 0)
  foreach(_l IN LISTS _lines)
    math(EXPR _i "${_i} + 1")
    string(STRIP "${_l}" _s)
    if(_cmath_line EQUAL 0 AND _s STREQUAL "#include <cmath>")
      set(_cmath_line ${_i})
    endif()
    if(_fwd_line EQUAL 0 AND _s MATCHES "^#include.*__clang_cuda_math_forward_declares\.h")
      set(_fwd_line ${_i})
    endif()
  endforeach()
  if(_fwd_line EQUAL 0 OR _cmath_line EQUAL 0)
    message(WARNING "hip_cxx23_shim: wrapper has an unexpected shape; not shimming")
    return()
  endif()
  if(_fwd_line LESS _cmath_line)
    message(STATUS "hip_cxx23_shim: installed wrapper already carries the fix; no shim")
    return()
  endif()

  set(_out "")
  set(_i 0)
  foreach(_l IN LISTS _lines)
    math(EXPR _i "${_i} + 1")
    if(_i EQUAL _fwd_line)
      continue()
    endif()
    if(_i EQUAL _cmath_line)
      string(APPEND _out "#include <__clang_cuda_math_forward_declares.h>\n")
    endif()
    string(APPEND _out "${_l}\n")
  endforeach()
  file(WRITE "${_work}/__clang_hip_runtime_wrapper.h" "${_out}")

  # 4. Prove the shim actually helps before handing it back.
  execute_process(
    COMMAND "${_compiler}" -std=c++${_std} ${_arch_flag} -x hip "-I${_work}"
            -c "${_probe}" -o "${_work}/probe_shim.o"
    RESULT_VARIABLE _shim_rc OUTPUT_QUIET ERROR_QUIET)
  if(NOT _shim_rc EQUAL 0)
    file(REMOVE "${_work}/__clang_hip_runtime_wrapper.h")
    message(WARNING
      "hip_cxx23_shim: generated shim did not fix the probe; discarding it. "
      "Build HIP sources at C++20, or upgrade to a toolchain carrying "
      "llvm/llvm-project#201563 (ROCm 10.1 or newer).")
    return()
  endif()

  message(STATUS
    "hip_cxx23_shim: toolchain predates llvm/llvm-project#201563; "
    "generated reordered wrapper in ${_work}")
  set(${OUT_VAR} "${_work}" PARENT_SCOPE)
endfunction()
