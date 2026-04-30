# CMakeLists.txt for gen_xe4_tmm_layouts
#
# Builds a host-side C++ binary that uses CuTe's real XE4 TMM layout types to
# generate TikZ/LaTeX documentation images for each MMA atom configuration.
#
# Standalone build (from repo root):
#   source /opt/intel/oneapi/setvars.sh   # required: sets icpx in PATH
#   cmake -S media/docs/cpp/cute -B build/xe4_tmm_doc -DCMAKE_CXX_COMPILER=icpx
#   cmake --build build/xe4_tmm_doc
#   # .tex files are written to build/xe4_tmm_doc/tex/
#
# Or from the repo's top-level CMake via add_subdirectory:
#   add_subdirectory(media/docs/cpp/cute)
#
# Requirements:
#   - Intel oneAPI (icpx) is required because mma_traits_xe4_tmm.hpp pulls in
#     <sycl/sycl.hpp> for type definitions (e.g. fp16 = sycl::half).
#   - No GPU is needed at run-time; the generator runs entirely on the host.

cmake_minimum_required(VERSION 3.22)

# Allow the project block to be skipped when added via add_subdirectory
# (the parent project already called project())
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(xe4_tmm_layouts CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate the repo's include/ directory.
# This file lives at <repo>/media/docs/cpp/cute/CMakeLists.txt, so the repo
# root is four levels up.
# ---------------------------------------------------------------------------
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
                       ABSOLUTE)
set(CUTE_INCLUDE_DIR "${REPO_ROOT}/include")

if(NOT EXISTS "${CUTE_INCLUDE_DIR}/cute/layout.hpp")
  message(FATAL_ERROR
    "Cannot find CuTe headers at ${CUTE_INCLUDE_DIR}. "
    "Please run cmake from the repository root or set CUTE_INCLUDE_DIR.")
endif()

# ---------------------------------------------------------------------------
# Warn if the compiler does not look like icpx; the SYCL headers that
# mma_traits_xe4_tmm.hpp transitively includes require the Intel oneAPI
# toolchain (icpx).  icx/clang++-sycl should also work.
# ---------------------------------------------------------------------------
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
  message(WARNING
    "Compiler '${CMAKE_CXX_COMPILER}' may not support <sycl/sycl.hpp>. "
    "Set CXX=icpx (after sourcing /opt/intel/oneapi/setvars.sh) for a "
    "guaranteed build.  Continuing anyway — configure-time check only.")
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(gen_xe4_tmm_layouts gen_xe4_tmm_layouts.cpp)

target_include_directories(gen_xe4_tmm_layouts PRIVATE "${CUTE_INCLUDE_DIR}")

# CUTLASS_ENABLE_SYCL makes cute/util/debug.hpp include <sycl/sycl.hpp>
# instead of <cuda_runtime_api.h>, which is what we want for the Intel path.
target_compile_definitions(gen_xe4_tmm_layouts PRIVATE CUTLASS_ENABLE_SYCL=1 
      SYCL_INTEL_TARGET=40 SYCL_INTEL_XE4_TARGET=1)
target_compile_options(gen_xe4_tmm_layouts PRIVATE -fsycl )

# ---------------------------------------------------------------------------
# Custom target: run the binary at build time to produce .tex files.
# The output directory is ${CMAKE_CURRENT_BINARY_DIR}/tex/.
# ---------------------------------------------------------------------------
set(TEX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/tex")
set(TEX_CONFIGS "f16f16_M32N1K16" "f16f16_M32N8K16")

# Collect expected .tex output paths
set(TEX_OUTPUTS "")
foreach(CONFIG ${TEX_CONFIGS})
  list(APPEND TEX_OUTPUTS "${TEX_OUTPUT_DIR}/XE4_TMM.${CONFIG}.tex")
endforeach()

add_custom_command(
  OUTPUT ${TEX_OUTPUTS}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${TEX_OUTPUT_DIR}"
  COMMAND "$<TARGET_FILE:gen_xe4_tmm_layouts>" "${TEX_OUTPUT_DIR}"
  DEPENDS gen_xe4_tmm_layouts
  COMMENT "Generating XE4 TMM atom TikZ/LaTeX images in ${TEX_OUTPUT_DIR}"
  VERBATIM
)

add_custom_target(gen_xe4_tmm_tex_images ALL DEPENDS ${TEX_OUTPUTS})

# ---------------------------------------------------------------------------
# Optional: compile .tex → .pdf → .png and update media/images/cute/.
# Requires pdflatex and pdftoppm (from poppler-utils).
# Run:  sudo apt install texlive-latex-base poppler-utils
# ---------------------------------------------------------------------------
find_program(PDFLATEX pdflatex)
find_program(PDFTOPPM pdftoppm)

if(PDFLATEX AND PDFTOPPM)
  set(PNG_OUTPUTS "")
  foreach(CONFIG ${TEX_CONFIGS})
    set(TEX_FILE  "${TEX_OUTPUT_DIR}/XE4_TMM.${CONFIG}.tex")
    set(PDF_FILE  "${TEX_OUTPUT_DIR}/XE4_TMM.${CONFIG}.pdf")
    set(PNG_STEM  "${TEX_OUTPUT_DIR}/XE4_TMM.${CONFIG}")
    set(PNG_SRC   "${PNG_STEM}.png")
    set(PNG_DEST  "${REPO_ROOT}/media/images/cute/XE4_TMM.${CONFIG}.png")

    add_custom_command(
      OUTPUT "${PNG_DEST}"
      COMMAND "${PDFLATEX}"
              -output-directory "${TEX_OUTPUT_DIR}"
              -interaction=nonstopmode
              "${TEX_FILE}"
      COMMAND "${PDFTOPPM}"
              -png -r 150 -singlefile
              "${PDF_FILE}" "${PNG_STEM}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${REPO_ROOT}/media/images/cute"
      COMMAND "${CMAKE_COMMAND}" -E copy "${PNG_SRC}" "${PNG_DEST}"
      DEPENDS "${TEX_FILE}"
      COMMENT "Compiling ${TEX_FILE} → ${PNG_DEST}"
      VERBATIM
    )
    list(APPEND PNG_OUTPUTS "${PNG_DEST}")
  endforeach()

  add_custom_target(gen_xe4_tmm_png_images ALL
    DEPENDS gen_xe4_tmm_tex_images ${PNG_OUTPUTS}
    COMMENT "XE4 TMM PNG images updated in ${REPO_ROOT}/media/images/cute/"
  )
  message(STATUS "PNG image generation enabled (pdflatex + pdftoppm found).")
else()
  message(STATUS
    "pdflatex or pdftoppm not found; PNG image generation skipped. "
    "Install texlive-latex-base and poppler-utils to enable it.")
endif()
