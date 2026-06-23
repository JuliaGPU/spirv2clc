# Copyright 2020-2022 The spirv2clc authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 3.24 gives us CMAKE_COLOR_DIAGNOSTICS and PROJECT_IS_TOP_LEVEL; the bundled
# SPIRV-Tools requires at least 3.22.1. The upper bound opts in to newer policy
# defaults without warnings on recent CMake.
cmake_minimum_required(VERSION 3.24...3.31)

project(spirv2clc
  VERSION 0.1.0
  DESCRIPTION "Experimental OpenCL SPIR-V to OpenCL C translator"
  HOMEPAGE_URL "https://github.com/kpet/spirv2clc"
  LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Project-wide settings (only applied when spirv2clc is the top-level project,
# so that embedding it via add_subdirectory() does not override the parent's
# choices).
# ---------------------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL)
  # Guard against in-source builds.
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
      "In-source builds are not allowed. Configure into a separate directory, "
      "e.g. `cmake --preset default` (builds in ./build).")
  endif()

  # Default to a useful build type for single-config generators.
  get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      Debug Release RelWithDebInfo MinSizeRel)
  endif()

  # Quality-of-life defaults: a compile_commands.json for tooling (clangd,
  # clang-tidy, editors) and coloured diagnostics regardless of generator.
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CMAKE_COLOR_DIAGNOSTICS ON)

  # Collect all build artifacts in predictable locations under the build tree.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Compiler warnings, scoped to our own targets via an INTERFACE target so that
# the bundled third-party code is built with its own (laxer) flags.
# ---------------------------------------------------------------------------
add_library(spirv2clc-warnings INTERFACE)
add_library(spirv2clc::warnings ALIAS spirv2clc-warnings)
if(MSVC)
  target_compile_options(spirv2clc-warnings INTERFACE /W4 /WX)
else()
  target_compile_options(spirv2clc-warnings INTERFACE -Wall -Wextra -pedantic)
endif()

# ---------------------------------------------------------------------------
# Dependencies (vendored as git submodules under external/).
#
# spirv2clc links against SPIRV-Tools' *internal* optimizer API (opt/ir_context,
# BuildModule, the analysis managers), which is not part of any installed
# package, so SPIRV-Tools must be built from source.
# ---------------------------------------------------------------------------
find_package(Git QUIET)
option(SPIRV2CLC_UPDATE_SUBMODULES
  "Initialize/update the git submodules under external/ during configure" ON)
if(SPIRV2CLC_UPDATE_SUBMODULES AND GIT_FOUND AND
   EXISTS "${PROJECT_SOURCE_DIR}/.git")
  message(STATUS "Updating git submodules under external/")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE _git_submod_result)
  if(NOT _git_submod_result EQUAL 0)
    message(FATAL_ERROR
      "`git submodule update --init --recursive` failed with "
      "${_git_submod_result}. Update the submodules manually or set "
      "-DSPIRV2CLC_UPDATE_SUBMODULES=OFF.")
  endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/CMakeLists.txt")
  message(FATAL_ERROR
    "The SPIRV-Tools submodule is missing. Run "
    "`git submodule update --init --recursive` and re-configure.")
endif()

set(OPENCL_HEADERS_DIR "${PROJECT_SOURCE_DIR}/external/OpenCL-Headers")
set(SPIRV-Headers_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/SPIRV-Headers")
set(SPIRV_TOOLS_DIR "${PROJECT_SOURCE_DIR}/external/SPIRV-Tools")
set(SPIRV_TOOLS_BINARY_DIR "${CMAKE_BINARY_DIR}/external/SPIRV-Tools")

# We only need the SPIRV-Tools libraries, not its tests (which pull in
# effcee/re2/abseil/googletest) and not warnings-as-errors on its sources
# (the bundled version may lag behind the host compiler).
set(SPIRV_SKIP_TESTS ON)
set(SPIRV_WERROR OFF CACHE BOOL "Treat SPIRV-Tools warnings as errors")
add_subdirectory("${SPIRV_TOOLS_DIR}" "${SPIRV_TOOLS_BINARY_DIR}" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# spirv2clc itself.
# ---------------------------------------------------------------------------
add_subdirectory(lib)
add_subdirectory(tools)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
