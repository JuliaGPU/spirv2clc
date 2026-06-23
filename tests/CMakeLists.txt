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

if(NOT WIN32)
  # testlayer doesn't work on Windows
  add_subdirectory(testlayer)
endif()

# ---------------------------------------------------------------------------
# lit-based regression tests
#
# Each case under cases/ drives spirv2clc through clang and FileCheck. The
# tests are only registered when the required tools are available; otherwise
# they are skipped with a status message rather than failing configuration.
# ---------------------------------------------------------------------------
find_program(LIT_COMMAND NAMES lit llvm-lit)
find_program(CLANG_EXECUTABLE NAMES clang)
find_program(FILECHECK_EXECUTABLE NAMES FileCheck)

if(LIT_COMMAND AND CLANG_EXECUTABLE AND FILECHECK_EXECUTABLE)
  # Single-config generators place the tool directly in the runtime output
  # directory; this is the common case for running the lit suite.
  set(SPIRV2CLC_TOOL
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/spirv2clc${CMAKE_EXECUTABLE_SUFFIX}")

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
    @ONLY)

  # Run via CTest (`ctest` / `ctest --preset default`). The tool must be built
  # first, e.g. `cmake --build build && ctest --test-dir build`.
  add_test(NAME lit
    COMMAND "${LIT_COMMAND}" -sv "${CMAKE_CURRENT_BINARY_DIR}")

  # Convenience target that builds the tool and then runs the suite in one go.
  add_custom_target(check
    COMMAND "${LIT_COMMAND}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS spirv2clc
    COMMENT "Running spirv2clc regression tests"
    USES_TERMINAL)
else()
  set(_missing_tools "")
  foreach(tool LIT_COMMAND CLANG_EXECUTABLE FILECHECK_EXECUTABLE)
    if(NOT ${tool})
      string(APPEND _missing_tools " ${tool}")
    endif()
  endforeach()
  message(STATUS
    "lit tests disabled; missing tools:${_missing_tools}")
endif()
