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

# The C API test is written in C, so that it also checks the header is valid C.
# It assembles its inputs with the SPIRV-Tools C API, a copy of which it links
# next to the one hidden inside the library.
add_executable(capi_test capi_test.c)
target_include_directories(capi_test PRIVATE "${SPIRV_TOOLS_DIR}/include")
target_link_libraries(capi_test
  PRIVATE
    spirv2clc::c
    SPIRV-Tools-static
    spirv2clc::warnings)
set_target_properties(capi_test PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  # SPIRV-Tools is C++
  LINKER_LANGUAGE CXX)
add_test(NAME capi COMMAND capi_test)

# The library exports the C API and nothing else.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_program(NM_EXECUTABLE NAMES nm)
  if(NM_EXECUTABLE)
    add_test(NAME capi_exports
      COMMAND sh -c "! '${NM_EXECUTABLE}' -D --defined-only \"$0\" | grep -v ' SPIRV2CLC'"
              "$<TARGET_FILE:spirv2clc::c>")
  endif()
endif()
