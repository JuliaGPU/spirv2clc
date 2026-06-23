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

include(GenerateExportHeader)
include(GNUInstallDirs)

# translator.cpp is a unity translation unit: it #includes the sources below
# directly, so they must NOT be compiled on their own. They are still listed as
# (header-only) sources so editors and IDEs pick them up.
set(SPIRV2CLC_UNITY_SOURCES
  utils.cpp
  types.cpp
  instns.cpp
  instns_ext.cpp)

add_library(libspirv2clc
  translator.cpp
  ${SPIRV2CLC_UNITY_SOURCES}
  "${PROJECT_SOURCE_DIR}/include/spirv2clc.h")
add_library(spirv2clc::libspirv2clc ALIAS libspirv2clc)

set_source_files_properties(${SPIRV2CLC_UNITY_SOURCES}
  PROPERTIES HEADER_FILE_ONLY ON)

set_target_properties(libspirv2clc PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(libspirv2clc
  PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}" # For the generated export header
  PRIVATE
    "${OPENCL_HEADERS_DIR}"
    "${SPIRV_TOOLS_DIR}"
    "${SPIRV_TOOLS_DIR}/source"
    "${SPIRV_TOOLS_BINARY_DIR}")

target_link_libraries(libspirv2clc
  PUBLIC SPIRV-Tools-opt
  PRIVATE spirv2clc::warnings)

generate_export_header(libspirv2clc)

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install(TARGETS libspirv2clc
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT libraries
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT libraries
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT libraries)
install(FILES
    "${PROJECT_SOURCE_DIR}/include/spirv2clc.h"
    "${CMAKE_CURRENT_BINARY_DIR}/libspirv2clc_export.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT headers)
install(DIRECTORY "${SPIRV_TOOLS_DIR}/include"
  DESTINATION .
  COMPONENT headers
  FILES_MATCHING PATTERN "spirv-tools/libspirv.h")
install(DIRECTORY "${SPIRV-Headers_SOURCE_DIR}/include"
  DESTINATION .
  COMPONENT headers
  FILES_MATCHING PATTERN "spirv/unified1/spirv.h")
