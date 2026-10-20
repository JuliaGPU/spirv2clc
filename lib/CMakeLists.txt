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
  groups.cpp
  instns_ext.cpp)

set_source_files_properties(${SPIRV2CLC_UNITY_SOURCES}
  PROPERTIES HEADER_FILE_ONLY ON)

# A library target named `name` with the translator and its C++ API, of the
# type given as the optional second argument (by default, per
# BUILD_SHARED_LIBS).
function(spirv2clc_add_translator name)
  add_library(${name} ${ARGN}
    translator.cpp
    translator.h
    cexpr.cpp
    cexpr.h
    ${SPIRV2CLC_UNITY_SOURCES}
    "${PROJECT_SOURCE_DIR}/include/spirv2clc.h")

  set_target_properties(${name} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

  target_include_directories(${name}
    PUBLIC
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_BINARY_DIR}" # For the generated export header
    PRIVATE
      "${OPENCL_HEADERS_DIR}"
      "${SPIRV_TOOLS_DIR}"
      "${SPIRV_TOOLS_DIR}/source"
      "${SPIRV_TOOLS_BINARY_DIR}")

  target_link_libraries(${name}
    PUBLIC SPIRV-Tools-opt
    PRIVATE spirv2clc::warnings)
endfunction()

spirv2clc_add_translator(libspirv2clc)
add_library(spirv2clc::libspirv2clc ALIAS libspirv2clc)

generate_export_header(libspirv2clc)

# The C API (include/libspirv2clc.h), as a shared library that exports nothing
# else: the translator, SPIRV-Tools and the C++ standard library templates
# they instantiate are linked in and kept local, so that the library can be
# loaded into a process that has its own copies of them, and no C++ type
# crosses its interface.
#
# That needs the translator as a static library, so build a second copy of it
# when the C++ library is shared.
if(BUILD_SHARED_LIBS)
  spirv2clc_add_translator(libspirv2clc-static STATIC)
  target_compile_definitions(libspirv2clc-static
    PUBLIC LIBSPIRV2CLC_STATIC_DEFINE)
  set(_c_api_translator libspirv2clc-static)
else()
  set(_c_api_translator libspirv2clc)
endif()

add_library(libspirv2clc-c SHARED
  libspirv2clc.cpp
  "${PROJECT_SOURCE_DIR}/include/libspirv2clc.h")
add_library(spirv2clc::c ALIAS libspirv2clc-c)

set_target_properties(libspirv2clc-c PROPERTIES
  OUTPUT_NAME spirv2clc
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(libspirv2clc-c
  PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(libspirv2clc-c
  PRIVATE
    ${_c_api_translator}
    spirv2clc::warnings)

# On Windows, only the functions declared with SPIRV2CLC_API are exported.
if(APPLE)
  target_link_options(libspirv2clc-c PRIVATE
    "LINKER:-exported_symbol,_SPIRV2CLC*")
elseif(NOT WIN32)
  set(_version_script "${CMAKE_CURRENT_BINARY_DIR}/libspirv2clc.map")
  file(WRITE "${_version_script}" "{ global: SPIRV2CLC*; local: *; };\n")
  target_link_options(libspirv2clc-c PRIVATE
    "LINKER:--version-script=${_version_script}")
  set_property(TARGET libspirv2clc-c APPEND PROPERTY
    LINK_DEPENDS "${_version_script}")
endif()

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

# The C API gets a component of its own: it is self-contained, so it can be
# shipped (e.g. next to the tool) without the static library and its headers.
install(TARGETS libspirv2clc-c
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT c_api
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT c_api
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT c_api)
install(FILES "${PROJECT_SOURCE_DIR}/include/libspirv2clc.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT c_api)
