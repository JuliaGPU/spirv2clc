// Copyright 2020-2022 The spirv2clc authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "libspirv2clc_export.h"
#include <spirv-tools/libspirv.h>

namespace spirv2clc {

struct translator_impl;

struct translator {

  // `opencl_c_version` selects the OpenCL C language version to emit for,
  // encoded as in `CL_TARGET_OPENCL_VERSION` (120, 200, 300). Constructs that
  // aren't available at that version (e.g. the generic address space below 2.0)
  // are diagnosed rather than silently emitted.
  LIBSPIRV2CLC_EXPORT translator(spv_target_env env = SPV_ENV_OPENCL_1_2,
                                 unsigned opencl_c_version = 120);

  LIBSPIRV2CLC_EXPORT translator(translator &&);
  LIBSPIRV2CLC_EXPORT translator &operator=(translator &&);
  LIBSPIRV2CLC_EXPORT ~translator();

  LIBSPIRV2CLC_EXPORT int translate(const std::string &assembly,
                                    std::string *srcout);
  LIBSPIRV2CLC_EXPORT int translate(const std::vector<uint32_t> &binary,
                                    std::string *srcout);

private:
  std::unique_ptr<translator_impl> m_impl;
};

} // namespace spirv2clc
