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

#include "libspirv2clc.h"
#include "spirv2clc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

char *make_message(const char *str, size_t len) {
  char *p = static_cast<char *>(std::malloc(len + 1));
  if (p) {
    std::memcpy(p, str, len);
    p[len] = '\0';
  }
  return p;
}

char *make_message(const std::string &str) {
  return make_message(str.data(), str.size());
}

// The validation environment for SPIR-V version `major`.`minor`, or
// SPV_ENV_MAX if there is none.
spv_target_env target_env(unsigned major, unsigned minor) {
  if (major == 0 && minor == 0) {
    return SPV_ENV_OPENCL_1_2;
  }
  if (major != 1) {
    return SPV_ENV_MAX;
  }
  switch (minor) {
  case 0:
    return SPV_ENV_UNIVERSAL_1_0;
  case 1:
    return SPV_ENV_UNIVERSAL_1_1;
  case 2:
    return SPV_ENV_UNIVERSAL_1_2;
  case 3:
    return SPV_ENV_UNIVERSAL_1_3;
  case 4:
    return SPV_ENV_UNIVERSAL_1_4;
  case 5:
    return SPV_ENV_UNIVERSAL_1_5;
  case 6:
    return SPV_ENV_UNIVERSAL_1_6;
  }
  return SPV_ENV_MAX;
}

} // namespace

extern "C" {

int SPIRV2CLCTranslate(const char *SPIRV, size_t Length,
                       const SPIRV2CLCTranslateOptions *Options,
                       SPIRV2CLCDiagnosticCallback Handler,
                       void *HandlerContext, char **OutSource,
                       char **OutMessage) {
  if (OutSource) {
    *OutSource = nullptr;
  }
  if (OutMessage) {
    *OutMessage = nullptr;
  }
  auto fail = [&](const char *message, size_t len) {
    if (OutMessage) {
      *OutMessage = make_message(message, len);
    }
    return 1;
  };
  auto fail_with = [&](const std::string &message) {
    return fail(message.data(), message.size());
  };

  // No C++ exception may unwind into the caller, so any (e.g. a failed
  // allocation) fails the translation instead.
  try {
    if (!OutSource) {
      return fail_with("no output pointer given");
    }
    if (Length % sizeof(uint32_t) != 0 || (!SPIRV && Length != 0)) {
      return fail_with("the input is not a SPIR-V binary (its size is not a "
                       "multiple of 4 bytes)");
    }

    SPIRV2CLCTranslateOptions defaults = {};
    if (!Options) {
      Options = &defaults;
    }
    spv_target_env env =
        target_env(Options->SPIRVVersionMajor, Options->SPIRVVersionMinor);
    if (env == SPV_ENV_MAX) {
      return fail_with("unsupported SPIR-V version " +
                       std::to_string(Options->SPIRVVersionMajor) + "." +
                       std::to_string(Options->SPIRVVersionMinor));
    }
    unsigned opencl_c_version =
        Options->OpenCLCVersion ? Options->OpenCLCVersion : 120;
    if (opencl_c_version != 120 && opencl_c_version != 200 &&
        opencl_c_version != 300) {
      return fail_with("unsupported OpenCL C version " +
                       std::to_string(opencl_c_version));
    }

    // The input need not be aligned for uint32_t.
    std::vector<uint32_t> binary(Length / sizeof(uint32_t));
    if (Length != 0) {
      std::memcpy(binary.data(), SPIRV, Length);
    }

    std::string errors;
    spirv2clc::translator translator(env, opencl_c_version);
    translator.set_diagnostic_handler(
        [&](spirv2clc::severity sev, const std::string &message) {
          SPIRV2CLCDiagnosticSeverity severity = SPIRV2CLCDSRemark;
          const char *prefix = "remark";
          switch (sev) {
          case spirv2clc::severity::error:
            severity = SPIRV2CLCDSError;
            prefix = "error";
            if (!errors.empty()) {
              errors += '\n';
            }
            errors += message;
            break;
          case spirv2clc::severity::warning:
            severity = SPIRV2CLCDSWarning;
            prefix = "warning";
            break;
          case spirv2clc::severity::remark:
            break;
          }
          if (Handler) {
            Handler(severity, message.c_str(), HandlerContext);
          } else {
            std::fprintf(stderr, "%s: %s\n", prefix, message.c_str());
          }
        });

    std::string source;
    if (translator.translate(binary, &source) != 0) {
      return fail_with(errors.empty() ? "failed to translate the module"
                                      : errors);
    }
    *OutSource = make_message(source);
    if (!*OutSource) {
      throw std::bad_alloc();
    }
    return 0;
  } catch (const std::bad_alloc &) {
    static const char message[] = "out of memory";
    return fail(message, sizeof(message) - 1);
  } catch (const std::exception &e) {
    return fail(e.what(), std::strlen(e.what()));
  } catch (...) {
    static const char message[] = "unknown internal error";
    return fail(message, sizeof(message) - 1);
  }
}

void SPIRV2CLCDisposeMessage(char *Message) { std::free(Message); }

} // extern "C"
