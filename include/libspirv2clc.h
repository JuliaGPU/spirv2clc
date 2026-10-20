/* Copyright 2020-2022 The spirv2clc authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* The C API of spirv2clc, for using the translator in-process from languages
 * that can't call C++, as an alternative to running the `spirv2clc` tool.
 *
 * Conventions follow llvm-c: status as the return value (0 = success), the
 * result and an error message returned through out-pointers and freed with
 * SPIRV2CLCDisposeMessage, and diagnostics delivered through an optional
 * callback. Calls are independent of each other and may run concurrently. */

#ifndef LIBSPIRV2CLC_H
#define LIBSPIRV2CLC_H

#include <stddef.h>

#if defined(_WIN32)
#define SPIRV2CLC_API __declspec(dllexport)
#else
#define SPIRV2CLC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SPIRV2CLCDSError = 0,
  SPIRV2CLCDSWarning = 1,
  SPIRV2CLCDSRemark = 2
} SPIRV2CLCDiagnosticSeverity;

/* Receives every diagnostic the translator emits, one message per call,
 * without a trailing newline. NULL means: print them to stderr. Errors are
 * also collected into the failure message returned by SPIRV2CLCTranslate. */
typedef void (*SPIRV2CLCDiagnosticCallback)(
    SPIRV2CLCDiagnosticSeverity Severity, const char *Message, void *Context);

typedef struct {
  /* SPIR-V version whose rules the module is validated against, e.g. 1 and 4
   * (`--spirv-version=1.4`). 0/0 selects the OpenCL 1.2 environment, which is
   * also the tool's default. */
  unsigned SPIRVVersionMajor, SPIRVVersionMinor;
  /* OpenCL C version to emit for, encoded as in CL_TARGET_OPENCL_VERSION:
   * 120, 200 or 300 (`--cl-std=CL1.2` etc.). 0 selects 120. Constructs that
   * need a higher version are diagnosed as errors. */
  unsigned OpenCLCVersion;
} SPIRV2CLCTranslateOptions;

/* Translate a SPIR-V module (a binary of `Length` bytes at `SPIRV`) to OpenCL
 * C source. `Options` may be NULL to select the defaults. On success returns
 * 0 and sets `*OutSource` to the NUL-terminated source. On failure returns
 * nonzero, sets `*OutSource` to NULL and, if `OutMessage` is not NULL, sets
 * `*OutMessage` to a description of the error(s). Both strings are freed with
 * SPIRV2CLCDisposeMessage. */
SPIRV2CLC_API int SPIRV2CLCTranslate(const char *SPIRV, size_t Length,
                                     const SPIRV2CLCTranslateOptions *Options,
                                     SPIRV2CLCDiagnosticCallback Handler,
                                     void *HandlerContext, char **OutSource,
                                     char **OutMessage);

SPIRV2CLC_API void SPIRV2CLCDisposeMessage(char *Message);

#ifdef __cplusplus
}
#endif

#endif /* LIBSPIRV2CLC_H */
