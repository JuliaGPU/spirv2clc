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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libspirv2clc.h"
#include <spirv-tools/libspirv.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      failures++;                                                              \
    }                                                                          \
  } while (0)

static const char copy_kernel[] = "OpCapability Addresses\n"
                                  "OpCapability Kernel\n"
                                  "OpMemoryModel Physical64 OpenCL\n"
                                  "OpEntryPoint Kernel %k \"copy\"\n"
                                  "%void = OpTypeVoid\n"
                                  "%uint = OpTypeInt 32 0\n"
                                  "%pg = OpTypePointer CrossWorkgroup %uint\n"
                                  "%fnty = OpTypeFunction %void %pg %pg\n"
                                  "%k = OpFunction %void None %fnty\n"
                                  "%dst = OpFunctionParameter %pg\n"
                                  "%src = OpFunctionParameter %pg\n"
                                  "%entry = OpLabel\n"
                                  "%v = OpLoad %uint %src\n"
                                  "OpStore %dst %v\n"
                                  "OpReturn\n"
                                  "OpFunctionEnd\n";

/* Uses an ordered atomic, which needs OpenCL C 2.0. */
static const char atomic_kernel[] = "OpCapability Addresses\n"
                                    "OpCapability Kernel\n"
                                    "OpMemoryModel Physical64 OpenCL\n"
                                    "OpEntryPoint Kernel %k \"clear\"\n"
                                    "%void = OpTypeVoid\n"
                                    "%uint = OpTypeInt 32 0\n"
                                    "%pg = OpTypePointer CrossWorkgroup %uint\n"
                                    "%device = OpConstant %uint 1\n"
                                    "%release = OpConstant %uint 4\n"
                                    "%fnty = OpTypeFunction %void %pg\n"
                                    "%k = OpFunction %void None %fnty\n"
                                    "%flag = OpFunctionParameter %pg\n"
                                    "%entry = OpLabel\n"
                                    "OpAtomicFlagClear %flag %device %release\n"
                                    "OpReturn\n"
                                    "OpFunctionEnd\n";

/* Fails validation with a message that quotes the instruction. */
static const char invalid_kernel[] = "OpCapability Addresses\n"
                                     "OpCapability Kernel\n"
                                     "OpMemoryModel Physical64 OpenCL\n"
                                     "%u0 = OpTypeInt 0 0\n";

static spv_binary assemble(const char *text) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
  spv_binary binary = NULL;
  spv_diagnostic diagnostic = NULL;
  if (spvTextToBinary(context, text, strlen(text), &binary, &diagnostic) !=
      SPV_SUCCESS) {
    spvDiagnosticPrint(diagnostic);
    exit(EXIT_FAILURE);
  }
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);
  return binary;
}

typedef struct {
  int errors, others;
  char last_error[256];
} diagnostics;

static void collect(SPIRV2CLCDiagnosticSeverity severity, const char *message,
                    void *context) {
  diagnostics *diags = (diagnostics *)context;
  size_t length = strlen(message);
  CHECK(length > 0 && message[length - 1] != '\n');
  if (severity == SPIRV2CLCDSError) {
    diags->errors++;
    snprintf(diags->last_error, sizeof(diags->last_error), "%s", message);
  } else {
    diags->others++;
  }
}

/* Translates `size` bytes at `data`, checking the invariants of the result. */
static int translate(const void *data, size_t size,
                     const SPIRV2CLCTranslateOptions *options,
                     diagnostics *diags, char **source, char **message) {
  int status = SPIRV2CLCTranslate((const char *)data, size, options, collect,
                                  diags, source, message);
  CHECK((status == 0) == (*source != NULL));
  CHECK((status == 0) == (*message == NULL));
  return status;
}

int main(void) {
  spv_binary copy = assemble(copy_kernel);
  spv_binary atomic = assemble(atomic_kernel);
  size_t copy_size = copy->wordCount * sizeof(uint32_t);
  size_t atomic_size = atomic->wordCount * sizeof(uint32_t);
  char *source, *message;

  /* Success, with the default options. */
  {
    diagnostics diags = {0};
    CHECK(translate(copy->code, copy_size, NULL, &diags, &source, &message) ==
          0);
    CHECK(source && strstr(source, "void kernel copy("));
    CHECK(diags.errors == 0);
    SPIRV2CLCDisposeMessage(source);
  }

  /* An unsupported construct: the error is reported to the callback and
   * returned as the message. */
  {
    SPIRV2CLCTranslateOptions options = {1, 0, 120};
    diagnostics diags = {0};
    CHECK(translate(atomic->code, atomic_size, &options, &diags, &source,
                    &message) != 0);
    CHECK(diags.errors == 1);
    CHECK(strstr(diags.last_error, "ordered atomics require OpenCL C 2.0"));
    CHECK(message && strcmp(message, diags.last_error) == 0);
    SPIRV2CLCDisposeMessage(message);
  }

  /* ... which translates for OpenCL C 2.0. */
  {
    SPIRV2CLCTranslateOptions options = {1, 0, 200};
    diagnostics diags = {0};
    CHECK(translate(atomic->code, atomic_size, &options, &diags, &source,
                    &message) == 0);
    CHECK(source && strstr(source, "atomic_flag_clear_explicit"));
    SPIRV2CLCDisposeMessage(source);
  }

  /* Input that isn't SPIR-V, at an address that isn't word-aligned. */
  {
    char garbage[] = "this is not SPIR-V at all";
    diagnostics diags = {0};
    CHECK(translate(garbage + 1, 24, NULL, &diags, &source, &message) != 0);
    CHECK(message && strstr(message, "Invalid SPIR-V magic number"));
    SPIRV2CLCDisposeMessage(message);

    CHECK(translate(garbage, 3, NULL, &diags, &source, &message) != 0);
    CHECK(message && strstr(message, "not a multiple of 4 bytes"));
    SPIRV2CLCDisposeMessage(message);

    CHECK(translate(NULL, 0, NULL, &diags, &source, &message) != 0);
    CHECK(message != NULL);
    SPIRV2CLCDisposeMessage(message);
  }

  /* A module that doesn't validate. */
  {
    spv_binary invalid = assemble(invalid_kernel);
    diagnostics diags = {0};
    CHECK(translate(invalid->code, invalid->wordCount * sizeof(uint32_t), NULL,
                    &diags, &source, &message) != 0);
    CHECK(diags.errors == 1);
    CHECK(message && strstr(message, "Invalid number of bits (0)"));
    SPIRV2CLCDisposeMessage(message);
    spvBinaryDestroy(invalid);
  }

  /* Invalid options. */
  {
    SPIRV2CLCTranslateOptions spirv_version = {2, 0, 0};
    SPIRV2CLCTranslateOptions opencl_c_version = {0, 0, 210};
    diagnostics diags = {0};
    CHECK(translate(copy->code, copy_size, &spirv_version, &diags, &source,
                    &message) != 0);
    CHECK(message && strstr(message, "unsupported SPIR-V version 2.0"));
    SPIRV2CLCDisposeMessage(message);
    CHECK(translate(copy->code, copy_size, &opencl_c_version, &diags, &source,
                    &message) != 0);
    CHECK(message && strstr(message, "unsupported OpenCL C version 210"));
    SPIRV2CLCDisposeMessage(message);
  }

  /* Without a message pointer, a failure is still reported. */
  CHECK(SPIRV2CLCTranslate("abc", 3, NULL, collect, &(diagnostics){0}, &source,
                           NULL) != 0);
  CHECK(source == NULL);

  spvBinaryDestroy(copy);
  spvBinaryDestroy(atomic);

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
