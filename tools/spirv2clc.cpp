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

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "spirv2clc.h"

std::unordered_map<std::string, spv_target_env> get_spirv_version_map() {
  return {
    {"1.0", SPV_ENV_UNIVERSAL_1_0},
    {"vulkan1.0", SPV_ENV_VULKAN_1_0},
    {"1.1", SPV_ENV_UNIVERSAL_1_1},
    {"opencl1.2", SPV_ENV_OPENCL_1_2},
    {"opencl-embedded1.2", SPV_ENV_OPENCL_EMBEDDED_1_2},
    {"opencl2.0", SPV_ENV_OPENCL_2_0},
    {"opencl-embedded2.0", SPV_ENV_OPENCL_EMBEDDED_2_0},
    {"opencl2.1", SPV_ENV_OPENCL_2_1},
    {"opencl-embedded2.1", SPV_ENV_OPENCL_EMBEDDED_2_1},
    {"opencl2.2", SPV_ENV_OPENCL_2_2},
    {"opencl-embedded2.2", SPV_ENV_OPENCL_EMBEDDED_2_2},
    {"opengl4.0", SPV_ENV_OPENGL_4_0},
    {"opengl4.1", SPV_ENV_OPENGL_4_1},
    {"opengl4.2", SPV_ENV_OPENGL_4_2},
    {"opengl4.3", SPV_ENV_OPENGL_4_3},
    {"opengl4.5", SPV_ENV_OPENGL_4_5},
    {"1.2", SPV_ENV_UNIVERSAL_1_2},
    {"1.3", SPV_ENV_UNIVERSAL_1_3},
    {"vulkan1.1", SPV_ENV_VULKAN_1_1},
    {"1.4", SPV_ENV_UNIVERSAL_1_4},
    {"vulkan1.1-spirv1.4", SPV_ENV_VULKAN_1_1_SPIRV_1_4},
    {"1.5", SPV_ENV_UNIVERSAL_1_5},
    {"vulkan1.2", SPV_ENV_VULKAN_1_2},
    {"1.6", SPV_ENV_UNIVERSAL_1_6},
    {"vulkan1.3", SPV_ENV_VULKAN_1_3}
  };
}

// OpenCL C language version to emit for, named like Clang's / clBuildProgram's
// -cl-std option, and encoded as in CL_TARGET_OPENCL_VERSION.
std::unordered_map<std::string, unsigned> get_cl_std_map() {
  return {
    {"CL1.2", 120}, {"cl1.2", 120},
    {"CL2.0", 200}, {"cl2.0", 200},
    {"CL3.0", 300}, {"cl3.0", 300},
  };
}

void fail_help(const char *prog) {
  std::cerr << "Usage: " << prog << " [ --asm ] [ --spirv-version=<version> ] [ --cl-std=<version> ] input.spv[asm]" << std::endl;
  std::cerr << "Available SPIR-V versions:" << std::endl;
  std::cerr << "  1.0, 1.1, universal1.2, 1.3, 1.4, 1.5, 1.6" << std::endl;
  std::cerr << "  opencl1.2, opencl2.0, opencl2.1, opencl2.2 (and embedded variants)" << std::endl;
  std::cerr << "  vulkan1.0, vulkan1.1, vulkan1.2, vulkan1.3" << std::endl;
  std::cerr << "  opengl4.0, opengl4.1, opengl4.2, opengl4.3, opengl4.5" << std::endl;
  std::cerr << "Available OpenCL C versions for --cl-std (default CL1.2):" << std::endl;
  std::cerr << "  CL1.2, CL2.0, CL3.0" << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  bool input_asm = false;
  spv_target_env target_env = SPV_ENV_OPENCL_1_2; // default
  unsigned opencl_c_version = 120; // default
  auto version_map = get_spirv_version_map();
  auto cl_std_map = get_cl_std_map();

  int arg = 1;

  if (argc < 2) {
    fail_help(argv[0]);
  }

  int num_options = 0;
  while (arg < argc) {
    if (!strcmp(argv[arg], "--asm")) {
      input_asm = true;
      num_options++;
    } else if (!strncmp(argv[arg], "--spirv-version=", 16)) {
      std::string version_str = argv[arg] + 16;
      auto it = version_map.find(version_str);
      if (it == version_map.end()) {
        std::cerr << "Unknown SPIR-V version '" << version_str << "'" << std::endl;
        fail_help(argv[0]);
      }
      target_env = it->second;
      num_options++;
    } else if (!strncmp(argv[arg], "--cl-std=", 9)) {
      std::string version_str = argv[arg] + 9;
      auto it = cl_std_map.find(version_str);
      if (it == cl_std_map.end()) {
        std::cerr << "Unknown OpenCL C version '" << version_str << "'" << std::endl;
        fail_help(argv[0]);
      }
      opencl_c_version = it->second;
      num_options++;
    } else if (!strncmp(argv[arg], "--", 2)) {
      std::cerr << "Unknown option '" << argv[arg] << "'" << std::endl;
      fail_help(argv[0]);
    }
    arg++;
  }

  if (argc < 2 + num_options) {
    fail_help(argv[0]);
  }

  const char *fname = argv[argc - 1];
  std::ifstream file(fname, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Could not open " << fname << std::endl;
    exit(EXIT_FAILURE);
  }

  spirv2clc::translator translator(target_env, opencl_c_version);
  std::string srcgen;
  int err;
  if (input_asm) {
    std::stringstream buffer;
    buffer << file.rdbuf();

    err = translator.translate(buffer.str(), &srcgen);
  } else {
    file.seekg(0, std::ios::end);
    uint32_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> binary(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(binary.data()), size);
    err = translator.translate(binary, &srcgen);
  }

  printf("%s\n", srcgen.c_str());

  if (err != 0) {
    std::cerr << "Failed to translate module." << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}
