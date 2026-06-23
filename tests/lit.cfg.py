# -*- Python -*-
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
#
# Main lit configuration for the spirv2clc regression tests. Concrete build
# paths are injected by the generated lit.site.cfg.py (see lit.site.cfg.py.in),
# which loads this file.

import os

import lit.formats

config.name = "spirv2clc"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".spt"]

# Test inputs live next to this file under cases/.
config.test_source_root = os.path.join(os.path.dirname(__file__), "cases")
config.test_exec_root = getattr(
    config, "spirv2clc_test_exec_root", config.test_source_root
)

# Tools used by the RUN lines, resolved by CMake and passed via the site config.
_tools = [
    ("%spirv2clc", getattr(config, "spirv2clc_tool", None)),
    ("%clang", getattr(config, "clang_executable", None)),
    ("%FileCheck", getattr(config, "filecheck_executable", None)),
]
for name, path in _tools:
    if not path or not os.path.exists(path):
        lit_config.fatal("tool for substitution {} not found: {!r}".format(name, path))
    config.substitutions.append((name, '"{}"'.format(path)))
