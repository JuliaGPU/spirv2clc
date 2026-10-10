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

#include "translator.h"
#include "spirv2clc.h"

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl_half.h"

#include "opt/build_module.h"
#include "opt/ir_context.h"
#include "spirv/unified1/OpenCL.std.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <set>

using namespace spvtools;
using namespace spvtools::opt;
using spvtools::opt::analysis::Type;

namespace {

std::ostream &operator<<(std::ostream &os, const spv::Op &op) {
  os << static_cast<int>(op);
  return os;
}

// Human-readable form of an OpenCL C version encoded as 120/200/300.
std::string opencl_c_version_str(unsigned v) {
  return std::to_string(v / 100) + "." + std::to_string((v % 100) / 10);
}

std::string rounding_mode(SpvFPRoundingMode mode) {
  switch (mode) {
  case SpvFPRoundingModeRTE:
    return "rte";
  case SpvFPRoundingModeRTZ:
    return "rtz";
  case SpvFPRoundingModeRTP:
    return "rtp";
  case SpvFPRoundingModeRTN:
    return "rtn";
  case SpvFPRoundingModeMax:
    break;
  }
  return "UNKNOWN ROUNDING MODE";
}

const spvtools::MessageConsumer spvtools_message_consumer =
    [](spv_message_level_t level, const char *, const spv_position_t &position,
       const char *message) {
      const char *levelstr = "UNKNOWN";
      switch (level) {
      case SPV_MSG_FATAL:
        levelstr = "FATAL";
        break;
      case SPV_MSG_INTERNAL_ERROR:
        levelstr = "INTERNAL ERROR";
        break;
      case SPV_MSG_ERROR:
        levelstr = "ERROR";
        break;
      case SPV_MSG_WARNING:
        levelstr = "WARNING";
        break;
      case SPV_MSG_INFO:
        levelstr = "INFO";
        break;
      case SPV_MSG_DEBUG:
        levelstr = "DEBUG";
        break;
      }
      printf("spvtools says '%s' (%s) at position %zu", message, levelstr,
             position.index);
    };

} // namespace

namespace spirv2clc {

translator::translator(spv_target_env env, unsigned opencl_c_version)
    : m_impl(std::make_unique<translator_impl>(env, opencl_c_version)) {}

translator::~translator() = default;
translator::translator(translator &&) = default;
translator &translator::operator=(translator &&) = default;

int translator::translate(const std::string &assembly, std::string *srcout) {
  return m_impl->translate(assembly, srcout);
}

int translator::translate(const std::vector<uint32_t> &binary,
                          std::string *srcout) {
  return m_impl->translate(binary, srcout);
}

translator_impl::translator_impl(spv_target_env env, unsigned opencl_c_version)
    : m_target_env(env), m_opencl_c_version(opencl_c_version) {}

translator_impl::~translator_impl() = default;

std::string translator_impl::note_unsupported(const std::string &what) const {
  std::cerr << "UNIMPLEMENTED " << what << std::endl;
  m_translation_failed = true;
  return "UNIMPLEMENTED";
}

// Unity sources retain their dependency order.
// clang-format off
#include "utils.cpp"
#include "types.cpp"
#include "groups.cpp"
#include "instns.cpp"
#include "instns_ext.cpp"
// clang-format on

void translator_impl::enable_extension(const char *extension) {
  if (m_enabled_extensions.insert(extension).second) {
    m_out.extensions << "#pragma OPENCL EXTENSION " << extension
                     << " : enable\n";
  }
}

bool translator_impl::translate_capabilities() {
  // cl_khr_subgroups is an OpenCL C 2.0 extension (promoted to the
  // __opencl_c_subgroups optional feature in 3.0). The subgroup capabilities
  // require it; refuse below 2.0 rather than emit an invalid pragma.
  auto require_subgroups = [&](const char *what) {
    if (m_opencl_c_version < 200) {
      std::cerr << "UNIMPLEMENTED: " << what << " require OpenCL C 2.0 (targeting "
                << opencl_c_version_str(m_opencl_c_version) << ").\n";
      return false;
    }
    enable_extension("cl_khr_subgroups");
    return true;
  };
  for (auto &inst : m_ir->capabilities()) {
    assert(inst.opcode() == spv::Op::OpCapability);
    auto cap = inst.GetSingleWordOperand(0);
    switch (cap) {
    case SpvCapabilityAddresses:
    case SpvCapabilityLinkage:
    case SpvCapabilityKernel:
    case SpvCapabilityInt8:
    case SpvCapabilityInt16:
    case SpvCapabilityInt64:
    case SpvCapabilityVector16:
    case SpvCapabilityImageBasic:
    case SpvCapabilityLiteralSampler:
      break;
    case SpvCapabilityFloat16Buffer:
      // Float16Buffer only promises half-typed *buffers*, but we materialize
      // half as a value type (return values, locals, parameters, constants),
      // all of which require cl_khr_fp16 in OpenCL C. Enable it here too; the
      // pragma is harmless when only half buffers are used.
      enable_extension("cl_khr_fp16");
      break;
    case SpvCapabilitySubgroupDispatch:
      // Gates the subgroup query builtins (get_sub_group_id, ...). Some devices
      // (e.g. pocl) declare it spuriously, so below 2.0 accept it silently and
      // let any actual subgroup use error at its use site rather than fail here.
      if (m_opencl_c_version >= 200) {
        enable_extension("cl_khr_subgroups");
      }
      break;
    case SpvCapabilityFloat16:
      enable_extension("cl_khr_fp16");
      break;
    case SpvCapabilityFloat64:
      enable_extension("cl_khr_fp64");
      break;
    case SpvCapabilityInt64Atomics:
      // 64-bit atomics. The atomic_*() builtins already cover the long/ulong
      // overloads; just enable the extensions (available since OpenCL C 1.1).
      enable_extension("cl_khr_int64_base_atomics");
      enable_extension("cl_khr_int64_extended_atomics");
      break;
    case SpvCapabilityGenericPointer:
      // The generic address space is core in OpenCL C 2.0; see the matching
      // mapping of SpvStorageClassGeneric in src_pointer_type (types.cpp).
      if (m_opencl_c_version < 200) {
        std::cerr << "UNIMPLEMENTED: generic address space requires OpenCL C "
                     "2.0 (targeting "
                  << opencl_c_version_str(m_opencl_c_version) << ").\n";
        return false;
      }
      break;
    case SpvCapabilityImageReadWrite:
      // read_write images are core in OpenCL C 2.0 (an optional feature in 3.0,
      // __opencl_c_read_write_images); see also the ReadWrite qualifier in
      // translate_type (types.cpp).
      if (m_opencl_c_version < 200) {
        std::cerr << "UNIMPLEMENTED: read_write images require OpenCL C 2.0 "
                     "(targeting "
                  << opencl_c_version_str(m_opencl_c_version) << ").\n";
        return false;
      }
      break;
    case SpvCapabilityGroups:
      // Groups also covers work-group collectives, which do not need subgroups.
      if (m_opencl_c_version < 200) {
        std::cerr << "UNIMPLEMENTED: group operations require OpenCL C 2.0\n";
        return false;
      }
      break;
    case SpvCapabilityGroupNonUniform:
      if (!require_subgroups("group operations")) {
        return false;
      }
      break;
    case SpvCapabilityGroupNonUniformVote:
    case SpvCapabilityGroupNonUniformArithmetic:
    case SpvCapabilityGroupNonUniformBallot:
    case SpvCapabilityGroupNonUniformShuffle:
    case SpvCapabilityGroupNonUniformShuffleRelative:
    case SpvCapabilityGroupNonUniformClustered:
    case SpvCapabilityGroupNonUniformRotateKHR: {
      if (!require_subgroups("subgroup operations")) {
        return false;
      }
      const char *ext = nullptr;
      switch (cap) {
      case SpvCapabilityGroupNonUniformVote:
        ext = "cl_khr_subgroup_non_uniform_vote";
        break;
      case SpvCapabilityGroupNonUniformArithmetic:
        ext = "cl_khr_subgroup_non_uniform_arithmetic";
        break;
      case SpvCapabilityGroupNonUniformBallot:
        ext = "cl_khr_subgroup_ballot";
        break;
      case SpvCapabilityGroupNonUniformShuffle:
        ext = "cl_khr_subgroup_shuffle";
        break;
      case SpvCapabilityGroupNonUniformShuffleRelative:
        ext = "cl_khr_subgroup_shuffle_relative";
        break;
      case SpvCapabilityGroupNonUniformRotateKHR:
        ext = "cl_khr_subgroup_rotate";
        break;
      case SpvCapabilityGroupNonUniformClustered:
        ext = "cl_khr_subgroup_clustered_reduce";
        break;
      }
      enable_extension(ext);
      break;
    }
    case SpvCapabilityAtomicFloat32AddEXT:
    case SpvCapabilityAtomicFloat64AddEXT:
    case SpvCapabilityAtomicFloat32MinMaxEXT:
    case SpvCapabilityAtomicFloat64MinMaxEXT:
      // Floating-point atomic add, min and max (atomic_fetch_*_explicit on
      // atomic_float/atomic_double).
      enable_extension("cl_ext_float_atomics");
      break;
    default:
      std::cerr << "UNIMPLEMENTED capability " << cap << ".\n";
      return false;
    }
  }
  return true;
}

bool translator_impl::translate_extensions() const {
  // SPIR-V extensions we can honor; they need no emission of their own (the
  // capabilities/instructions they enable are handled elsewhere).
  static const std::unordered_set<std::string> handled = {
      "SPV_KHR_no_integer_wrap_decoration",
      "SPV_KHR_subgroup_rotate",
      "SPV_EXT_shader_atomic_float_add",
      "SPV_EXT_shader_atomic_float_min_max",
  };
  for (auto &inst : m_ir->module()->extensions()) {
    assert(inst.opcode() == spv::Op::OpExtension);
    auto &op_ext = inst.GetOperand(0);
    auto ext = op_ext.AsString();
    if (!handled.count(ext)) {
      std::cerr << "UNIMPLEMENTED extension " << ext << ".\n";
      return false;
    }
  }
  return true;
}

bool translator_impl::translate_extended_instructions_imports() const {
  for (auto &inst : m_ir->ext_inst_imports()) {
    assert(inst.opcode() == spv::Op::OpExtInstImport);
    auto name = inst.GetOperand(1).AsString();
    if (name != "OpenCL.std") {
      std::cerr << "UNIMPLEMENTED extended instruction set.\n";
      return false;
    }
  }
  return true;
}

bool translator_impl::translate_memory_model() {
  auto inst = m_ir->module()->GetMemoryModel();
  auto add = inst->GetSingleWordOperand(0);
  auto mem = inst->GetSingleWordOperand(1);

  if (add == SpvAddressingModelPhysical32) {
    m_pointer_width = 32;
  } else if (add == SpvAddressingModelPhysical64) {
    m_pointer_width = 64;
  } else {
    return false;
  }
  if (mem != SpvMemoryModelOpenCL) {
    return false;
  }

  return true;
}

bool translator_impl::translate_entry_points() {
  for (auto &ep : m_ir->module()->entry_points()) {
    auto model = ep.GetSingleWordOperand(0);
    auto func = ep.GetSingleWordOperand(1);
    auto &op_name = ep.GetOperand(2);

    if (model != SpvExecutionModelKernel) {
      return false;
    }

    m_entry_points[func] = op_name.AsString();
  }

  return true;
}

bool translator_impl::translate_execution_modes() {
  for (auto &em : m_ir->module()->execution_modes()) {
    auto ep = em.GetSingleWordOperand(0);
    auto mode = em.GetSingleWordOperand(1);
    switch (mode) {
    case SpvExecutionModeLocalSize: {
      auto x = em.GetSingleWordOperand(2);
      auto y = em.GetSingleWordOperand(3);
      auto z = em.GetSingleWordOperand(4);
      m_entry_points_local_size[ep] = std::make_tuple(x, y, z);
      break;
    }
    case SpvExecutionModeContractionOff:
      m_entry_points_contraction_off.insert(ep);
      break;
    // Required sub-group size -> intel_reqd_sub_group_size kernel attribute
    // (see emit_function_signature).
    case SpvExecutionModeSubgroupSize:
      m_entry_points_subgroup_size[ep] = em.GetSingleWordOperand(2);
      break;
    // Optimization hints with no semantic effect (vec_type_hint,
    // work_group_size_hint). Accept and drop them rather than abort the module.
    case SpvExecutionModeVecTypeHint:
    case SpvExecutionModeLocalSizeHint:
      break;
    default:
      std::cerr << "UNIMPLEMENTED execution mode " << mode << ".\n";
      return false;
    }
  }
  return true;
}

bool translator_impl::translate_debug_instructions() {
  // Debug 1
  for (auto &inst : m_ir->module()->debugs1()) {
    auto opcode = inst.opcode();
    switch (opcode) {
    case spv::Op::OpSource:
    case spv::Op::OpString:
      break;
    default:
      std::cerr << "UNIMPLEMENTED debug instructions in 7a " << opcode
                << std::endl;
      return false;
    }
  }

  // Debug 2
  for (auto &inst : m_ir->module()->debugs2()) {
    auto opcode = inst.opcode();
    switch (opcode) {
    case spv::Op::OpName: {
      auto id = inst.GetSingleWordOperand(0);
      m_debug_names[id] = inst.GetOperand(1).AsString();
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED debug instructions " << opcode << ".\n";
      return false;
    }
  }

  // Debug 3
  for (auto &inst : m_ir->module()->debugs3()) {
    std::cerr << "UNIMPLEMENTED debug instruction " << inst.opcode()
              << " in 7c.\n";
    return false;
  }

  return true;
}

bool translator_impl::translate_annotations() {
  for (auto &inst : m_ir->module()->annotations()) {
    auto opcode = inst.opcode();
    switch (opcode) {
    case spv::Op::OpDecorate: {
      auto target = inst.GetSingleWordOperand(0);
      auto decoration = inst.GetSingleWordOperand(1);
      switch (decoration) {
      case SpvDecorationFuncParamAttr: {
        auto param_attr = inst.GetSingleWordOperand(2);
        switch (param_attr) {
        case SpvFunctionParameterAttributeZext:
        case SpvFunctionParameterAttributeSext:
          // Integer extension hints for the ABI; nothing to emit.
          break;
        case SpvFunctionParameterAttributeNoAlias:
          // Aliasing hint (like `restrict`); safe to ignore.
        case SpvFunctionParameterAttributeNoCapture:
          break;
        case SpvFunctionParameterAttributeSret:
          // Struct-return pointer: informational in LLVM-derived SPIR-V (the
          // parameter is an ordinary pointer and the body stores through it
          // explicitly), so no special handling is required.
          break;
        case SpvFunctionParameterAttributeNoWrite:
          m_nowrite_params.insert(target);
          break;
        case SpvFunctionParameterAttributeByVal:
          m_byval_params.insert(target);
          break;
        default:
          std::cerr << "UNIMPLEMENTED FuncParamAttr " << param_attr
                    << std::endl;
          return false;
        }
        break;
      }
      case SpvDecorationBuiltIn: {
        auto builtin = inst.GetSingleWordOperand(2);
        switch (builtin) {
        case SpvBuiltInGlobalInvocationId:
        case SpvBuiltInGlobalSize:
        case SpvBuiltInGlobalOffset:
        case SpvBuiltInWorkgroupId:
        case SpvBuiltInWorkgroupSize:
        case SpvBuiltInLocalInvocationId:
        case SpvBuiltInNumWorkgroups:
        case SpvBuiltInWorkDim:
          m_builtin_variables[target] = static_cast<SpvBuiltIn>(builtin);
          break;
        case SpvBuiltInSubgroupEqMask:
        case SpvBuiltInSubgroupGeMask:
        case SpvBuiltInSubgroupGtMask:
        case SpvBuiltInSubgroupLeMask:
        case SpvBuiltInSubgroupLtMask:
        case SpvBuiltInSubgroupSize:
        case SpvBuiltInSubgroupMaxSize:
        case SpvBuiltInNumSubgroups:
        case SpvBuiltInNumEnqueuedSubgroups:
        case SpvBuiltInSubgroupId:
        case SpvBuiltInSubgroupLocalInvocationId:
          // Mapped to the get_sub_group_*() builtins, which need
          // cl_khr_subgroups (OpenCL C 2.0+), including when no dispatch
          // capability is declared.
          if (m_opencl_c_version < 200) {
            std::cerr << "UNIMPLEMENTED: subgroup builtins require OpenCL C 2.0 "
                         "(targeting "
                      << opencl_c_version_str(m_opencl_c_version) << ").\n";
            return false;
          }
          enable_extension("cl_khr_subgroups");
          m_builtin_variables[target] = static_cast<SpvBuiltIn>(builtin);
          break;
        default:
          std::cerr << "UNIMPLEMENTED builtin " << builtin << std::endl;
          return false;
        }
        break;
      }
      case SpvDecorationConstant:
      case SpvDecorationAliased:
        break;
      case SpvDecorationRestrict:
        m_restricts.insert(target);
        break;
      case SpvDecorationVolatile:
        m_volatiles.insert(target);
        break;
      case SpvDecorationCoherent: // TODO anything to do?
        break;
      case SpvDecorationCPacked:
        m_packed.insert(target);
        break;
      case SpvDecorationNonReadable:
      case SpvDecorationNonWritable: // TODO const?
        break;
      case SpvDecorationAlignment: {
        auto align = inst.GetSingleWordOperand(2);
        m_alignments[target] = align;
        break;
      }
      case SpvDecorationLinkageAttributes: {
        auto name = inst.GetOperand(2).AsString();
        auto type = inst.GetSingleWordOperand(3);
        if (type == SpvLinkageTypeExport) {
          m_exports[target] = name;
        }
        if (type == SpvLinkageTypeImport) {
          m_imports[target] = name;
        }
        break;
      }
      case SpvDecorationFPFastMathMode:
        // Ignore for now as that's always correct
        // TODO add a relaxed mode where the whole program is built with fast
        // math
        break;
      case SpvDecorationFPRoundingMode: {
        auto mode = inst.GetSingleWordOperand(2);
        m_rounding_mode_decorations[target] =
            static_cast<SpvFPRoundingMode>(mode);
        break;
      }
      case SpvDecorationSaturatedConversion:
        m_saturated_conversions.insert(target);
        break;
      case SpvDecorationNoSignedWrap:
      case SpvDecorationNoUnsignedWrap:
        break;
      case SpvDecorationMaxByteOffset:
      case SpvDecorationMaxByteOffsetId:
        // Pointer access-range hints; nothing to emit.
        break;
      default:
        std::cerr << "UNIMPLEMENTED decoration " << decoration << std::endl;
        return false;
      }
      break;
    }
    case spv::Op::OpDecorationGroup:
      break;
    case spv::Op::OpGroupDecorate: {
      auto group = inst.GetSingleWordOperand(0);
      bool restrict = m_restricts.count(group) != 0;
      bool hasvolatile = m_volatiles.count(group) != 0;
      bool packed = m_packed.count(group) != 0;
      bool nowrite = m_nowrite_params.count(group) != 0;
      bool byval = m_byval_params.count(group) != 0;
      bool saturated_conversion = m_saturated_conversions.count(group) != 0;
      bool has_rounding_mode = m_rounding_mode_decorations.count(group) != 0;
      SpvFPRoundingMode rounding_mode;
      if (has_rounding_mode) {
        rounding_mode = m_rounding_mode_decorations.at(group);
      }
      bool has_alignment = m_alignments.count(group) != 0;
      uint32_t alignment;
      if (has_alignment) {
        alignment = m_alignments.at(group);
      }
      for (unsigned i = 1; i < inst.NumOperands(); i++) {
        auto target = inst.GetSingleWordOperand(i);
        if (restrict) {
          m_restricts.insert(target);
        }
        if (hasvolatile) {
          m_volatiles.insert(target);
        }
        if (packed) {
          m_packed.insert(target);
        }
        if (nowrite) {
          m_nowrite_params.insert(target);
        }
        if (byval) {
          m_byval_params.insert(target);
        }
        if (saturated_conversion) {
          m_saturated_conversions.insert(target);
        }
        if (has_rounding_mode) {
          m_rounding_mode_decorations[target] = rounding_mode;
        }
        if (has_alignment) {
          m_alignments[target] = alignment;
        }
      }
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED annotation instruction " << opcode
                << std::endl;
      return false;
    }
  }
  return true;
}

void translator_impl::assign_names() {
  // Names fixed by the module's interface first, so nothing else can take
  // them. Entry points are named by their OpEntryPoint.
  auto fix = [this](uint32_t id, const std::string &name) {
    if (!m_names.count(id)) {
      m_name_allocator.reserve(name);
      m_names[id] = name;
    }
  };
  for (auto &ep : m_entry_points) {
    fix(ep.first, ep.second);
  }
  for (auto &exp : m_exports) {
    fix(exp.first, exp.second);
  }
  for (auto &imp : m_imports) {
    fix(imp.first, imp.second);
  }
  // Then the OpNames, before the made-up "v<id>" names so that those yield on
  // a clash. Both in module order, for stable output.
  for (bool named : {true, false}) {
    m_ir->module()->ForEachInst([this, named](const Instruction *inst) {
      auto id = inst->result_id();
      if (id == 0 || m_names.count(id)) {
        return;
      }
      auto it = m_debug_names.find(id);
      bool has_name = it != m_debug_names.end() && !it->second.empty();
      if (has_name == named) {
        m_names[id] = m_name_allocator.allocate(
            named ? it->second : "v" + std::to_string(id));
      }
    });
  }
}

const std::string &translator_impl::derived_name(uint32_t id,
                                                 const std::string &suffix) {
  auto key = std::make_pair(id, suffix);
  auto it = m_derived_names.find(key);
  if (it == m_derived_names.end()) {
    it = m_derived_names
             .emplace(key, m_name_allocator.allocate(name_of(id) + suffix))
             .first;
  }
  return it->second;
}

void translator_impl::compute_workgroup_params() {
  auto defuse = m_ir->get_def_use_mgr();
  for (auto &func : *m_ir->module()) {
    auto fid = func.DefInst().result_id();
    // Entry points declare the Workgroup storage themselves (see
    // translate_function), so they take no extra parameters. Imported functions
    // are external declarations we don't define or call with our own storage.
    if (m_entry_points.count(fid) != 0 || m_imports.count(fid) != 0) {
      continue;
    }

    // Collect every module-scope Workgroup variable referenced anywhere in this
    // function's call tree (itself included). Sorted (std::set) so the parameter
    // order is identical in the prototype, the definition and at the call sites.
    std::set<uint32_t> used;
    IRContext::ProcessFunction collect =
        [&used, defuse](Function *f) -> bool {
      for (auto &bb : *f) {
        for (auto &inst : bb) {
          for (auto &op : inst) {
            if (!spvIsIdType(op.type)) {
              continue;
            }
            auto def = defuse->GetDef(op.AsId());
            if (def && def->opcode() == spv::Op::OpVariable &&
                def->GetSingleWordOperand(2) == SpvStorageClassWorkgroup) {
              used.insert(op.AsId());
            }
          }
        }
      }
      return false;
    };
    std::queue<uint32_t> roots;
    roots.push(fid);
    m_ir->ProcessCallTreeFromRoots(collect, &roots);

    if (!used.empty()) {
      m_function_workgroup_params[fid] =
          std::vector<uint32_t>(used.begin(), used.end());
    }
  }
}

void translator_impl::emit_function_signature(Function &func,
                                              bool is_prototype) {
  auto &dinst = func.DefInst();
  auto rtype = dinst.type_id();
  auto result = dinst.result_id();
  auto control = dinst.GetSingleWordOperand(2);

  bool entrypoint = m_entry_points.count(result) != 0;
  bool is_external = m_imports.count(result) != 0;
  auto &os = is_prototype ? m_out.prototypes : m_out.functions;

  if (is_prototype) {
    // For prototypes, only emit static non-entry, non-external functions
    if (entrypoint || is_external) {
      return;
    }
    os << "static ";
  } else {
    // For definitions, emit full declaration logic
    if (is_external) {
      os << "extern ";
    } else if ((m_exports.count(result) == 0) && !entrypoint) {
      os << "static ";
    }
  }

  if (control & SpvFunctionControlInlineMask) {
    os << "inline ";
  }

  os << src_type(rtype) + " ";
  if (entrypoint && !is_prototype) {
    os << "kernel ";
    if (m_entry_points_local_size.count(result)) {
      auto &req = m_entry_points_local_size.at(result);
      os << "__attribute((reqd_work_group_size(";
      os << std::get<0>(req) << "," << std::get<1>(req) << ","
         << std::get<2>(req);
      os << "))) ";
    }
    if (m_entry_points_subgroup_size.count(result)) {
      os << "__attribute((intel_reqd_sub_group_size("
         << m_entry_points_subgroup_size.at(result) << "))) ";
    }
    os << m_entry_points.at(result);
  } else {
    os << name_of(result);
  }
  os << "(";
  std::string sep = "";
  func.ForEachParam([this, &sep, &os](const Instruction *inst) {
    auto type = inst->type_id();
    auto result = inst->result_id();
    os << sep;
    if (m_nowrite_params.count(result)) {
      os << "const ";
    }

    if (m_byval_params.count(result)) {
      // Pass byval parameters by value instead of by pointer
      auto param_type = type_for(type);
      assert(param_type->kind() == Type::Kind::kPointer);
      auto ptr_type = param_type->AsPointer();
      auto pointee_type = ptr_type->pointee_type();
      auto pointee_type_id = type_id_for(pointee_type);
      os << src_type(pointee_type_id) << " " << derived_name(result, "_value");
    } else {
      os << src_type_memory_object_declaration(type, result);
    }
    sep = ", ";
  });

  // Append local-pointer parameters for the module-scope Workgroup variables
  // this non-entry function references (see compute_workgroup_params). The entry
  // kernel owns the storage and passes these down at each call site.
  if (!entrypoint) {
    auto it = m_function_workgroup_params.find(result);
    if (it != m_function_workgroup_params.end()) {
      auto defuse = m_ir->get_def_use_mgr();
      for (auto wgvar : it->second) {
        os << sep;
        os << src_type(defuse->GetDef(wgvar)->type_id()) << " "
           << name_of(wgvar);
        sep = ", ";
      }
    }
  }

  os << ")";
  if (is_prototype) {
    os << ";" << std::endl;
  }
}

bool translator_impl::translate_function(Function &func) {
  auto &dinst = func.DefInst();
  auto result = dinst.result_id();

  bool entrypoint = m_entry_points.count(result) != 0;
  auto &os = m_out.functions;

  if (m_entry_points_contraction_off.count(result)) {
    os << "#pragma OPENCL FP_CONTRACT OFF" << std::endl;
  }

  emit_function_signature(func, false);

  // Imported functions are only declared.
  if (m_imports.count(result)) {
    os << ";" << std::endl;
    return true;
  }

  function_builder fb;

  // Declare variables in the local address space used by each kernel at the
  // beginning of the kernel function. If the kernel's call tree references
  // a Workgroup variable, paste the declaration we have prepared as part of
  // translating global variables.
  if (entrypoint) {
    std::set<uint32_t> used_globals_in_local_as;
    IRContext::ProcessFunction process_fn =
        [this, &used_globals_in_local_as](Function *func) -> bool {
      for (auto &bb : *func) {
        for (auto &inst : bb) {
          for (auto &op : inst) {
            if (spvIsIdType(op.type)) {
              auto used_inst_id = op.AsId();
              auto defuse = m_ir->get_def_use_mgr();
              auto used_inst = defuse->GetDef(used_inst_id);
              if (used_inst->opcode() == spv::Op::OpVariable) {
                if (used_inst->GetSingleWordOperand(2) ==
                    SpvStorageClassWorkgroup) {
                  used_globals_in_local_as.insert(used_inst_id);
                }
              }
            }
          }
        }
      }
      return false;
    };
    std::queue<uint32_t> roots;
    roots.push(result);
    m_ir->ProcessCallTreeFromRoots(process_fn, &roots);

    for (auto lvarid : used_globals_in_local_as) {
      fb.declare_upfront(m_local_variable_decls.at(lvarid));
    }
  }

  // Built-in variables aren't declared: loads from them are bound to queries
  // (see OpLoad). A function that uses a built-in's pointer some other way,
  // e.g. bitcast and offset to load one component, gets a private copy. That
  // includes uses through module-scope OpSpecConstantOps, which are bound to
  // expressions naming their operands.
  std::set<uint32_t> used_builtins;
  std::function<void(uint32_t)> find_builtins = [&](uint32_t id) {
    if (m_builtin_variables.count(id)) {
      used_builtins.insert(id);
      return;
    }
    auto def = m_ir->get_def_use_mgr()->GetDef(id);
    if (def && def->opcode() == spv::Op::OpSpecConstantOp) {
      def->ForEachInId([&](const uint32_t *op) { find_builtins(*op); });
    }
  };
  for (auto &bb : func) {
    for (auto &inst : bb) {
      if (inst.opcode() == spv::Op::OpLoad &&
          m_builtin_variables.count(inst.GetSingleWordOperand(2))) {
        continue;
      }
      inst.ForEachInId([&](const uint32_t *id) { find_builtins(*id); });
    }
  }
  for (auto var : used_builtins) {
    auto tyid = type_id_for(type_for_val(var)->AsPointer()->pointee_type());
    auto &storagename = derived_name(var, "_storage");
    fb.declare_upfront(
        src_var_decl(tyid, storagename) + " = " +
        c::print(builtin_value(m_builtin_variables.at(var), tyid)));
    fb.declare_upfront(src_var_decl(var) + " = " +
                       c::print(c::address_of(c::name(storagename))));
  }

  // ByVal parameters are passed by value; the body expects a pointer to it.
  func.ForEachParam([this, &fb](const Instruction *inst) {
    auto result = inst->result_id();
    if (m_byval_params.count(result)) {
      fb.declare_upfront(
          src_var_decl(result) + " = " +
          c::print(c::address_of(c::name(derived_name(result, "_value")))));
    }
  });

  // Lower OpPhi out of SSA in two phases. Each predecessor stages the incoming
  // value in a per-phi temporary before its terminator, and the phi block
  // commits the temporaries to the phi variables on entry. Writing the phi
  // variables directly in the predecessor would be wrong twice over: the write
  // also happens when the branch leaves through another edge (the loop exit
  // then sees the next iteration's value), and sequential writes break when
  // one phi feeds another of the same block (a swap reads the clobbered value).
  // Block -> (phi, incoming value) pairs to stage at its end.
  std::unordered_map<const BasicBlock *,
                     std::vector<std::pair<uint32_t, uint32_t>>>
      phi_incoming;
  for (auto &bb : func) {
    for (auto &inst : bb) {
      if (inst.opcode() != spv::Op::OpPhi) {
        continue;
      }
      auto phi = inst.result_id();
      fb.declare_upfront(src_var_decl(phi));
      fb.declare_upfront(
          src_var_decl(type_id_for(phi), derived_name(phi, "_phi")));
      for (unsigned i = 2; i < inst.NumOperands(); i += 2) {
        auto incoming = inst.GetSingleWordOperand(i);
        auto parent = func.FindBlock(inst.GetSingleWordOperand(i + 1));
        phi_incoming[&*parent].emplace_back(phi, incoming);
      }
    }
  }

  bool error = false;
  for (auto &bb : func) {
    fb.label(name_of(bb.id()));
    // Translate all instructions except the terminator. The phis lead the
    // block; each commits the value staged by the predecessor we came from.
    for (auto &inst : bb) {
      if (&inst == bb.terminator()) {
        break;
      }
      if (inst.opcode() == spv::Op::OpPhi) {
        auto phi = inst.result_id();
        fb.assign(c::name(name_of(phi)), c::name(derived_name(phi, "_phi")));
        continue;
      }
      if (!translate_instruction(inst, fb)) {
        error = true;
      }
    }
    // Stage the incoming values of the successors' phis. Staging for every
    // successor, not just the one taken, is harmless: a temporary is only read
    // on entry to its phi's block.
    auto it = phi_incoming.find(&bb);
    if (it != phi_incoming.end()) {
      for (auto &phi_value : it->second) {
        fb.assign(c::name(derived_name(phi_value.first, "_phi")),
                  value(phi_value.second));
      }
    }
    if (!translate_instruction(*bb.ctail(), fb)) {
      error = true;
    }
  }

  os << "{" << std::endl << fb.render() << "}\n";

  if (m_entry_points_contraction_off.count(result)) {
    os << "#pragma OPENCL FP_CONTRACT ON" << std::endl;
  }

  return !error;
}

int translator_impl::translate() {

  reset();

  // 1. Capabilities
  if (!translate_capabilities()) {
    return 1;
  }

  // 2. Extensions
  if (!translate_extensions()) {
    return 1;
  }

  // 3. Extended instructions imports
  if (!translate_extended_instructions_imports()) {
    return 1;
  }

  // 4. Memory model
  if (!translate_memory_model()) {
    return 1;
  }

  // 5. Entry point declarations
  if (!translate_entry_points()) {
    return 1;
  }

  // 6. Execution modes
  if (!translate_execution_modes()) {
    return 1;
  }

  // 7. Debug instructions
  if (!translate_debug_instructions()) {
    return 1;
  }

  // 8. Annotations
  if (!translate_annotations()) {
    return 1;
  }

  assign_names();

  // 9. Type declarations, constants and global variables
  if (!translate_types_values()) {
    return 1;
  }

  // Work out which non-entry functions reference module-scope Workgroup
  // variables, so their signatures and call sites can thread them through as
  // local-pointer parameters (OpenCL C has no program-scope local storage).
  compute_workgroup_params();

  // 10. Function declarations (prototypes)
  for (auto &func : *m_ir->module()) {
    emit_function_signature(func, true);
  }

  // 11. Function definitions
  for (auto &func : *m_ir->module()) {
    if (!translate_function(func)) {
      return 1;
    }
  }

  // Catch anything that emitted an UNIMPLEMENTED placeholder via note_unsupported
  // rather than failing on the spot (e.g. an unknown value or type referenced
  // while building an expression).
  if (m_translation_failed) {
    return 1;
  }

  return 0;
}

bool translator_impl::validate_module(
    const std::vector<uint32_t> &binary) const {
  spv_diagnostic diag;
  spv_context ctx = spvContextCreate(m_target_env);
  spv_result_t res =
      spvValidateBinary(ctx, binary.data(), binary.size(), &diag);
  spvContextDestroy(ctx);
  spvDiagnosticPrint(diag);
  spvDiagnosticDestroy(diag);
  if (res != SPV_SUCCESS) {
    return false;
  }
  // std::cout << "Module valid.\n";
  return true;
}

int translator_impl::translate(const std::string &assembly,
                               std::string *srcout) {

  m_ir = BuildModule(m_target_env, spvtools_message_consumer, assembly);

  std::vector<uint32_t> module_bin;
  m_ir->module()->ToBinary(&module_bin, false);
  if (!validate_module(module_bin)) {
    return 1;
  }

  int ret = translate();

  if (ret == 0) {
    *srcout = m_out.render();
  }

  return ret;
}

int translator_impl::translate(const std::vector<uint32_t> &binary,
                               std::string *srcout) {

  m_ir = BuildModule(m_target_env, spvtools_message_consumer,
                     binary.data(), binary.size());

  if (!validate_module(binary)) {
    return 1;
  }

  int ret = translate();

  if (ret == 0) {
    *srcout = m_out.render();
  }

  return ret;
}

} // namespace spirv2clc
