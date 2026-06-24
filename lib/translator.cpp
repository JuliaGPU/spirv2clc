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

#include "spirv2clc.h"

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl_half.h"

#include "opt/build_module.h"
#include "opt/ir_context.h"
#include "spirv/unified1/OpenCL.std.h"

#include <algorithm>
#include <cstring>
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
    : m_target_env(env), m_opencl_c_version(opencl_c_version) {}

translator::~translator() = default;
translator::translator(translator &&) = default;
translator &translator::operator=(translator &&) = default;

std::string translator::note_unsupported(const std::string &what) const {
  std::cerr << "UNIMPLEMENTED " << what << std::endl;
  m_translation_failed = true;
  return "UNIMPLEMENTED";
}

#include "utils.cpp"
#include "types.cpp"
#include "instns.cpp"
#include "instns_ext.cpp"

bool translator::translate_capabilities() {
  // Emit each enabling pragma at most once, even when several capabilities map
  // to the same extension (e.g. the various subgroup capabilities).
  std::unordered_set<std::string> enabled_extensions;
  auto enable_extension = [&](const char *ext) {
    if (enabled_extensions.insert(ext).second) {
      m_src << "#pragma OPENCL EXTENSION " << ext << " : enable" << std::endl;
    }
  };
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
    case SpvCapabilityGroups:
    case SpvCapabilityGroupNonUniform:
    case SpvCapabilityGroupNonUniformShuffle:
      // Work-group / sub-group collective builtins (shuffle, ...), all available
      // under cl_khr_subgroups.
      if (!require_subgroups("subgroups")) {
        return false;
      }
      break;
    case SpvCapabilityAtomicFloat32AddEXT:
    case SpvCapabilityAtomicFloat64AddEXT:
      // Floating-point atomic add (atomic_add on float/double).
      enable_extension("cl_ext_float_atomics");
      break;
    default:
      std::cerr << "UNIMPLEMENTED capability " << cap << ".\n";
      return false;
    }
  }
  return true;
}

bool translator::translate_extensions() const {
  // SPIR-V extensions we can honor; they need no emission of their own (the
  // capabilities/instructions they enable are handled elsewhere).
  static const std::unordered_set<std::string> handled = {
      "SPV_KHR_no_integer_wrap_decoration",
      "SPV_EXT_shader_atomic_float_add",
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

bool translator::translate_extended_instructions_imports() const {
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

bool translator::translate_memory_model() const {
  auto inst = m_ir->module()->GetMemoryModel();
  auto add = inst->GetSingleWordOperand(0);
  auto mem = inst->GetSingleWordOperand(1);

  if ((add != SpvAddressingModelPhysical32) &&
      (add != SpvAddressingModelPhysical64)) {
    return false;
  }
  if (mem != SpvMemoryModelOpenCL) {
    return false;
  }

  return true;
}

bool translator::translate_entry_points() {
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

bool translator::translate_execution_modes() {
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

bool translator::translate_debug_instructions() {
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
      auto name = inst.GetOperand(1).AsString();
      // Sanitize into a valid C identifier: map any character that isn't a
      // letter, digit or underscore to '_' (e.g. Julia names like
      // "a::CLDeviceArray"), and avoid a leading digit.
      auto valid_char = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
      };
      for (auto &ch : name) {
        if (!valid_char(ch)) {
          ch = '_';
        }
      }
      if (!name.empty() && name[0] >= '0' && name[0] <= '9') {
        name = "_" + name;
      }
      m_names[id] = name;
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED debug instructions " << opcode << ".\n";
      return false;
    }
  }

  // Fixup names to avoid identifiers invalid in OpenCL C
  for (auto &id_name : m_names) {
    auto &id = id_name.first;
    auto &name = id_name.second;
    if (gReservedIdentifiers.count(name)) {
      std::string newname = make_valid_identifier(name);
      m_names[id] = newname;
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

bool translator::translate_annotations() {
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
        case SpvBuiltInSubgroupSize:
        case SpvBuiltInSubgroupMaxSize:
        case SpvBuiltInNumSubgroups:
        case SpvBuiltInSubgroupId:
        case SpvBuiltInSubgroupLocalInvocationId:
          // Mapped to the get_sub_group_*() builtins, which need cl_khr_subgroups
          // (OpenCL C 2.0+; the pragma is emitted from the subgroup capability).
          if (m_opencl_c_version < 200) {
            std::cerr << "UNIMPLEMENTED: subgroup builtins require OpenCL C 2.0 "
                         "(targeting "
                      << opencl_c_version_str(m_opencl_c_version) << ").\n";
            return false;
          }
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

void translator::compute_workgroup_params() {
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
        [this, &used, defuse](Function *f) -> bool {
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

void translator::emit_function_signature(Function &func, bool is_prototype) {
  auto &dinst = func.DefInst();
  auto rtype = dinst.type_id();
  auto result = dinst.result_id();
  auto control = dinst.GetSingleWordOperand(2);

  bool entrypoint = m_entry_points.count(result) != 0;
  bool is_external = m_imports.count(result) != 0;

  if (is_prototype) {
    // For prototypes, only emit static non-entry, non-external functions
    if (entrypoint || is_external) {
      return;
    }
    m_src << "static ";
  } else {
    // For definitions, emit full declaration logic
    if (is_external) {
      m_src << "extern ";
    } else if ((m_exports.count(result) == 0) && !entrypoint) {
      m_src << "static ";
    }
  }

  if (control & SpvFunctionControlInlineMask) {
    m_src << "inline ";
  }

  m_src << src_type(rtype) + " ";
  if (entrypoint && !is_prototype) {
    m_src << "kernel ";
    if (m_entry_points_local_size.count(result)) {
      auto &req = m_entry_points_local_size.at(result);
      m_src << "__attribute((reqd_work_group_size(";
      m_src << std::get<0>(req) << "," << std::get<1>(req) << ","
            << std::get<2>(req);
      m_src << "))) ";
    }
    if (m_entry_points_subgroup_size.count(result)) {
      m_src << "__attribute((intel_reqd_sub_group_size("
            << m_entry_points_subgroup_size.at(result) << "))) ";
    }
    m_src << m_entry_points.at(result);
  } else {
    m_src << var_for(result);
  }
  m_src << "(";
  std::string sep = "";
  func.ForEachParam([this, &sep](const Instruction *inst) {
    auto type = inst->type_id();
    auto result = inst->result_id();
    m_src << sep;
    if (m_nowrite_params.count(result)) {
      m_src << "const ";
    }

    if (m_byval_params.count(result)) {
      // Pass byval parameters by value instead of by pointer
      auto param_type = type_for(type);
      assert(param_type->kind() == Type::Kind::kPointer);
      auto ptr_type = param_type->AsPointer();
      auto pointee_type = ptr_type->pointee_type();
      auto pointee_type_id = type_id_for(pointee_type);
      m_src << src_type(pointee_type_id) << " " << var_for(result) << "_value";
    } else {
      m_src << src_type_memory_object_declaration(type, result);
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
        m_src << sep;
        m_src << src_type(defuse->GetDef(wgvar)->type_id()) << " "
              << var_for(wgvar);
        sep = ", ";
      }
    }
  }

  m_src << ")";
  if (is_prototype) {
    m_src << ";" << std::endl;
  }
}

bool translator::translate_function(Function &func) {
  auto &dinst = func.DefInst();
  auto result = dinst.result_id();

  bool decl = false;
  bool entrypoint = m_entry_points.count(result) != 0;

  if (m_entry_points_contraction_off.count(result)) {
    m_src << "#pragma OPENCL FP_CONTRACT OFF" << std::endl;
  }

  // Check if this is just a declaration
  if (m_imports.count(result)) {
    decl = true;
  }

  // Emit function signature
  emit_function_signature(func, false);

  if (decl) {
    m_src << ";" << std::endl;
    return true;
  } else {
    m_src << "{" << std::endl;
  }

  // Declare variables in the local address space used by each kernel at the
  // beginning of the kernel function. If the kernel's call tree references
  // a Workgroup variable, paste the declaration we have prepared as part of
  // translating global variables.
  if (entrypoint) {
    std::unordered_set<uint32_t> used_globals_in_local_as;
    IRContext::ProcessFunction process_fn = [this, &used_globals_in_local_as](Function* func) -> bool {
      for (auto &bb : *func) {
        for (auto &inst : bb) {
          for (auto& op : inst) {
            if (spvIsIdType(op.type)) {
              auto used_inst_id = op.AsId();
              auto defuse = m_ir->get_def_use_mgr();
              auto used_inst = defuse->GetDef(used_inst_id);
              if (used_inst->opcode() == spv::Op::OpVariable) {
                if (used_inst->GetSingleWordOperand(2) == SpvStorageClassWorkgroup) {
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
      m_src << m_local_variable_decls.at(lvarid) << ";\n";
    }
  }

  // First collect information about OpPhi's
  for (auto &bb : func) {
    for (auto &inst : bb) {
      auto result = inst.result_id();
      if (inst.opcode() != spv::Op::OpPhi) {
        continue;
      }
      m_phi_vals[&func].push_back(result);

      for (unsigned i = 2; i < inst.NumOperands(); i += 2) {
        auto var = inst.GetSingleWordOperand(i);
        auto parent = inst.GetSingleWordOperand(i + 1);
        auto parentbb = func.FindBlock(parent);

        m_phi_assigns[&*parentbb].push_back(std::make_pair(result, var));
      }
    }
  }

  // Now translate
  bool error = false;

  // Add helper variables for byval arguments containing a pointer
  // (for compatibility with existing code)
  func.ForEachParam([this](const Instruction *inst) {
    auto result = inst->result_id();
    if (m_byval_params.count(result)) {
      m_src << "  " << src_type(inst->type_id()) << " " << var_for(result) << " = &" << var_for(result) << "_value;\n";
    }
  });

  if (m_phi_vals.count(&func)) {
    for (auto phival : m_phi_vals.at(&func)) {
      auto phitype = type_id_for(phival);
      m_src << "  " << src_type(phitype) << " " << var_for(phival) << ";\n";
    }
  }
  for (auto &bb : func) {
    m_src << var_for(bb.id()) + ":;" << std::endl;
    // Translate all instructions except the terminator
    for (auto &inst : bb) {
      if (&inst == bb.terminator()) {
        break;
      }
      std::string isrc;
      if (!translate_instruction(inst, isrc)) {
        error = true;
      }
      if (isrc != "") {
        m_src << "  " << isrc << ";\n";
      }
    }
    // Assign phi variables if this block can branch to other blocks with phi
    // refering to this block
    if (m_phi_assigns.count(&bb)) {
      for (auto &phival_var : m_phi_assigns.at(&bb)) {
        m_src << "  " << var_for(phival_var.first) << " = "
              << var_for(phival_var.second) << ";\n";
      }
    }

    // Translate the terminator
    std::string isrc;
    if (!translate_instruction(*bb.ctail(), isrc)) {
      error = true;
    }
    if (isrc != "") {
      m_src << "  " << isrc << ";\n";
    }
  }

  m_src << "}\n";

  if (m_entry_points_contraction_off.count(result)) {
    m_src << "#pragma OPENCL FP_CONTRACT ON" << std::endl;
  }

  return !error;
}

int translator::translate() {

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

bool translator::validate_module(const std::vector<uint32_t> &binary) const {
  spv_diagnostic diag;
  spv_context ctx = spvContextCreate(m_target_env);
  spv_result_t res =
      spvValidateBinary(ctx, binary.data(), binary.size(), &diag);
  spvDiagnosticPrint(diag);
  spvDiagnosticDestroy(diag);
  if (res != SPV_SUCCESS) {
    return false;
  }
  // std::cout << "Module valid.\n";
  return true;
}

int translator::translate(const std::string &assembly, std::string *srcout) {

  m_ir = BuildModule(m_target_env, spvtools_message_consumer, assembly);

  std::vector<uint32_t> module_bin;
  m_ir->module()->ToBinary(&module_bin, false);
  if (!validate_module(module_bin)) {
    return 1;
  }

  int ret = translate();

  if (ret == 0) {
    *srcout = std::move(m_src.str());
  }

  return ret;
}

int translator::translate(const std::vector<uint32_t> &binary,
                          std::string *srcout) {

  m_ir = BuildModule(m_target_env, spvtools_message_consumer,
                     binary.data(), binary.size());

  if (!validate_module(binary)) {
    return 1;
  }

  int ret = translate();

  if (ret == 0) {
    *srcout = std::move(m_src.str());
  }

  return ret;
}

} // namespace spirv2clc
