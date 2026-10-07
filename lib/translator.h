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

// The translator's internals, kept out of the installed spirv2clc.h.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.h>
#include <spirv/unified1/spirv.h>

namespace spvtools {
namespace opt {

class BasicBlock;
class Instruction;
class IRContext;
class Function;

namespace analysis {
class Type;
}

} // namespace opt
} // namespace spvtools

namespace spirv2clc {

// Hands out C identifiers, unique across the whole program and clear of C and
// OpenCL C keywords and type names. Every identifier the translator emits comes
// from here, both those derived from the module (linkage names, OpName) and
// those it makes up (storage, temporaries, typedefs), so they can't collide.
class name_allocator {
public:
  // Claim `name` verbatim, for names fixed by the module's interface (entry
  // points, linkage names) that can't be changed.
  void reserve(const std::string &name) { m_used.insert(name); }

  // A fresh identifier based on `hint`.
  std::string allocate(const std::string &hint);

private:
  std::unordered_set<std::string> m_used;
};

// The program text, assembled from sections that are concatenated once
// translation is done. A declaration goes into its section whenever the need
// for it arises (e.g. a typedef first required by a function body), so the
// order of translation doesn't dictate the order of the output.
struct output {
  std::ostringstream extensions; // #pragma OPENCL EXTENSION
  std::ostringstream types;      // struct definitions and typedefs
  std::ostringstream globals;    // program-scope variables and samplers
  std::ostringstream prototypes; // declarations of the static functions
  std::ostringstream functions;  // function definitions

  std::string render() const {
    return extensions.str() + types.str() + globals.str() + prototypes.str() +
           functions.str();
  }
};

struct translator_impl {

  translator_impl(spv_target_env env, unsigned opencl_c_version);
  ~translator_impl();

  int translate(const std::string &assembly, std::string *srcout);
  int translate(const std::vector<uint32_t> &binary, std::string *srcout);

private:
  uint32_t type_id_for(uint32_t val) const;

  uint32_t type_id_for(const spvtools::opt::analysis::Type *type) const;

  spvtools::opt::analysis::Type *type_for(uint32_t tyid) const;

  spvtools::opt::analysis::Type *type_for_val(uint32_t val) const;

  uint32_t array_type_get_length(uint32_t tyid) const;

  // The C spelling of value `id`: its literal or builtin query for values
  // that are never declared, its identifier otherwise.
  std::string var_for(uint32_t id) const {
    if (m_literals.count(id)) {
      return m_literals.at(id);
    } else if (m_builtin_values.count(id)) {
      switch (m_builtin_values.at(id)) {
      case SpvBuiltInWorkDim:
        return src_function_call("get_work_dim");
      case SpvBuiltInSubgroupSize:
        return src_function_call("get_sub_group_size");
      case SpvBuiltInSubgroupMaxSize:
        return src_function_call("get_max_sub_group_size");
      case SpvBuiltInNumSubgroups:
        return src_function_call("get_num_sub_groups");
      case SpvBuiltInSubgroupId:
        return src_function_call("get_sub_group_id");
      case SpvBuiltInSubgroupLocalInvocationId:
        return src_function_call("get_sub_group_local_id");
      case SpvBuiltInGlobalInvocationId:
      case SpvBuiltInGlobalOffset:
      case SpvBuiltInGlobalSize:
      case SpvBuiltInWorkgroupId:
      case SpvBuiltInWorkgroupSize:
      case SpvBuiltInLocalInvocationId:
      case SpvBuiltInNumWorkgroups:
        // Vector built-in used as a whole value (e.g. an OpPhi incoming).
        return builtin_vector(id);
      default:
        return note_unsupported("builtin value " +
                                std::to_string(m_builtin_values.at(id)));
      }
    } else if (m_names.count(id)) {
      return m_names.at(id);
    } else {
      return note_unsupported("unnamed id " + std::to_string(id));
    }
  }

  // Give every result id of the module its identifier (see m_names).
  void assign_names();

  // The identifier of an object derived from `id`, e.g. the storage behind a
  // variable or the temporary staging a phi's incoming value. Allocated on
  // first request; later requests return the same name.
  const std::string &derived_name(uint32_t id, const std::string &suffix);

  std::string src_var_decl(uint32_t tyid, const std::string &name) const;

  std::string src_var_decl(uint32_t val) const {
    return src_var_decl(type_id_for(val), var_for(val));
  }

  std::string src_access_chain(const std::string &src_base,
                               const spvtools::opt::analysis::Type *ty,
                               uint32_t index) const;

  // Render an access-chain index as a signed offset. SPIR-V/LLVM access-chain
  // indices are signed (Julia emits -1 for 1-based-to-0-based pointer
  // adjustment); emitting them as their raw unsigned value overflows the
  // pointer arithmetic (undefined behavior) and miscompiles.
  std::string src_signed_index(uint32_t index) const;

  bool emit_access_chain(const spvtools::opt::Instruction &inst,
                         bool ptr_variant, std::string &sval) const;

  std::string src_vec_comp(uint32_t val, uint32_t comp) const {
    std::stringstream scomp;
    scomp << std::hex << comp;
    return var_for(val) + ".s" + scomp.str();
  }

  std::string src_as(uint32_t dtyid, const std::string &src) const {
    return "as_" + src_type(dtyid) + "(" + src + ")";
  }

  std::string src_as(uint32_t dtyid, uint32_t val) const {
    return src_as(dtyid, var_for(val));
  }

  std::string src_as_signed(uint32_t val) const {
    auto varty = type_id_for(val);
    return "as_" + src_type_signed(varty) + "(" + var_for(val) + ")";
  }

  // How SPIR-V types are represented in C. Most are the C type src_type
  // spells, with two exceptions:
  //  - OpenCL C forbids pointers inside structs and arrays, so a pointer stored
  //    in one is an integer of pointer width there (src_storage_type), and is
  //    converted on the way in and out (src_to_storage, src_from_storage).
  //  - OpenCL C has no boolean vectors. A vector of OpTypeBool is an intN mask
  //    with true as -1: what the vector relational and logical operators
  //    produce and what select, any and all test (the sign bit). A scalar bool
  //    is a C bool. Relational results are brought into this form by
  //    src_relational_mask, and src_select_condition adapts a mask to the
  //    width a vector select needs.
  std::string src_type(uint32_t id) const {
    auto it = m_type_info.find(id);
    if (it != m_type_info.end()) {
      return it->second.name;
    }
    return note_unsupported("type " + std::to_string(id));
  }

  // Integer types (and vectors of and pointers to them) are spelled unsigned;
  // this is the signed spelling, for operations that interpret their operands
  // as signed.
  std::string src_type_signed(uint32_t id) const {
    auto it = m_type_info.find(id);
    if (it != m_type_info.end() && !it->second.signed_name.empty()) {
      return it->second.signed_name;
    }
    return note_unsupported("signed type " + std::to_string(id));
  }

  bool has_signed_type(uint32_t id) const {
    auto it = m_type_info.find(id);
    return it != m_type_info.end() && !it->second.signed_name.empty();
  }

  // The integer type pointers are stored as (see above).
  std::string src_pointer_int_type() const {
    return m_pointer_width == 32 ? "uint" : "ulong";
  }

  std::string src_storage_type(uint32_t tyid) const;
  std::string src_to_storage(uint32_t tyid, const std::string &value) const;
  std::string src_from_storage(uint32_t tyid, const std::string &stored) const;

  // `result` is the value of a relational builtin or operator applied to
  // `operand`. For a vector operand it is a mask as wide as the operand's
  // elements; narrow or widen it to the canonical intN.
  std::string src_relational_mask(uint32_t operand,
                                  const std::string &result) const;

  // The condition of a select of `result_tyid` values: a vector select needs a
  // mask as wide as the selected elements.
  std::string src_select_condition(uint32_t cond, uint32_t result_tyid) const;

  // The C type of the components of vector type `tyid`.
  std::string src_vector_element_type(uint32_t tyid) const;

  std::string src_type_memory_object_declaration(uint32_t tid, uint32_t val,
                                                 const std::string &name) const;

  std::string src_type_memory_object_declaration(uint32_t tid,
                                                 uint32_t val) const {
    return src_type_memory_object_declaration(tid, val, var_for(val));
  }

  std::string src_cast(uint32_t ty, std::string src) const {
    return "((" + src_type(ty) + ")" + src + ")";
  }

  std::string src_cast_signed(uint32_t ty, std::string src) const {
    return "((" + src_type_signed(ty) + ")" + src + ")";
  }

  std::string src_cast(uint32_t ty, uint32_t val) const {
    return src_cast(ty, var_for(val));
  }

  std::string src_cast_signed(uint32_t ty, uint32_t val) const {
    return src_cast_signed(ty, var_for(val));
  }

  std::string src_convert(uint32_t val, uint32_t ty) {
    return "convert_" + src_type(ty) + "(" + var_for(val) + ")";
  }

  std::string src_convert_signed(uint32_t val, uint32_t ty) {
    return "convert_" + src_type_signed(ty) + "(" + src_as_signed(val) + ")";
  }

  std::string src_function_call(const std::string &fn) const {
    return fn + "()";
  }

  std::string src_function_call(const std::string &fn,
                                const std::string &srcop1) const {
    return fn + "(" + srcop1 + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1) const {
    return src_function_call(fn, var_for(op1));
  }

  std::string src_function_call_signed(const std::string &fn,
                                       uint32_t op1) const {
    return src_function_call(fn, src_as_signed(op1));
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ")";
  }

  std::string src_function_call_signed(const std::string &fn, uint32_t op1,
                                       uint32_t op2) const {
    return fn + "(" + src_as_signed(op1) + ", " + src_as_signed(op2) + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2, uint32_t op3) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ", " + var_for(op3) +
           ")";
  }

  std::string src_function_call_signed(const std::string &fn, uint32_t op1,
                                       uint32_t op2, uint32_t op3) const {
    return fn + "(" + src_as_signed(op1) + ", " + src_as_signed(op2) + ", " +
           src_as_signed(op3) + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2, uint32_t op3,
                                uint32_t op4) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ", " + var_for(op3) +
           ", " + var_for(op4) + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2, uint32_t op3, uint32_t op4,
                                uint32_t op5) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ", " + var_for(op3) +
           ", " + var_for(op4) + ", " + var_for(op5) + ")";
  }

  // OpenCL spells 64-bit integer atomics atom_* (cl_khr_int64_*_atomics) and
  // 32-bit atomics atomic_* (core since OpenCL C 1.1); pick the name for `op`
  // ("add", "inc", ...) from the atomic pointer's pointee width.
  std::string atomic_builtin(const std::string &op, uint32_t ptr) const;

  // C11 atomic pointer reinterpretation for atomic load/store (OpenCL C 2.0+),
  // e.g. "(volatile global atomic_uint*)(v12)". Picks atomic_int/uint/long/
  // ulong/float/double from the pointee type.
  std::string atomic_c11_pointer(uint32_t ptr) const;

  // The CLK_*_MEM_FENCE flags string for a SPIR-V memory-semantics mask (empty
  // if no memory class is set). Shared by the barrier and fence instructions.
  std::string fence_flags(uint32_t mem_sem) const;

  // The OpenCL address-space keyword for a SPIR-V storage class ("global",
  // "private", ...), empty for storage classes with no qualifier (e.g. Input).
  // Fails translation (returns "UNIMPLEMENTED") for unsupported classes.
  std::string address_space_qualifier(uint32_t storage) const;

  std::string src_pointer_type(uint32_t storage, uint32_t tyid,
                               bool signedty) const;

  // Emit the may_alias typedef(s) for a pointee type into the type section,
  // once per type; see the definition for why every pointee must be
  // may_alias. Called when translating each OpTypePointer.
  void declare_pointee_alias(uint32_t tyid);

  // A parsed MemoryAccess operand of a load/store/copy: the literal mask, the
  // alignment literal when the Aligned bit is set, and the operand index just
  // past this memory operand (== the queried index when absent).
  struct MemoryAccess {
    uint32_t mask = 0;
    uint32_t alignment = 0;
    unsigned next = 0;
  };
  MemoryAccess memory_access_operands(const spvtools::opt::Instruction &inst,
                                      unsigned index) const;

  // The type id of the pointee of a pointer-typed value.
  uint32_t pointee_type_id(uint32_t val) const;

  // The alignment a C dereference asserts of a pointee (OpenCL C rules:
  // scalars/vectors align to their size, aggregates to their strictest
  // member). 1 for types never accessed through reinterpreted pointers.
  uint32_t natural_alignment(uint32_t tyid) const;

  // Whether an access promises less alignment than a C dereference of its
  // pointee would assert.
  bool is_underaligned(uint32_t tyid, const MemoryAccess &access) const;

  // The reduced-alignment typedef for accessing a `tyid` at `alignment`,
  // declared on first request. See the definition for why.
  std::string underaligned_alias(uint32_t tyid, uint32_t alignment);

  // The pointee spelling for an access: the reduced-alignment alias when the
  // access is under-aligned, the plain may_alias alias otherwise.
  std::string src_access_pointee(uint32_t tyid, const MemoryAccess &access);

  // The lvalue for a load/store through `ptr`, honouring the access's
  // MemoryAccess operands: a plain dereference when they claim nothing a C
  // dereference doesn't, a cast through src_access_pointee otherwise.
  std::string src_dereference(uint32_t ptr, const MemoryAccess &access);

  // Render the member path selected by the literal indices of an
  // OpCompositeExtract/OpCompositeInsert (operands `first` onwards), starting
  // from a composite of type `tyid`, e.g. ".m1.e[2].s0". Sets `leaf_tyid` to
  // the type of the selected member and `parent_tyid` to that of the
  // composite holding it.
  bool src_composite_path(const spvtools::opt::Instruction &inst,
                          unsigned first, uint32_t tyid, std::string &path,
                          uint32_t &leaf_tyid, uint32_t &parent_tyid) const;

  // A value of type `elem_tyid` as a component of a composite of type
  // `composite_tyid`, and back: pointers in aggregates are in storage form
  // and booleans in vectors are mask elements (see src_type).
  std::string src_to_component(uint32_t composite_tyid, uint32_t elem_tyid,
                               const std::string &value) const;
  std::string src_from_component(uint32_t composite_tyid, uint32_t elem_tyid,
                                 const std::string &component) const;

  std::string builtin_vector_extract(uint32_t id, uint32_t idx,
                                     bool constant) const;

  // Materialize a whole vector built-in load (e.g. WorkgroupId) as a vector
  // literal of its per-dimension queries, used when the load is consumed as a
  // whole value (OpPhi incoming, store, ...) rather than component-extracted.
  std::string builtin_vector(uint32_t id) const;

  std::optional<std::string>
  get_string_literal(const spvtools::opt::Instruction &inst) const;
  std::optional<std::string> string_literal_for(uint32_t var_id) const;

  bool get_null_constant(uint32_t tyid, std::string &src) const;
  std::string
  translate_extended_unary(const spvtools::opt::Instruction &inst) const;
  std::string
  translate_extended_binary(const spvtools::opt::Instruction &inst) const;
  std::string
  translate_extended_ternary(const spvtools::opt::Instruction &inst) const;
  bool translate_extended_instruction(const spvtools::opt::Instruction &inst,
                                      std::string &src);
  std::string translate_binop(const spvtools::opt::Instruction &inst) const;
  std::string
  translate_binop_signed(const spvtools::opt::Instruction &inst) const;
  bool translate_instruction(const spvtools::opt::Instruction &inst,
                             std::string &src);

  bool translate_capabilities();
  bool translate_extensions() const;
  bool translate_extended_instructions_imports() const;
  bool translate_memory_model();
  bool translate_entry_points();
  bool translate_execution_modes();
  bool translate_debug_instructions();
  bool translate_annotations();
  bool translate_type(const spvtools::opt::Instruction &inst);
  bool translate_types_values();
  bool translate_function(spvtools::opt::Function &func);

  void emit_function_signature(spvtools::opt::Function &func,
                               bool is_prototype);

  // Compute, per non-entry function, the module-scope Workgroup variables it
  // references transitively (see m_function_workgroup_params).
  void compute_workgroup_params();

  bool validate_module(const std::vector<uint32_t> &binary) const;
  int translate();

  // Record that `what` can't be translated: emit a diagnostic, flag the failure
  // so translate() returns an error rather than silently producing a
  // placeholder marker in the output, and return that marker for the caller to
  // splice in.
  std::string note_unsupported(const std::string &what) const;

  void reset() {
    m_translation_failed = false;
    m_out = output();
    m_name_allocator = name_allocator();
    m_debug_names.clear();
    m_names.clear();
    m_derived_names.clear();
    m_type_info.clear();
    m_underaligned_aliases.clear();
    m_literals.clear();
    m_entry_points.clear();
    m_entry_points_local_size.clear();
    m_entry_points_contraction_off.clear();
    m_entry_points_subgroup_size.clear();
    m_builtin_variables.clear();
    m_builtin_values.clear();
    m_rounding_mode_decorations.clear();
    m_saturated_conversions.clear();
    m_exports.clear();
    m_imports.clear();
    m_restricts.clear();
    m_volatiles.clear();
    m_packed.clear();
    m_nowrite_params.clear();
    m_byval_params.clear();
    m_alignments.clear();
    m_phi_vals.clear();
    m_phi_assigns.clear();
    m_sampled_images.clear();
    m_local_variable_decls.clear();
    m_constant_string_literals.clear();
    m_function_workgroup_params.clear();
  }

  spv_target_env m_target_env;
  // Target OpenCL C language version (120, 200, 300).
  unsigned m_opencl_c_version;
  // Set when an unsupported value/type/construct was encountered (see
  // note_unsupported); makes translate() fail instead of emitting a marker.
  mutable bool m_translation_failed = false;

  std::unique_ptr<spvtools::opt::IRContext> m_ir;
  output m_out;
  name_allocator m_name_allocator;
  // OpName strings, as given.
  std::unordered_map<uint32_t, std::string> m_debug_names;
  // Every result id's C identifier (see assign_names).
  std::unordered_map<uint32_t, std::string> m_names;
  std::map<std::pair<uint32_t, std::string>, std::string> m_derived_names;
  // Pointer width in bits, from the addressing model.
  unsigned m_pointer_width = 64;
  struct type_info {
    std::string name;
    std::string signed_name; // empty if the type has no signed variant
    // The may_alias typedefs for pointers to the type (see
    // declare_pointee_alias).
    std::string alias;
    std::string signed_alias;
  };
  std::unordered_map<uint32_t, type_info> m_type_info;
  // (pointee type id, alignment) -> reduced-alignment typedef name (see
  // underaligned_alias).
  std::map<std::pair<uint32_t, uint32_t>, std::string> m_underaligned_aliases;
  std::unordered_map<uint32_t, std::string> m_literals;
  std::unordered_map<uint32_t, std::string> m_entry_points;
  std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t, uint32_t>>
      m_entry_points_local_size;
  std::unordered_set<uint32_t> m_entry_points_contraction_off;
  std::unordered_map<uint32_t, uint32_t> m_entry_points_subgroup_size;
  std::unordered_map<uint32_t, SpvBuiltIn> m_builtin_variables;
  std::unordered_map<uint32_t, SpvBuiltIn> m_builtin_values;
  std::unordered_map<uint32_t, SpvFPRoundingMode> m_rounding_mode_decorations;
  std::unordered_set<uint32_t> m_saturated_conversions;
  std::unordered_map<uint32_t, std::string> m_exports;
  std::unordered_map<uint32_t, std::string> m_imports;
  std::unordered_set<uint32_t> m_restricts;
  std::unordered_set<uint32_t> m_volatiles;
  std::unordered_set<uint32_t> m_packed;
  std::unordered_set<uint32_t> m_nowrite_params;
  std::unordered_set<uint32_t> m_byval_params;
  std::unordered_map<uint32_t, uint32_t> m_alignments;
  std::unordered_map<spvtools::opt::Function *, std::vector<uint32_t>>
      m_phi_vals;
  // phival, val pairs
  std::unordered_map<spvtools::opt::BasicBlock *,
                     std::vector<std::pair<uint32_t, uint32_t>>>
      m_phi_assigns;
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> m_sampled_images;
  std::unordered_map<uint32_t, std::string> m_local_variable_decls;
  std::unordered_map<uint32_t, std::string>
      m_constant_string_literals; // variable id -> string literal
  // Per non-entry function, the module-scope Workgroup (local) variables it
  // references transitively, sorted by id for a stable parameter order. OpenCL
  // C has no program-scope local storage, so the entry kernel owns the storage
  // and passes each to its callees as a `local`-qualified pointer parameter.
  std::unordered_map<uint32_t, std::vector<uint32_t>>
      m_function_workgroup_params;
};

} // namespace spirv2clc
