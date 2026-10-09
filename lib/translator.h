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

#include "cexpr.h"
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

// The body of the function being translated. Translation hands it
// declarations and statements rather than text, and it decides where each goes
// and how it is spelled.
class function_builder {
public:
  // A declaration for the whole function, placed before the first block so
  // that it is in scope wherever control enters (e.g. phi variables).
  void declare_upfront(const std::string &declaration) {
    m_upfront << "  " << declaration << ";\n";
  }

  // A declaration at the current point, with an optional initializer.
  void declare(const std::string &declaration, c::expr_ref init = nullptr) {
    m_body << "  " << declaration;
    if (init) {
      m_body << " = " << c::print(init);
    }
    m_body << ";\n";
  }

  void label(const std::string &name) { m_body << name << ":;\n"; }

  void assign(c::expr_ref lhs, c::expr_ref rhs) {
    expression(c::binary("=", std::move(lhs), std::move(rhs)));
  }

  void expression(const c::expr_ref &e) {
    m_body << "  " << c::print(e) << ";\n";
  }

  // A statement C expressions can't express: control flow, loops.
  void statement(const std::string &text) { m_body << "  " << text << ";\n"; }

  std::string render() const { return m_upfront.str() + m_body.str(); }

private:
  std::ostringstream m_upfront;
  std::ostringstream m_body;
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

  // The identifier of `id` (see assign_names).
  const std::string &name_of(uint32_t id) const;

  // Value `id` where it is used: the expression bound to it if it has one
  // (constants, addresses of program-scope variables, builtin queries: values
  // that are never declared), else its identifier.
  c::expr_ref value(uint32_t id) const;

  // Give every result id of the module its identifier (see m_names).
  void assign_names();

  // The identifier of an object derived from `id`, e.g. the storage behind a
  // variable or the temporary staging a phi's incoming value. Allocated on
  // first request; later requests return the same name.
  const std::string &derived_name(uint32_t id, const std::string &suffix);

  std::string src_var_decl(uint32_t tyid, const std::string &name) const;

  std::string src_var_decl(uint32_t val) const {
    return src_var_decl(type_id_for(val), name_of(val));
  }

  // An access-chain index as a signed offset. SPIR-V/LLVM access-chain indices
  // are signed (Julia emits -1 for 1-based-to-0-based pointer adjustment);
  // using them as their raw unsigned value overflows the pointer arithmetic
  // (undefined behavior) and miscompiles.
  c::expr_ref signed_index(uint32_t index) const;

  bool emit_access_chain(const spvtools::opt::Instruction &inst,
                         bool ptr_variant, c::expr_ref &result) const;

  // Component `comp` of vector value `val`.
  c::expr_ref vector_component(uint32_t val, uint32_t comp) const;

  // Reinterpret the bits of `e` (as_<type>), or the bits of value `val` as the
  // signed variant of its type.
  c::expr_ref as_type(uint32_t dtyid, c::expr_ref e) const {
    return c::call("as_" + src_type(dtyid), {std::move(e)});
  }
  c::expr_ref as_signed(uint32_t val) const {
    return c::call("as_" + src_type_signed(type_id_for(val)), {value(val)});
  }

  // A C conversion of `e` to `tyid`, or to its signed variant.
  c::expr_ref cast_to(uint32_t tyid, c::expr_ref e) const {
    return c::cast(src_type(tyid), std::move(e));
  }
  c::expr_ref cast_to_signed(uint32_t tyid, c::expr_ref e) const {
    return c::cast(src_type_signed(tyid), std::move(e));
  }

  // Call `fn` on values, or on their signed reinterpretations.
  c::expr_ref call_values(const std::string &fn,
                          const std::vector<uint32_t> &args) const;
  c::expr_ref call_signed(const std::string &fn,
                          const std::vector<uint32_t> &args) const;

  // How SPIR-V types are represented in C. Most are the C type src_type
  // spells, with two exceptions:
  //  - OpenCL C forbids pointers inside structs and arrays, so a pointer stored
  //    in one is an integer of pointer width there (src_storage_type), and is
  //    converted on the way in and out (to_storage, from_storage).
  //  - OpenCL C has no boolean vectors. A vector of OpTypeBool is an intN mask
  //    with true as -1: what the vector relational and logical operators
  //    produce and what select, any and all test (the sign bit). A scalar bool
  //    is a C bool. Relational results are brought into this form by
  //    relational_mask, and select_condition adapts a mask to the width a
  //    vector select needs.
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
  c::expr_ref to_storage(uint32_t tyid, c::expr_ref value) const;
  c::expr_ref from_storage(uint32_t tyid, c::expr_ref stored) const;

  // `result` is the value of a relational builtin or operator applied to
  // `operand`. For a vector operand it is a mask as wide as the operand's
  // elements; narrow or widen it to the canonical intN.
  c::expr_ref relational_mask(uint32_t operand, c::expr_ref result) const;

  // The condition of a select of `result_tyid` values: a vector select needs a
  // mask as wide as the selected elements.
  c::expr_ref select_condition(uint32_t cond, uint32_t result_tyid) const;

  // The C type of the components of vector type `tyid`.
  std::string src_vector_element_type(uint32_t tyid) const;

  std::string src_type_memory_object_declaration(uint32_t tid, uint32_t val,
                                                 const std::string &name) const;

  std::string src_type_memory_object_declaration(uint32_t tid,
                                                 uint32_t val) const {
    return src_type_memory_object_declaration(tid, val, name_of(val));
  }

  // OpenCL spells 64-bit integer atomics atom_* (cl_khr_int64_*_atomics) and
  // 32-bit atomics atomic_* (core since OpenCL C 1.1); pick the name for `op`
  // ("add", "inc", ...) from the atomic pointer's pointee width.
  std::string atomic_builtin(const std::string &op, uint32_t ptr) const;

  // `ptr` reinterpreted as a pointer to the C11 atomic type the OpenCL C 2.0
  // atomics operate on, e.g. "(volatile global atomic_uint*)v12". Integers are
  // atomic_uint/ulong, or atomic_int/long for `is_signed`, which selects the
  // signed comparison of atomic_fetch_min/max.
  c::expr_ref atomic_c11_pointer(uint32_t ptr, bool is_signed = false) const;

  // The value of the constant Scope or Memory Semantics operand `id`, or
  // nothing (after reporting) if it is computed at run time.
  std::optional<uint32_t> constant_operand(uint32_t id, const char *what) const;

  // The OpenCL C 2.0 memory_scope for a SPIR-V Scope, or null (after
  // reporting) for one OpenCL C has no counterpart for.
  c::expr_ref memory_scope(uint32_t scope) const;

  // The CLK_*_MEM_FENCE flags for a SPIR-V memory-semantics mask (0 if no
  // memory class is set). Shared by the barrier and fence instructions.
  c::expr_ref fence_flags(uint32_t mem_sem) const;

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
  c::expr_ref dereference(uint32_t ptr, const MemoryAccess &access);

  // The member of composite `base` (of type `tyid`) selected by the literal
  // indices of an OpCompositeExtract/OpCompositeInsert (operands `first`
  // onwards), e.g. base.m1.e[2].s0. Sets `leaf_tyid` to the type of the
  // selected member and `parent_tyid` to that of the composite holding it.
  bool composite_member(const spvtools::opt::Instruction &inst, unsigned first,
                        uint32_t tyid, c::expr_ref base, c::expr_ref &member,
                        uint32_t &leaf_tyid, uint32_t &parent_tyid) const;

  // A value of type `elem_tyid` as a component of a composite of type
  // `composite_tyid`, and back: pointers in aggregates are in storage form
  // and booleans in vectors are mask elements (see src_type).
  c::expr_ref to_component(uint32_t composite_tyid, uint32_t elem_tyid,
                           c::expr_ref value) const;
  c::expr_ref from_component(uint32_t composite_tyid, uint32_t elem_tyid,
                             c::expr_ref component) const;

  // The query of component `idx` of vector built-in `builtin`.
  c::expr_ref builtin_vector_extract(SpvBuiltIn builtin, c::expr_ref idx) const;

  // The whole value of built-in `builtin` of type `tyid`. A vector built-in
  // (e.g. WorkgroupId) is a vector literal of its per-dimension queries.
  c::expr_ref builtin_value(SpvBuiltIn builtin, uint32_t tyid) const;

  // The value of scalar built-in `builtin`.
  c::expr_ref builtin_scalar(SpvBuiltIn builtin) const;

  std::optional<std::string>
  get_string_literal(const spvtools::opt::Instruction &inst) const;
  std::optional<std::string> string_literal_for(uint32_t var_id) const;

  c::expr_ref null_constant(uint32_t tyid) const;
  c::expr_ref
  translate_extended_unary(const spvtools::opt::Instruction &inst) const;
  c::expr_ref
  translate_extended_binary(const spvtools::opt::Instruction &inst) const;
  c::expr_ref
  translate_extended_ternary(const spvtools::opt::Instruction &inst) const;
  bool translate_extended_instruction(const spvtools::opt::Instruction &inst,
                                      function_builder &fb);
  c::expr_ref translate_binop(const spvtools::opt::Instruction &inst) const;
  c::expr_ref
  translate_binop_signed(const spvtools::opt::Instruction &inst) const;
  bool translate_instruction(const spvtools::opt::Instruction &inst,
                             function_builder &fb);

  // Value `id` as an lvalue: a variable it is copied into first if it is bound
  // to an expression (e.g. a constant).
  c::expr_ref materialize(uint32_t id, function_builder &fb);

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
    m_read_write_images = false;
    m_out = output();
    m_name_allocator = name_allocator();
    m_debug_names.clear();
    m_names.clear();
    m_derived_names.clear();
    m_type_info.clear();
    m_underaligned_aliases.clear();
    m_bindings.clear();
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
  // Whether the module declares read-write images, the only ones a work-item
  // can need to fence its own accesses to.
  bool m_read_write_images = false;
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
  // Expressions standing for values that are never declared (see value).
  std::unordered_map<uint32_t, c::expr_ref> m_bindings;
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
