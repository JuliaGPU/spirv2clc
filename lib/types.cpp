c::expr_ref translator_impl::to_component(uint32_t composite_tyid,
                                          uint32_t elem_tyid,
                                          c::expr_ref value) const {
  auto composite = type_for(composite_tyid)->kind();
  if (composite == Type::Kind::kVector &&
      type_for(elem_tyid)->kind() == Type::Kind::kBool) {
    return c::unary("-", c::cast("int", std::move(value)));
  }
  if (composite == Type::Kind::kStruct || composite == Type::Kind::kArray) {
    return to_storage(elem_tyid, std::move(value));
  }
  return value;
}

c::expr_ref translator_impl::from_component(uint32_t composite_tyid,
                                            uint32_t elem_tyid,
                                            c::expr_ref component) const {
  // A mask element (-1 or 0) needs no conversion: it is only ever used to
  // initialize a bool, which normalizes it.
  auto composite = type_for(composite_tyid)->kind();
  if (composite == Type::Kind::kStruct || composite == Type::Kind::kArray) {
    return from_storage(elem_tyid, std::move(component));
  }
  return component;
}

std::string translator_impl::src_storage_type(uint32_t tyid) const {
  if (type_for(tyid)->kind() == Type::Kind::kPointer) {
    return src_pointer_int_type();
  }
  return src_type(tyid);
}

c::expr_ref translator_impl::to_storage(uint32_t tyid,
                                        c::expr_ref value) const {
  if (type_for(tyid)->kind() == Type::Kind::kPointer) {
    return c::cast(src_storage_type(tyid), std::move(value));
  }
  return value;
}

c::expr_ref translator_impl::from_storage(uint32_t tyid,
                                          c::expr_ref stored) const {
  if (type_for(tyid)->kind() == Type::Kind::kPointer) {
    return cast_to(tyid, std::move(stored));
  }
  return stored;
}

bool translator_impl::composite_member(const spvtools::opt::Instruction &inst,
                                       unsigned first, uint32_t tyid,
                                       c::expr_ref base, c::expr_ref &member,
                                       uint32_t &leaf_tyid,
                                       uint32_t &parent_tyid) const {
  member = std::move(base);
  parent_tyid = 0;
  for (unsigned i = first; i < inst.NumOperands(); i++) {
    parent_tyid = tyid;
    auto idx = inst.GetSingleWordOperand(i);
    auto type = type_for(tyid);
    switch (type->kind()) {
    case Type::Kind::kVector: {
      std::stringstream scomp;
      scomp << std::hex << idx;
      member = c::member(member, "s" + scomp.str());
      tyid = type_id_for(type->AsVector()->element_type());
      break;
    }
    case Type::Kind::kArray:
      // Arrays are struct-wrapped; index through the 'e' member.
      member =
          c::index(c::member(member, "e"), c::literal(std::to_string(idx)));
      tyid = type_id_for(type->AsArray()->element_type());
      break;
    case Type::Kind::kStruct:
      member = c::member(member, "m" + std::to_string(idx));
      tyid = type_id_for(type->AsStruct()->element_types()[idx]);
      break;
    default:
      std::cerr << "UNIMPLEMENTED composite member access, type "
                << type->kind() << std::endl;
      return false;
    }
  }
  leaf_tyid = tyid;
  return true;
}

std::string translator_impl::address_space_qualifier(uint32_t storage) const {
  switch (storage) {
  case SpvStorageClassCrossWorkgroup:
    return "global";
  case SpvStorageClassUniformConstant:
    return "constant";
  case SpvStorageClassWorkgroup:
    return "local";
  case SpvStorageClassInput:
    return "";
  case SpvStorageClassFunction:
    return "private";
  case SpvStorageClassGeneric:
    // The generic address space is core from OpenCL C 2.0 on. Below that there
    // is no equivalent qualifier (we could try to resolve the storage class
    // from the pointer's origin, but that's not implemented), so fail loudly
    // rather than silently emit something unsupported.
    if (m_opencl_c_version >= 200) {
      return "generic";
    }
    return note_unsupported("generic address space requires OpenCL C 2.0 "
                            "(targeting " +
                            opencl_c_version_str(m_opencl_c_version) + ")");
  default:
    return note_unsupported("pointer storage class " +
                            std::to_string(storage));
  }
}

void translator_impl::declare_pointee_alias(uint32_t tyid) {
  // SPIR-V memory is untyped: OpBitcast freely reinterprets pointers and the
  // resulting loads/stores are well-defined, so the emitted C must not be
  // subject to type-based alias analysis (e.g. a `*(ulong*)` load of float
  // storage is UB that optimizers exploit by dropping the load). Spell every
  // pointee through a may_alias typedef, the portable equivalent of compiling
  // the generated code with -fno-strict-aliasing. A typedef is the one form
  // Clang honours for TBAA: the attribute written inline in a declaration
  // binds to the declared variable instead of the pointee type and has no
  // effect. For a pointer pointee the attribute lands after the inner '*'
  // (qualifier position), correctly marking that pointer level so loads of
  // pointer values are exempted too.
  //
  // Called for the pointee of every OpTypePointer as it is translated, so all
  // typedefs land in the type section (after the pointee's own definition)
  // and every later spelling of a pointer type is a pure lookup.
  auto &info = m_type_info[tyid];
  if (!info.alias.empty()) {
    return;
  }
  switch (type_for(tyid)->kind()) {
  case Type::Kind::kBool:
  case Type::Kind::kInteger:
  case Type::Kind::kFloat:
  case Type::Kind::kVector:
  case Type::Kind::kArray:
  case Type::Kind::kStruct:
  case Type::Kind::kPointer:
    break;
  default:
    // Not dereferenceable through reinterpreted pointers (void, images,
    // samplers, events, opaque structs); keep the raw spelling.
    info.alias = src_type(tyid);
    return;
  }
  auto name = m_name_allocator.allocate("ma" + std::to_string(tyid));
  m_out.types << "typedef " << src_type(tyid) << " __attribute__((may_alias)) "
              << name << ";" << std::endl;
  info.alias = name;
  if (has_signed_type(tyid)) {
    auto signed_name = m_name_allocator.allocate(name + "s");
    m_out.types << "typedef " << src_type_signed(tyid)
                << " __attribute__((may_alias)) " << signed_name << ";"
                << std::endl;
    info.signed_alias = signed_name;
  }
}

translator_impl::MemoryAccess
translator_impl::memory_access_operands(const Instruction &inst,
                                        unsigned index) const {
  MemoryAccess access;
  access.next = index;
  if (inst.NumOperands() <= index) {
    return access;
  }
  access.mask = inst.GetSingleWordOperand(index);
  access.next = index + 1;
  if (access.mask & SpvMemoryAccessAlignedMask) {
    access.alignment = inst.GetSingleWordOperand(access.next);
    access.next += 1;
  }
  // MakePointerAvailable/Visible carry an extra scope id, but they require
  // the Vulkan memory model and cannot appear in OpenCL modules.
  return access;
}

uint32_t translator_impl::pointee_type_id(uint32_t val) const {
  auto pointee = type_for_val(val)->AsPointer()->pointee_type();
  return m_ir->get_type_mgr()->GetId(pointee);
}

uint32_t translator_impl::natural_alignment(uint32_t tyid) const {
  auto tymgr = m_ir->get_type_mgr();
  const Type *ty = type_for(tyid);
  switch (ty->kind()) {
  case Type::Kind::kInteger:
    return std::max(1u, ty->AsInteger()->width() / 8);
  case Type::Kind::kFloat:
    return std::max(1u, ty->AsFloat()->width() / 8);
  case Type::Kind::kVector: {
    // OpenCL C vectors align to their size, with 3-component vectors sized
    // like 4-component ones.
    auto count = ty->AsVector()->element_count();
    if (count == 3) {
      count = 4;
    }
    return count *
           natural_alignment(tymgr->GetId(ty->AsVector()->element_type()));
  }
  case Type::Kind::kArray:
    return natural_alignment(tymgr->GetId(ty->AsArray()->element_type()));
  case Type::Kind::kStruct: {
    if (m_packed.count(tyid)) {
      return 1;
    }
    uint32_t align = 1;
    for (auto *elem : ty->AsStruct()->element_types()) {
      align = std::max(align, natural_alignment(tymgr->GetId(elem)));
    }
    return align;
  }
  case Type::Kind::kPointer:
    return m_pointer_width / 8;
  default:
    // bool, images, samplers, ...: never accessed through reinterpreted
    // pointers, and 1 can never be under-aligned.
    return 1;
  }
}

bool translator_impl::is_underaligned(uint32_t tyid,
                                      const MemoryAccess &access) const {
  return (access.mask & SpvMemoryAccessAlignedMask) && access.alignment != 0 &&
         access.alignment < natural_alignment(tyid);
}

std::string translator_impl::underaligned_alias(uint32_t tyid,
                                                uint32_t alignment) {
  // A C dereference asserts the pointee's natural alignment, but a SPIR-V
  // access can promise less (an Aligned memory operand below the type's
  // natural alignment -- e.g. an i64 load of a 4-aligned pair of floats
  // reached through OpBitcast), and consumers exploit the stronger claim
  // (vectorizers emit aligned instructions, strict targets fault). Such
  // accesses are dereferenced through a reduced-alignment typedef instead --
  // only a typedef can lower alignment in C -- built on top of the pointee's
  // may_alias alias so both properties hold at once.
  auto key = std::make_pair(tyid, alignment);
  auto it = m_underaligned_aliases.find(key);
  if (it != m_underaligned_aliases.end()) {
    return it->second;
  }
  // Every pointee reached by a load/store has an OpTypePointer, so its
  // may_alias alias exists (and is a real typedef: non-dereferenceable kinds
  // have a natural alignment of 1 and are never under-aligned).
  auto base = m_type_info.at(tyid).alias;
  auto name = m_name_allocator.allocate(base + "a" + std::to_string(alignment));
  m_out.types << "typedef " << base << " __attribute__((aligned(" << alignment
              << "))) " << name << ";" << std::endl;
  m_underaligned_aliases.emplace(key, name);
  return name;
}

std::string translator_impl::src_access_pointee(uint32_t tyid,
                                                const MemoryAccess &access) {
  if (is_underaligned(tyid, access)) {
    return underaligned_alias(tyid, access.alignment);
  }
  return m_type_info.at(tyid).alias;
}

std::string translator_impl::src_pointer_type(uint32_t storage, uint32_t tyid,
                                              bool signedty) const {
  // Every pointee type (including arrays, which are struct-wrapped) has a flat
  // type name, so a pointer is just "<pointee> <addrspace>*". A
  // pointer-to-array becomes a pointer-to-wrapper, which carries the correct
  // element stride. Pointees are spelled through their may_alias typedef; see
  // declare_pointee_alias for why. The alias always exists here: every caller
  // spells a pointer type whose OpTypePointer has already been translated.
  auto it = m_type_info.find(tyid);
  std::string typestr;
  if (it != m_type_info.end()) {
    typestr = signedty ? it->second.signed_alias : it->second.alias;
  }
  if (typestr.empty()) {
    return note_unsupported("pointer to type " + std::to_string(tyid) +
                            " without a pointee alias");
  }
  std::string as = address_space_qualifier(storage);
  if (as == "UNIMPLEMENTED") {
    return as;
  }
  typestr += " " + as + "*";
  return typestr;
}

bool translator_impl::translate_type(const Instruction &inst) {
  std::string typestr;
  std::string signedtypestr;
  auto opcode = inst.opcode();
  auto result = inst.result_id();
  switch (opcode) {
  case spv::Op::OpTypePointer: {
    auto storage = inst.GetSingleWordOperand(1);
    auto type = inst.GetSingleWordOperand(2);
    declare_pointee_alias(type);
    if (has_signed_type(type)) {
      signedtypestr = src_pointer_type(storage, type, true);
    }
    typestr = src_pointer_type(storage, type, false);
    break;
  }
  case spv::Op::OpTypeInt: {
    auto width = inst.GetSingleWordOperand(1);
    switch (width) {
    case 8:
      typestr = "uchar";
      signedtypestr = "char";
      break;
    case 16:
      typestr = "ushort";
      signedtypestr = "short";
      break;
    case 32:
      typestr = "uint";
      signedtypestr = "int";
      break;
    case 64:
      typestr = "ulong";
      signedtypestr = "long";
      break;
    default:
      std::cerr << "UNIMPLEMENTED OpTypeInt width " << width << std::endl;
      return false;
    }
    break;
  }
  case spv::Op::OpTypeFloat: {
    auto width = inst.GetSingleWordOperand(1);
    switch (width) {
    case 16:
      typestr = "half";
      break;
    case 32:
      typestr = "float";
      break;
    case 64:
      typestr = "double";
      break;
    default:
      std::cerr << "UNIMPLEMENTED OpTypeFloat width " << width << std::endl;
      return false;
    }
    break;
  }
  case spv::Op::OpTypeVector: {
    auto ctype = inst.GetSingleWordOperand(1);
    auto cnum = inst.GetSingleWordOperand(2);
    if (type_for(ctype)->kind() == Type::Kind::kBool) {
      // A mask; see src_type.
      typestr = "int" + std::to_string(cnum);
      break;
    }
    typestr = src_type(ctype) + std::to_string(cnum);
    // Only integer element types have a signed counterpart; a float vector has
    // no signed form (and asking for one would now fail the translation).
    if (has_signed_type(ctype)) {
      signedtypestr = src_type_signed(ctype) + std::to_string(cnum);
    }
    break;
  }
  case spv::Op::OpTypeStruct: { // TODO support volatile members
    // Declare the structure type. Pointer leaves are encoded as integers (see
    // src_storage_type), as OpenCL forbids pointers in aggregates.
    auto &os = m_out.types;
    os << "struct " + name_of(result) + " {" << std::endl;
    for (uint32_t opidx = 1; opidx < inst.NumOperands(); opidx++) {
      auto mid = inst.GetSingleWordOperand(opidx);
      os << "  " << src_storage_type(mid) << " m" << std::to_string(opidx - 1)
         << ";" << std::endl;
    }
    os << "}";
    if (m_packed.count(result)) {
      os << " __attribute__((packed))";
    }
    os << ";" << std::endl;

    // Prepare the type name
    typestr = "struct " + name_of(result);
    break;
  }
  case spv::Op::OpTypeArray: {
    // C cannot spell a bare array type as a prefix (the name sits inside the
    // declarator) and bare arrays are not assignable. Wrap every array in a
    // struct so it becomes a first-class, copyable value type with a real
    // name. Layout matches ELEM[N] exactly (single trailing array member), so
    // a pointer-to-array is just a pointer-to-wrapper with the correct stride.
    auto elemtyid = inst.GetSingleWordOperand(1);
    uint32_t len = array_type_get_length(result);
    if (len == 0) {
      return false;
    }
    auto aname = m_name_allocator.allocate("arr" + std::to_string(result));
    m_out.types << "typedef struct { " << src_storage_type(elemtyid) << " e["
                << std::to_string(len) << "]; } " << aname << ";" << std::endl;
    typestr = aname;
    break;
  }
  case spv::Op::OpTypeImage: {
    // auto sampledty = inst.GetSingleWordOperand(1);
    auto dim = inst.GetSingleWordOperand(2);
    auto depth = inst.GetSingleWordOperand(3);
    auto arrayed = inst.GetSingleWordOperand(4);
    auto ms = inst.GetSingleWordOperand(5);
    auto sampled = inst.GetSingleWordOperand(6);
    // auto format = inst.GetSingleWordOperand(7);
    auto qual = inst.GetSingleWordOperand(8);

    if ((depth != 0) || (arrayed != 0) || (ms != 0) || (sampled != 0)) {
      std::cerr << "UNIMPLEMENTED image type (depth = " << depth
                << ", arrayed = " << arrayed << ", ms = " << ms
                << "sampled = " << sampled << ")" << std::endl;
      return false;
    }

    switch (qual) {
    case SpvAccessQualifierReadOnly:
      typestr = "read_only";
      break;
    case SpvAccessQualifierWriteOnly:
      typestr = "write_only";
      break;
    case SpvAccessQualifierReadWrite:
      typestr = "read_write";
      m_read_write_images = true;
      break;
    default:
      std::cerr << "UNIMPLEMENTED image access qualifier " << qual << std::endl;
      return false;
    }

    typestr += " ";

    switch (dim) {
    case SpvDim1D:
      typestr += "image1d_t";
      break;
    case SpvDim2D:
      typestr += "image2d_t";
      break;
    case SpvDim3D:
      typestr += "image3d_t";
      // Core in OpenCL C 2.0. Non-aggregate types are unique, so this is
      // emitted at most once.
      if (qual == SpvAccessQualifierWriteOnly && m_opencl_c_version < 200) {
        m_out.extensions
            << "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable"
            << std::endl;
      }
      break;
    default:
      std::cerr << "UNIMPLEMENTED image dimensionality " << dim << std::endl;
      return false;
    }

    break;
  }
  case spv::Op::OpTypeSampledImage: // TODO anything?
    break;
  case spv::Op::OpTypeSampler:
    typestr = "sampler_t";
    break;
  case spv::Op::OpTypeOpaque: {
    auto name = inst.GetOperand(1).AsString();
    typestr = "struct " + name;
    m_out.types << typestr << ";" << std::endl;
    break;
  }
  case spv::Op::OpTypeBool:
    typestr = "bool";
    break;
  case spv::Op::OpTypeVoid:
    typestr = "void";
    break;
  case spv::Op::OpTypeFunction: // FIXME
    break;
  case spv::Op::OpTypeEvent:
    typestr = "event_t";
    break;
  default:
    std::cerr << "UNIMPLEMENTED type instuction " << opcode << std::endl;
    return false;
  }

  auto &info = m_type_info[result];
  info.name = typestr;
  info.signed_name = signedtypestr;

  return true;
}

bool translator_impl::translate_types_values() {
  for (auto &inst : m_ir->module()->types_values()) {
    auto opcode = inst.opcode();
    auto rtype = inst.type_id();
    auto result = inst.result_id();

    switch (opcode) {
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeVoid:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeOpaque:
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampler:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypeEvent:
      if (!translate_type(inst)) {
        return false;
      }
      break;

    case spv::Op::OpConstant: {
      auto &op_val = inst.GetOperand(2);
      auto type = type_for(rtype);
      switch (type->kind()) {
      case Type::Kind::kInteger: {
        auto tint = type->AsInteger();
        if (tint->width() <= 32) {
          m_bindings[result] =
              cast_to(rtype, c::literal(std::to_string(op_val.words[0])));
        } else if (tint->width() == 64) {
          uint64_t w0 = op_val.words[0];
          uint64_t w1 = op_val.words[1];
          auto w = w1 << 32 | w0;
          m_bindings[result] = cast_to(rtype, c::literal(std::to_string(w)));
        } else {
          std::cerr << "UNIMPLEMENTED integer constant width " << tint->width()
                    << std::endl;
          return false;
        }
        break;
      }
      case Type::Kind::kFloat: {
        auto width = type->AsFloat()->width();
        double val;
        std::ostringstream out;
        if (width == 16) {
          val = cl_half_to_float(op_val.words[0] & 0xFFFF);
          out.precision(11);
          out << std::fixed << val << "h";
        } else if (width == 32) {
          uint32_t w0 = op_val.words[0];
          float f;
          std::memcpy(&f, &w0, sizeof(f));
          val = f;
          out.precision(24);
          out << std::fixed << f << "f";
        } else if (width == 64) {
          uint64_t w0 = op_val.words[0];
          uint64_t w1 = op_val.words[1];
          auto w = w1 << 32 | w0;
          std::memcpy(&val, &w, sizeof(val));
          out.precision(53);
          out << std::fixed << val;
        } else {
          std::cerr << "UNIMPLEMENTED float constant width " << width
                    << std::endl;
          return false;
        }
        if (std::isinf(val) || std::isnan(val)) {
          // INFINITY/NAN are float macros. Cast them to the constant's type:
          // for half there is no literal to spell ("infh" is not one), and
          // for double the cast keeps double-typed uses (e.g.
          // copysign(0.0, (double)NAN)) from being ambiguous against the
          // float overloads.
          auto special = c::literal(std::isinf(val) ? "INFINITY" : "NAN");
          if (width != 32) {
            special = cast_to(rtype, special);
          }
          if (std::isinf(val) && std::signbit(val)) {
            special = c::unary("-", special);
          }
          m_bindings[result] = special;
        } else {
          m_bindings[result] = c::literal(out.str());
        }
        break;
      }
      default:
        std::cerr << "UNIMPLEMENTED OpConstant type " << type->kind()
                  << std::endl;
        return false;
      }
      break;
    }
    case spv::Op::OpUndef:
    case spv::Op::OpConstantNull:
      m_bindings[result] = null_constant(rtype);
      break;
    case spv::Op::OpConstantTrue:
      m_bindings[result] = c::literal("true");
      break;
    case spv::Op::OpConstantFalse:
      m_bindings[result] = c::literal("false");
      break;
    case spv::Op::OpConstantSampler: {
      auto addressing_mode = inst.GetSingleWordOperand(2);
      auto normalised = inst.GetSingleWordOperand(3);
      auto filter_mode = inst.GetSingleWordOperand(4);
      auto &os = m_out.globals;
      os << "constant sampler_t " << name_of(result) << " = ";
      switch (addressing_mode) {
      case SpvSamplerAddressingModeClampToEdge:
        os << "CLK_ADDRESS_CLAMP_TO_EDGE";
        break;
      case SpvSamplerAddressingModeClamp:
        os << "CLK_ADDRESS_CLAMP";
        break;
      case SpvSamplerAddressingModeRepeat:
        os << "CLK_ADDRESS_REPEAT";
        break;
      case SpvSamplerAddressingModeRepeatMirrored:
        os << "CLK_ADDRESS_MIRRORED_REPEAT";
        break;
      case SpvSamplerAddressingModeNone:
        os << "CLK_ADDRESS_NONE";
        break;
      }

      os << " | ";

      if (normalised) {
        os << "CLK_NORMALIZED_COORDS_TRUE";
      } else {
        os << "CLK_NORMALIZED_COORDS_FALSE";
      }

      os << " | ";

      switch (filter_mode) {
      case SpvSamplerFilterModeNearest:
        os << "CLK_FILTER_NEAREST";
        break;
      case SpvSamplerFilterModeLinear:
        os << "CLK_FILTER_LINEAR";
        break;
      }

      os << ";" << std::endl;

      break;
    }
    case spv::Op::OpConstantComposite: {
      auto type = type_for(rtype);
      std::vector<c::expr_ref> elems;
      auto defuse = m_ir->get_def_use_mgr();
      for (uint32_t opidx = 2; opidx < inst.NumOperands(); opidx++) {
        auto eid = inst.GetSingleWordOperand(opidx);
        if (type->kind() == Type::Kind::kVector &&
            type->AsVector()->element_type()->kind() == Type::Kind::kBool) {
          // A mask; see src_type.
          bool set = defuse->GetDef(eid)->opcode() == spv::Op::OpConstantTrue;
          elems.push_back(c::literal(set ? "-1" : "0"));
        } else {
          elems.push_back(to_component(rtype, type_id_for(eid), value(eid)));
        }
      }
      switch (type->kind()) {
      case Type::Kind::kVector:
        m_bindings[result] = c::vector_literal(src_type(rtype), elems);
        break;
      case Type::Kind::kStruct:
        m_bindings[result] = c::compound_literal(src_type(rtype), elems);
        break;
      case Type::Kind::kArray:
        // Array types are wrapped in a struct, so the initializer is
        // (arrN){{e0, e1, ...}}: outer braces for the wrapper, inner for the
        // element array member 'e'.
        m_bindings[result] =
            c::compound_literal(src_type(rtype), {c::init_list(elems)});
        break;
      default:
        std::cerr << "UNIMPLEMENTED OpConstantComposite type " << type->kind()
                  << std::endl;
        return false;
      }
      break;
    }
    case spv::Op::OpSpecConstantOp: {
      // llvm-spirv folds constant-expression global initializers into an
      // OpSpecConstantOp carrying an embedded sub-opcode (not runtime
      // specialization). Lower the common pointer/cast forms into a literal.
      auto subop = static_cast<spv::Op>(inst.GetSingleWordOperand(2));
      switch (subop) {
      case spv::Op::OpPtrCastToGeneric:
      case spv::Op::OpGenericCastToPtr:
      case spv::Op::OpBitcast:
      case spv::Op::OpConvertUToPtr:
      case spv::Op::OpConvertPtrToU: {
        // The result type already carries the destination type/address space.
        m_bindings[result] =
            cast_to(rtype, value(inst.GetSingleWordOperand(3)));
        break;
      }
      case spv::Op::OpPtrAccessChain:
      case spv::Op::OpInBoundsPtrAccessChain: {
        // Single-index element offset on a pointer: base + index. (Further
        // struct/array indices are not produced by these constant initializers.)
        if (inst.NumOperands() != 5) {
          std::cerr << "UNIMPLEMENTED OpSpecConstantOp PtrAccessChain with "
                       "multiple indices"
                    << std::endl;
          return false;
        }
        auto base = inst.GetSingleWordOperand(3);
        auto index = inst.GetSingleWordOperand(4);
        m_bindings[result] =
            cast_to(rtype, c::binary("+", value(base), value(index)));
        break;
      }
      default:
        std::cerr << "UNIMPLEMENTED OpSpecConstantOp sub-opcode "
                  << static_cast<uint32_t>(subop) << std::endl;
        return false;
      }
      break;
    }
    case spv::Op::OpVariable: {
      if (m_builtin_variables.count(result) != 0) {
        break;
      }

      auto tyvar = type_for(rtype);
      auto tykind = tyvar->kind();
      if (tykind != Type::Kind::kPointer) {
        std::cerr << "UNIMPLEMENTED global variable with type " << tykind
                  << std::endl;
        return false;
      }

      auto typtr = tyvar->AsPointer();
      auto tymgr = m_ir->get_type_mgr();
      auto typointeeid = tymgr->GetId(typtr->pointee_type());

      auto storage = inst.GetSingleWordOperand(2);

      if (storage == SpvStorageClassWorkgroup) {
        // Mirror the function-local OpVariable pattern: declare the storage and
        // a pointer to it, so the variable's value is a pointer. Array types
        // are struct-wrapped (no array-to-pointer decay), so the variable can't
        // be used directly as a pointer the way a bare local array used to be.
        auto &storagename = derived_name(result, "_storage");
        std::string local_var_decl =
            "local " +
            src_type_memory_object_declaration(typointeeid, result,
                                               storagename) +
            "; " + src_var_decl(result) + " = " +
            c::print(c::address_of(c::name(storagename)));
        m_local_variable_decls[result] = local_var_decl;
      } else if (storage == SpvStorageClassUniformConstant) {
        // Check if initializer is a string array and cache it for later use
        if (inst.NumOperands() > 3) {
          auto init = inst.GetSingleWordOperand(3);
          auto defuse = m_ir->get_def_use_mgr();
          auto init_inst = defuse->GetDef(init);
          if (init_inst &&
              init_inst->opcode() == spv::Op::OpConstantComposite) {
            auto string_literal = get_string_literal(*init_inst);
            if (string_literal) {
              m_constant_string_literals[result] = *string_literal;
            }
          }
        }

        // The SPIR-V id of the variable is a pointer, but OpenCL 1.2 forbids
        // program-scope pointer variables, so we declare the storage as a value
        // and make every reference to the variable take its address.
        auto &storagename = derived_name(result, "_storage");
        m_out.globals << "constant "
                      << src_type_memory_object_declaration(typointeeid, result,
                                                            storagename);
        if (inst.NumOperands() > 3) {
          auto init = inst.GetSingleWordOperand(3);
          m_out.globals << " = " << c::print(value(init));
        }
        m_out.globals << ";" << std::endl;
        m_bindings[result] = c::address_of(c::name(storagename));
      } else if (storage == SpvStorageClassCrossWorkgroup) {
        // Program-scope global variable. Legal only from OpenCL C 2.0 on; below
        // that, program-scope variables must live in the constant address space.
        if (m_opencl_c_version < 200) {
          std::cerr << "UNIMPLEMENTED: program-scope global variable requires "
                       "OpenCL C 2.0 (targeting "
                    << opencl_c_version_str(m_opencl_c_version) << ").\n";
          return false;
        }
        // As with UniformConstant, declare the storage as a value and make every
        // reference take its address (the SPIR-V id is a pointer to it).
        auto &storagename = derived_name(result, "_storage");
        m_out.globals << "global "
                      << src_type_memory_object_declaration(typointeeid, result,
                                                            storagename);
        if (inst.NumOperands() > 3) {
          auto init = inst.GetSingleWordOperand(3);
          m_out.globals << " = " << c::print(value(init));
        }
        m_out.globals << ";" << std::endl;
        m_bindings[result] = c::address_of(c::name(storagename));
      } else {
        std::cerr << "UNIMPLEMENTED global variable with storage class "
                  << storage << std::endl;
        return false;
      }

      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED type/value instruction " << opcode << ".\n";
      return false;
    }
  }
  return true;
}
