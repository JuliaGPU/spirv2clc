std::string translator::src_aggregate_element_type(uint32_t tyid) const {
  // OpenCL forbids pointers inside structs/arrays, so encode every pointer leaf
  // as a same-width integer (ulong under Physical64). Access chains reconstruct
  // the real pointer type on the way in (see emit_access_chain). Non-pointer
  // members keep their normal flat type name (arrays are struct-wrapped).
  if (type_for(tyid)->kind() == Type::Kind::kPointer) {
    return "ulong";
  }
  return src_type(tyid);
}

std::string translator::address_space_qualifier(uint32_t storage) const {
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

std::string translator::src_pointer_type(uint32_t storage, uint32_t tyid, bool signedty) const {
  // Every pointee type (including arrays, which are struct-wrapped) has a flat
  // type name, so a pointer is just "<pointee> <addrspace>*". A pointer-to-array
  // becomes a pointer-to-wrapper, which carries the correct element stride.
  std::string typestr;
  if (signedty) {
    typestr += src_type_signed(tyid);
  } else {
    typestr += src_type(tyid);
  }
  std::string as = address_space_qualifier(storage);
  if (as == "UNIMPLEMENTED") {
    return as;
  }
  typestr += " " + as + "*";
  return typestr;
}

bool translator::translate_type(const Instruction &inst) {
  std::string typestr;
  std::string signedtypestr;
  auto opcode = inst.opcode();
  auto result = inst.result_id();
  switch (opcode) {
  case spv::Op::OpTypePointer: {
    auto storage = inst.GetSingleWordOperand(1);
    auto type = inst.GetSingleWordOperand(2);
    if (m_types_signed.count(type)) {
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
    typestr = src_type(ctype) + std::to_string(cnum);
    // Only integer element types have a signed counterpart; a float vector has
    // no signed form (and asking for one would now fail the translation).
    if (m_types_signed.count(ctype)) {
      signedtypestr = src_type_signed(ctype) + std::to_string(cnum);
    }
    break;
  }
  case spv::Op::OpTypeStruct: { // TODO support volatile members
    // Declare the structure type. Pointer leaves are encoded as integers (see
    // src_aggregate_element_type), as OpenCL forbids pointers in aggregates.
    m_src << "struct " + var_for(result) + " {" << std::endl;
    for (uint32_t opidx = 1; opidx < inst.NumOperands(); opidx++) {
      auto mid = inst.GetSingleWordOperand(opidx);
      m_src << "  " << src_aggregate_element_type(mid) << " m"
            << std::to_string(opidx - 1) << ";" << std::endl;
    }
    m_src << "}";
    if (m_packed.count(result)) {
      m_src << " __attribute__((packed))";
    }
    m_src << ";" << std::endl;

    // Prepare the type name
    typestr = "struct " + var_for(result);
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
    std::string aname = make_valid_identifier("arr" + std::to_string(result));
    m_src << "typedef struct { " << src_aggregate_element_type(elemtyid)
          << " e[" << std::to_string(len) << "]; } " << aname << ";"
          << std::endl;
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
    m_src << typestr << ";" << std::endl;
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

  m_types[result] = typestr;
  if (signedtypestr != "") {
    m_types_signed[result] = signedtypestr;
  }

  return true;
}

bool translator::translate_types_values() {
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
          m_literals[result] = src_cast(rtype, std::to_string(op_val.words[0]));
        } else if (tint->width() == 64) {
          uint64_t w0 = op_val.words[0];
          uint64_t w1 = op_val.words[1];
          auto w = w1 << 32 | w0;
          m_literals[result] = src_cast(rtype, std::to_string(w));
        } else {
          std::cerr << "UNIMPLEMENTED integer constant width " << tint->width()
                    << std::endl;
          return false;
        }
        break;
      }
      case Type::Kind::kFloat: {
        auto tfloat = type->AsFloat();
        auto width = tfloat->width();
        std::ostringstream out;
        if (width == 16) {
          uint32_t w0 = op_val.words[0];
          cl_half h = w0 & 0xFFFF;
          float val = cl_half_to_float(h);
          out.precision(11);
          out << std::fixed << val << "h";
        } else if (width == 32) {
          uint32_t w0 = op_val.words[0];
          float val = *reinterpret_cast<float *>(&w0);
          if (std::isinf(val)) {
            if (std::signbit(val)) {
              out << "-";
            }
            out << "INFINITY";
          } else if (std::isnan(val)) {
            out << "NAN";
          } else {
            out.precision(24);
            out << std::fixed << val << "f";
          }
        } else if (width == 64) {
          uint64_t w0 = op_val.words[0];
          uint64_t w1 = op_val.words[1];
          auto w = w1 << 32 | w0;
          double val = *reinterpret_cast<double *>(&w);
          // NAN/INFINITY are float macros; cast to double so double-typed uses
          // (e.g. copysign(0.0, (double)NAN)) aren't ambiguous against the float
          // overloads.
          if (std::isinf(val)) {
            if (std::signbit(val)) {
              out << "-";
            }
            out << "(double)INFINITY";
          } else if (std::isnan(val)) {
            out << "(double)NAN";
          } else {
            out.precision(53);
            out << std::fixed << val;
          }
        } else {
          std::cerr << "UNIMPLEMENTED float constant width " << width
                    << std::endl;
          return false;
        }
        m_literals[result] = out.str();
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
    case spv::Op::OpConstantNull: {
      std::string cst;
      if (!get_null_constant(rtype, cst)) {
        return false;
      }
      m_literals[result] = cst;
      break;
    }
    case spv::Op::OpConstantTrue: {
      m_literals[result] = "true";
      break;
    }
    case spv::Op::OpConstantFalse: {
      m_literals[result] = "false";
      break;
    }
    case spv::Op::OpConstantSampler: {
      auto addressing_mode = inst.GetSingleWordOperand(2);
      auto normalised = inst.GetSingleWordOperand(3);
      auto filter_mode = inst.GetSingleWordOperand(4);
      m_src << "constant sampler_t " << var_for(result) << " = ";
      switch (addressing_mode) {
      case SpvSamplerAddressingModeClampToEdge:
        m_src << "CLK_ADDRESS_CLAMP_TO_EDGE";
        break;
      case SpvSamplerAddressingModeClamp:
        m_src << "CLK_ADDRESS_CLAMP";
        break;
      case SpvSamplerAddressingModeRepeat:
        m_src << "CLK_ADDRESS_REPEAT";
        break;
      case SpvSamplerAddressingModeRepeatMirrored:
        m_src << "CLK_ADDRESS_MIRRORED_REPEAT";
        break;
      case SpvSamplerAddressingModeNone:
        m_src << "CLK_ADDRESS_NONE";
        break;
      }

      m_src << " | ";

      if (normalised) {
        m_src << "CLK_NORMALIZED_COORDS_TRUE";
      } else {
        m_src << "CLK_NORMALIZED_COORDS_FALSE";
      }

      m_src << " | ";

      switch (filter_mode) {
      case SpvSamplerFilterModeNearest:
        m_src << "CLK_FILTER_NEAREST";
        break;
      case SpvSamplerFilterModeLinear:
        m_src << "CLK_FILTER_LINEAR";
        break;
      }

      m_src << ";" << std::endl;

      break;
    }
    case spv::Op::OpConstantComposite: {
      auto type = type_for(rtype);
      std::string lit;
      switch (type->kind()) {
      case Type::Kind::kVector: {
        auto tvec = type->AsVector();
        // ((type)(c0, c1, ..., cN))
        lit = "((" + src_type(rtype) + ")(";
        const char *sep = "";
        for (uint32_t opidx = 2; opidx < tvec->element_count() + 2; opidx++) {
          auto cid = inst.GetSingleWordOperand(opidx);
          lit += sep;
          lit += m_literals[cid];
          sep = ", ";
        }
        lit += "))";
        m_literals[result] = lit;
        break;
      }
      case Type::Kind::kStruct: {
        auto tstruct = type->AsStruct();
        // ((type){m0, m1, ..., mN})
        lit = "((" + src_type(rtype) + "){";
        const char *sep = "";
        for (uint32_t opidx = 2; opidx < tstruct->element_types().size() + 2;
             opidx++) {
          auto mid = inst.GetSingleWordOperand(opidx);
          lit += sep;
          lit += m_literals[mid];
          sep = ", ";
        }
        lit += "})";
        m_literals[result] = lit;
        break;
      }
      case Type::Kind::kArray: {
        // Array types are wrapped in a struct, so the initializer is
        // ((arrN){{ e0, e1, ... }}): outer braces for the wrapper, inner for
        // the element array member 'e'.
        uint32_t num_elems = array_type_get_length(rtype);
        if (num_elems == 0) {
            return false;
        }
        lit = "((" + src_type(rtype) + "){{";
        const char *sep = "";
        for (uint32_t opidx = 2; opidx < num_elems + 2; opidx++) {
          auto mid = inst.GetSingleWordOperand(opidx);
          lit += sep;
          lit += m_literals[mid];
          sep = ", ";
        }
        lit += "}})";
        m_literals[result] = lit;
        break;
      }
      default:
        std::cerr << "UNIMPLEMENTED OpConstantComposite type " << type->kind()
                  << std::endl;
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
        // a pointer to it, so var_for() is a pointer. Array types are
        // struct-wrapped (no array-to-pointer decay), so the variable can't be
        // used directly as a pointer the way a bare local array used to be.
        auto storagename =
            make_valid_identifier(var_for(result) + "_storage");
        std::string local_var_decl =
            "local " +
            src_type_memory_object_declaration(typointeeid, result,
                                               storagename) +
            "; " + src_type(rtype) + " " + var_for(result) + " = &" +
            storagename;
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
        auto storagename = make_valid_identifier(var_for(result) + "_storage");
        m_src << "constant "
              << src_type_memory_object_declaration(typointeeid, result,
                                                    storagename);
        if (inst.NumOperands() > 3) {
          auto init = inst.GetSingleWordOperand(3);
          m_src << " = " << var_for(init);
        }
        m_src << ";" << std::endl;
        m_names[result] = "(&" + storagename + ")";
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
        auto storagename = make_valid_identifier(var_for(result) + "_storage");
        m_src << "global "
              << src_type_memory_object_declaration(typointeeid, result,
                                                    storagename);
        if (inst.NumOperands() > 3) {
          auto init = inst.GetSingleWordOperand(3);
          m_src << " = " << var_for(init);
        }
        m_src << ";" << std::endl;
        m_names[result] = "(&" + storagename + ")";
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
