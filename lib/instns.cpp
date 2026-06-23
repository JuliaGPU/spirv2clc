
bool translator::emit_access_chain(const Instruction &inst, bool ptr_variant,
                                   std::string &sval) const {
  auto base = inst.GetSingleWordOperand(2);
  const Type *cty = type_for_val(base)->AsPointer()->pointee_type();
  unsigned i;
  if (ptr_variant) {
    // The Ptr* variants take a leading Element index that does pointer
    // arithmetic on the base (scaled by the pointee size, which is correct now
    // that pointer-to-array is a pointer-to-wrapper).
    auto elem = inst.GetSingleWordOperand(3);
    sval = "&" + var_for(base) + "[" + var_for(elem) + "]";
    i = 4;
  } else {
    // The plain variants' first index walks into the pointee directly.
    sval = var_for(base);
    i = 3;
  }
  bool descended_aggregate = false;
  for (; i < inst.NumOperands(); i++) {
    descended_aggregate = true;
    auto idx = inst.GetSingleWordOperand(i);
    sval = src_access_chain(sval, cty, idx);
    switch (cty->kind()) {
    case Type::Kind::kArray:
      cty = cty->AsArray()->element_type();
      break;
    case Type::Kind::kStruct: {
      auto cstmgr = m_ir->get_constant_mgr();
      auto idx_cst = cstmgr->FindDeclaredConstant(idx);
      if (idx_cst == nullptr) {
        std::cerr << "UNIMPLEMENTED struct access with non-constant index"
                  << std::endl;
        return false;
      }
      uint32_t struct_idx = idx_cst->GetU32();
      cty = cty->AsStruct()->element_types()[struct_idx];
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED access chain type " << cty->kind()
                << std::endl;
      return false;
    }
  }
  // If the chain descended through an aggregate to land on a pointer slot, that
  // slot was encoded as an integer (see src_aggregate_element_type). Cast the
  // address back to the real pointer type so the subsequent load/store of the
  // reconstructed pointer type-checks; the bit pattern is preserved under
  // Physical64 where pointers are pointer-width integers.
  if (descended_aggregate && cty->kind() == Type::Kind::kPointer) {
    sval = "(" + src_type(inst.type_id()) + ")(" + sval + ")";
  }
  return true;
}

bool translator::translate_instruction(const Instruction &inst,
                                       std::string &src) {
  auto opcode = inst.opcode();
  auto rtype = inst.type_id();
  auto result = inst.result_id();

  std::string sval;
  bool assign_result = true;
  bool boolean_result = false;
  std::string boolean_result_src_type;

  switch (opcode) {
  case spv::Op::OpUndef: {
    if (!get_null_constant(rtype, sval)) {
      return false;
    }
    break;
  }
  case spv::Op::OpUnreachable: // TODO trigger crash? end invocation?
    break;
  case spv::Op::OpReturn:
    src = "return";
    break;
  case spv::Op::OpReturnValue: {
    auto val = inst.GetSingleWordOperand(0);
    src = "return " + var_for(val);
    break;
  }
  case spv::Op::OpFunctionCall: {
    auto func = inst.GetSingleWordOperand(2);
    sval = var_for(func) + "(";
    const char *sep = "";
    for (unsigned i = 3; i < inst.NumOperands(); i++) {
      auto param = inst.GetSingleWordOperand(i);
      sval += sep;
      sval += var_for(param);
      sep = ", ";
    }
    sval += ")";
    if (type_for(rtype)->kind() == Type::Kind::kVoid) {
      assign_result = false;
      src = sval;
    }
    break;
  }
  case spv::Op::OpCopyObject: {
    auto obj = inst.GetSingleWordOperand(2);
    sval = var_for(obj);
    break;
  }
  case spv::Op::OpLifetimeStart:
  case spv::Op::OpLifetimeStop:
    break;
  case spv::Op::OpVariable: {
    // auto storage = inst.GetSingleWordOperand(2); TODO make storage explicit?
    assign_result = false;
    auto varty = type_for(rtype)->AsPointer()->pointee_type();
    auto storagename = var_for(result) + "_storage";
    storagename = make_valid_identifier(storagename);
    // Declare storage
    auto tymgr = m_ir->get_type_mgr();
    src = src_type_memory_object_declaration(tymgr->GetId(varty), result,
                                             storagename);
    if (inst.NumOperands() == 4) {
      auto init = inst.GetSingleWordOperand(3);
      src += " = " + var_for(init);
    }
    src += "; ";
    // Declare pointer
    src += src_type(rtype) + " " + var_for(result) + " = &" + storagename;
    break;
  }
  case spv::Op::OpLoad: {
    auto ptr = inst.GetSingleWordOperand(2);
    if (m_builtin_variables.count(ptr)) {
      m_builtin_values[result] = m_builtin_variables.at(ptr);
      assign_result = false;
    } else {
      sval = "*" + var_for(ptr);
    }
    break;
  }
  case spv::Op::OpStore: {
    auto ptr = inst.GetSingleWordOperand(0);
    auto val = inst.GetSingleWordOperand(1);
    src = "*" + var_for(ptr) + " = " + var_for(val);
    break;
  }
  case spv::Op::OpCopyMemory:
  case spv::Op::OpCopyMemorySized: {
    auto target = inst.GetSingleWordOperand(0);
    auto source = inst.GetSingleWordOperand(1);
    assign_result = false;
    // Copy the whole pointee by value (arrays are struct-wrapped, so this is a
    // legal aggregate assignment). The sized form may have differently-typed
    // but same-sized operands, so reinterpret the target through the source's
    // pointer type before the copy.
    src = "*((" + src_type(type_id_for(source)) + ")(" + var_for(target) +
          ")) = *(" + var_for(source) + ")";
    break;
  }
  case spv::Op::OpConvertPtrToU:
  case spv::Op::OpConvertUToPtr: {
    auto src = inst.GetSingleWordOperand(2);
    sval = src_cast(rtype, src);
    break;
  }
  case spv::Op::OpPtrAccessChain:
  case spv::Op::OpInBoundsPtrAccessChain: {
    if (!emit_access_chain(inst, /*ptr_variant=*/true, sval)) {
      return false;
    }
    break;
  }
  case spv::Op::OpAccessChain:
  case spv::Op::OpInBoundsAccessChain: {
    if (!emit_access_chain(inst, /*ptr_variant=*/false, sval)) {
      return false;
    }
    break;
  }
  case spv::Op::OpSampledImage: {
    auto image = inst.GetSingleWordOperand(2);
    auto sampler = inst.GetSingleWordOperand(3);
    m_sampled_images[result] = std::make_pair(image, sampler);
    assign_result = false;
    break;
  }
  case spv::Op::OpImageSampleExplicitLod: {
    auto sampledimage = inst.GetSingleWordOperand(2);
    auto coord = inst.GetSingleWordOperand(3);
    // auto operands = inst.GetSingleWordOperand(4); FIXME translate
    bool is_float = type_for(rtype)->kind() == Type::Kind::kFloat;
    bool is_float_coord = type_for_val(coord)->kind() == Type::Kind::kFloat;

    if (!is_float) {
      sval += "as_uint4(";
    }

    sval += "read_image";

    if (is_float) {
      sval += "f";
    } else {
      sval += "i"; // FIXME i vs. ui
    }

    sval += "(";
    sval += var_for(m_sampled_images.at(sampledimage).first);
    sval += ", ";
    sval += var_for(m_sampled_images.at(sampledimage).second);
    sval += ", ";
    if (!is_float_coord) {
      sval += "as_int2(";
    }
    sval += var_for(coord);
    if (!is_float_coord) {
      sval += ")";
    }
    sval += ")";
    if (!is_float) {
      sval += ")";
    }
    // TODO check Lod

    break;
  }
#if 0
    case spv::Op::OpImageWrite: {
        auto image = inst.GetSingleWordOperand(0);
        auto coord = inst.GetSingleWordOperand(1);
        auto texel = inst.GetSingleWordOperand(2);
        auto tycoord = type_for_val(coord);
        auto tytexel = type_for_val(texel);
        bool is_float = tytexel->kind() == Type::Kind::kFloat;
        bool is_float_coord = tycoord->kind() == Type::Kind::kFloat;
        src = "write_image";
        if (is_float) {
            src += "f";
        } else {
            src += "ui"; // FIXME i vs. ui
        }
        src += "(";
        src += var_for(image);
        src += ", ";
        if (!is_float_coord) {
            src += "as_int2(";
        }
        src += var_for(coord);
        if (!is_float_coord) {
            src += ")";
        }
        src += ", ";
        src += var_for(texel);
        src += ")";
        break;
    }
#endif
  case spv::Op::OpImageQuerySizeLod: {
    auto image = inst.GetSingleWordOperand(2);
    // auto lod = inst.GetSingleWordOperand(3); // FIXME validate
    sval = "((" + src_type(rtype) + ")(";
    auto tyimg = type_for_val(image);
    sval += "get_image_width(" + var_for(image) + ")";
    auto dim = tyimg->AsImage()->dim();
    if ((dim == spv::Dim::Dim2D) || (dim == spv::Dim::Dim3D)) {
      sval += ", get_image_height(" + var_for(image) + ")";
    }
    if (dim == spv::Dim::Dim3D) {
      sval += ", get_image_depth(" + var_for(image) + ")";
    }
    sval += "))";
    break;
  }
  case spv::Op::OpAtomicIIncrement: {
    auto ptr = inst.GetSingleWordOperand(2);
    sval = src_function_call("atomic_inc", ptr); // FIXME exact semantics
    break;
  }
  case spv::Op::OpAtomicIDecrement: {
    auto ptr = inst.GetSingleWordOperand(2);
    sval = src_function_call("atomic_dec", ptr); // FIXME exact semantics
    break;
  }
  case spv::Op::OpAtomicAnd:
  case spv::Op::OpAtomicExchange:
  case spv::Op::OpAtomicIAdd:
  case spv::Op::OpAtomicISub:
  case spv::Op::OpAtomicOr:
  case spv::Op::OpAtomicSMax:
  case spv::Op::OpAtomicSMin:
  case spv::Op::OpAtomicUMax:
  case spv::Op::OpAtomicUMin:
  case spv::Op::OpAtomicXor: {
    static std::unordered_map<spv::Op, const char *> fns{
        {spv::Op::OpAtomicAnd, "atomic_and"},
        {spv::Op::OpAtomicExchange, "atomic_xchg"},
        {spv::Op::OpAtomicIAdd, "atomic_add"},
        {spv::Op::OpAtomicISub, "atomic_sub"},
        {spv::Op::OpAtomicOr, "atomic_or"},
        {spv::Op::OpAtomicSMax, "atomic_max"},
        {spv::Op::OpAtomicSMin, "atomic_min"},
        {spv::Op::OpAtomicUMax, "atomic_max"},
        {spv::Op::OpAtomicUMin, "atomic_min"},
        {spv::Op::OpAtomicXor, "atomic_xor"},
    };
    auto ptr = inst.GetSingleWordOperand(2);
    auto val = inst.GetSingleWordOperand(5);
    sval = src_function_call(fns.at(opcode), ptr, val); // FIXME exact semantics
    break;
  }
  case spv::Op::OpAtomicCompareExchange: {
    auto ptr = inst.GetSingleWordOperand(2);
    auto val = inst.GetSingleWordOperand(6);
    auto cmp = inst.GetSingleWordOperand(7);
    sval = src_function_call("atomic_cmpxchg", ptr, cmp,
                             val); // FIXME exact semantics
    break;
  }
  case spv::Op::OpCompositeExtract: {
    auto comp = inst.GetSingleWordOperand(2);
    auto idx = inst.GetSingleWordOperand(3); // FIXME support multiple indices
    if (m_builtin_values.count(comp)) {
      sval = builtin_vector_extract(comp, idx, true);
      break;
    }
    auto type = type_for_val(comp);
    switch (type->kind()) {
    case Type::Kind::kVector: {
      sval = src_vec_comp(comp, idx);
      break;
    }
    case Type::Kind::kArray: {
      // Arrays are struct-wrapped; index through the 'e' member.
      sval = var_for(comp) + ".e[" + std::to_string(idx) + "]";
      break;
    }
    case Type::Kind::kStruct: {
      sval = var_for(comp) + ".m" + std::to_string(idx);
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED OpCompositeExtract, type " << type->kind()
                << std::endl;
      return false;
    }
    break;
  }
  case spv::Op::OpCompositeInsert: {
    auto object = inst.GetSingleWordOperand(2);
    auto composite = inst.GetSingleWordOperand(3);
    auto index = inst.GetSingleWordOperand(4);

    if (inst.NumOperands() > 5) {
      std::cerr << "UNIMPLEMENTED OpCompositeInsert with multiple indices"
                << std::endl;
      return false;
    }

    assign_result = false;
    src = src_type(rtype) + " " + var_for(result) + " = " + var_for(composite) +
          "; ";
    auto type = type_for(rtype);
    switch (type->kind()) {
    case Type::Kind::kVector:
      src += src_vec_comp(result, index) + " = " + var_for(object);
      break;
    case Type::Kind::kArray:
      // Arrays are struct-wrapped; index through the 'e' member.
      src += var_for(result) + ".e[" + std::to_string(index) +
             "] = " + var_for(object);
      break;
    case Type::Kind::kStruct:
      src += var_for(result) + ".m" + std::to_string(index) + " = " +
             var_for(object);
      break;
    default:
      std::cerr << "UNIMPLEMENTED OpCompositeInsert, type " << type->kind()
                << std::endl;
      return false;
    }
    break;
  }
  case spv::Op::OpCompositeConstruct: {
    // Arrays are struct-wrapped, so their elements live in the 'e' member and
    // need an extra brace level: { { e0, e1, ... } }.
    bool is_array = type_for(rtype)->kind() == Type::Kind::kArray;
    sval = is_array ? "{{" : "{";
    const char *sep = "";
    for (unsigned i = 2; i < inst.NumOperands(); i++) {
      auto mem = inst.GetSingleWordOperand(i);
      sval += sep;
      sval += var_for(mem);
      sep = ", ";
    }
    sval += is_array ? "}}" : "}";
    break;
  }
  case spv::Op::OpVectorExtractDynamic: {
    // ((elemtype)&vec)[elem]
    auto vec = inst.GetSingleWordOperand(2);
    auto idx = inst.GetSingleWordOperand(3);
    if (m_builtin_values.count(vec)) {
      sval = builtin_vector_extract(vec, idx, false);
    } else {
      sval = "((" + src_type(rtype) + "*)&" + var_for(vec) + ")[" + var_for(idx) +
           "]";
    }
    break;
  }
  case spv::Op::OpVectorInsertDynamic: {
    auto vec = inst.GetSingleWordOperand(2);
    auto comp = inst.GetSingleWordOperand(3);
    auto comp_type_id = type_id_for(comp);
    auto idx = inst.GetSingleWordOperand(4);
    sval = var_for(vec);
    sval += "; ";
    sval += "((" + src_type(comp_type_id) + "*)&" + var_for(result) + ")[" +
            var_for(idx) + "] = " + var_for(comp);
    break;
  }
  case spv::Op::OpVectorShuffle: {
    auto v1 = inst.GetSingleWordOperand(2);
    auto v2 = inst.GetSingleWordOperand(3);
    auto n1 = type_for_val(v1)->AsVector()->element_count();
    sval = "((" + src_type(rtype) + ")(";
    const char *sep = "";
    for (unsigned i = 4; i < inst.NumOperands(); i++) {
      auto comp = inst.GetSingleWordOperand(i);
      auto srcvec = v1;
      sval += sep;
      if (comp == 0xFFFFFFFFU) {
        sval += "0";
      } else {
        if (comp >= n1) {
          srcvec = v2;
          comp -= n1;
        }
        sval += src_vec_comp(srcvec, comp);
      }
      sep = ", ";
    }
    sval += "))";
    break;
  }
  case spv::Op::OpSDiv:
  case spv::Op::OpSRem:
  case spv::Op::OpShiftRightArithmetic:
    sval = src_as(rtype, translate_binop_signed(inst));
    break;
  case spv::Op::OpVectorTimesScalar:
  case spv::Op::OpShiftLeftLogical:
  case spv::Op::OpShiftRightLogical:
  case spv::Op::OpFAdd:
  case spv::Op::OpFSub:
  case spv::Op::OpFDiv:
  case spv::Op::OpFMul:
  case spv::Op::OpISub:
  case spv::Op::OpIAdd:
  case spv::Op::OpIMul:
  case spv::Op::OpUDiv:
  case spv::Op::OpUMod:
  case spv::Op::OpBitwiseOr:
  case spv::Op::OpBitwiseXor:
  case spv::Op::OpBitwiseAnd:
    sval = translate_binop(inst);
    break;
  case spv::Op::OpFMod:
  case spv::Op::OpFRem: {
    auto op1 = inst.GetSingleWordOperand(2);
    auto op2 = inst.GetSingleWordOperand(3);
    sval = src_function_call("fmod", op1, op2);
    break;
  }
  case spv::Op::OpSNegate:
  case spv::Op::OpFNegate: {
    auto op = inst.GetSingleWordOperand(2);
    sval = "-" + var_for(op);
    break;
  }
  case spv::Op::OpLogicalNot: {
    auto op = inst.GetSingleWordOperand(2);
    sval = "!" + var_for(op);
    break;
  }
  case spv::Op::OpNot: {
    auto op = inst.GetSingleWordOperand(2);
    sval = "~" + var_for(op);
    break;
  }
  case spv::Op::OpLessOrGreater: {
    auto op1 = inst.GetSingleWordOperand(2);
    auto op2 = inst.GetSingleWordOperand(3);
    boolean_result = true;
    boolean_result_src_type = src_type_boolean_for_val(op1);
    sval = src_function_call("islessgreater", op1, op2);
    break;
  }
  case spv::Op::OpFOrdEqual:
  case spv::Op::OpFOrdNotEqual:
  case spv::Op::OpFOrdLessThan:
  case spv::Op::OpFOrdGreaterThan:
  case spv::Op::OpFOrdLessThanEqual:
  case spv::Op::OpFOrdGreaterThanEqual:
  case spv::Op::OpFUnordEqual:
  case spv::Op::OpFUnordNotEqual:
  case spv::Op::OpFUnordLessThan:
  case spv::Op::OpFUnordGreaterThan:
  case spv::Op::OpFUnordLessThanEqual:
  case spv::Op::OpFUnordGreaterThanEqual:
  case spv::Op::OpLogicalOr:
  case spv::Op::OpLogicalAnd:
  case spv::Op::OpULessThan:
  case spv::Op::OpULessThanEqual:
  case spv::Op::OpUGreaterThan:
  case spv::Op::OpUGreaterThanEqual:
  case spv::Op::OpLogicalEqual:
  case spv::Op::OpLogicalNotEqual:
  case spv::Op::OpIEqual:
  case spv::Op::OpINotEqual: {
    auto op1 = inst.GetSingleWordOperand(2);
    boolean_result = true;
    boolean_result_src_type = src_type_boolean_for_val(op1);
    sval = translate_binop(inst);
    break;
  }
  case spv::Op::OpSLessThanEqual:
  case spv::Op::OpSGreaterThan:
  case spv::Op::OpSGreaterThanEqual:
  case spv::Op::OpSLessThan: {
    auto op1 = inst.GetSingleWordOperand(2);
    boolean_result = true;
    boolean_result_src_type = src_type_boolean_for_val(op1);
    sval = translate_binop_signed(inst);
    break;
  }
  case spv::Op::OpAny: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("any", val);
    break;
  }
  case spv::Op::OpAll: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("all", val);
    break;
  }
  case spv::Op::OpIsNan: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isnan", val);
    break;
  }
  case spv::Op::OpIsInf: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isinf", val);
    break;
  }
  case spv::Op::OpIsFinite: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isfinite", val);
    break;
  }
  case spv::Op::OpIsNormal: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isnormal", val);
    break;
  }
  case spv::Op::OpSignBitSet: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("signbit", val);
    break;
  }
  case spv::Op::OpBitCount: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("popcount", val);
    break;
  }
  case spv::Op::OpOrdered: {
    auto x = inst.GetSingleWordOperand(2);
    auto y = inst.GetSingleWordOperand(3);
    sval = src_function_call("isordered", x, y);
    break;
  }
  case spv::Op::OpUnordered: {
    auto x = inst.GetSingleWordOperand(2);
    auto y = inst.GetSingleWordOperand(3);
    sval = src_function_call("isunordered", x, y);
    break;
  }
  case spv::Op::OpConvertFToU:
  case spv::Op::OpConvertFToS: {
    auto op = inst.GetSingleWordOperand(2);
    bool sat = m_saturated_conversions.count(result);
    sval = "convert_";
    if (opcode == spv::Op::OpConvertFToU) {
      sval += src_type(rtype);
    } else {
      sval += src_type_signed(rtype);
    }

    if (sat) {
      sval += "_sat";
    }

    if (m_rounding_mode_decorations.count(result)) {
      auto rmode = m_rounding_mode_decorations.at(result);
      sval += "_" + rounding_mode(rmode);
    } else {
      sval += "_" + rounding_mode(SpvFPRoundingModeRTZ);
    }

    sval += "(" + var_for(op) + ")";

    // SPIR-V requires that NaNs be converted to 0 for saturating conversions
    // but OpenCL C just recommends it (§6.2.3)
    if (sat) {
      sval = src_function_call("isnan", op) + " ? 0 : " + sval;
    }

    break;
  }
  case spv::Op::OpDot: {
    auto v1 = inst.GetSingleWordOperand(2);
    auto v2 = inst.GetSingleWordOperand(3);
    sval = src_function_call("dot", v1, v2);
    break;
  }
  case spv::Op::OpConvertUToF:
  case spv::Op::OpConvertSToF: {
    auto op = inst.GetSingleWordOperand(2);
    bool sat = m_saturated_conversions.count(result);
    sval = "convert_";
    sval += src_type(rtype);

    if (sat) {
      sval += "_sat";
    }

    if (m_rounding_mode_decorations.count(result)) {
      auto rmode = m_rounding_mode_decorations.at(result);
      sval += "_" + rounding_mode(rmode);
    }

    sval += "(" + var_for(op) + ")";

    break;
  }
  case spv::Op::OpSatConvertSToU: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_as(
        rtype,
        src_function_call("convert_" + src_type_signed(rtype) + "_sat", val));
    break;
  }
  case spv::Op::OpSatConvertUToS: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call_signed("convert_" + src_type(rtype) + "_sat", val);
    break;
  }
  case spv::Op::OpBitcast: {
    auto val = inst.GetSingleWordOperand(2);
    auto dstty = type_for(rtype);
    auto srcty = type_for_val(val);
    if ((srcty->kind() == Type::Kind::kPointer) ||
        (dstty->kind() == Type::Kind::kPointer)) {
      sval = src_cast(rtype, val);
    } else {
      sval = src_as(rtype, val);
    }
    break;
  }
  case spv::Op::OpSConvert: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_convert_signed(val, rtype);
    break;
  }
  case spv::Op::OpFConvert:
  case spv::Op::OpUConvert: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_convert(val, rtype);
    break;
  }
  case spv::Op::OpSelect: {
    auto cond = inst.GetSingleWordOperand(2);
    auto val_true = inst.GetSingleWordOperand(3);
    auto val_false = inst.GetSingleWordOperand(4);
    sval =
        var_for(cond) + " ? " + var_for(val_true) + " : " + var_for(val_false);
    break;
  }
  case spv::Op::OpBranch: {
    auto target = inst.GetSingleWordOperand(0);
    assign_result = false;
    src = "goto " + var_for(target);
    break;
  }
  case spv::Op::OpBranchConditional: {
    auto cond = inst.GetSingleWordOperand(0);
    auto label_true = inst.GetSingleWordOperand(1);
    auto label_false = inst.GetSingleWordOperand(2);
    assign_result = false;
    src = "if (" + var_for(cond) + ") { goto " + var_for(label_true) +
          ";} else { goto " + var_for(label_false) + ";}";
    break;
  }
  case spv::Op::OpLoopMerge:      // Nothing to do for now TODO loop controls
  case spv::Op::OpSelectionMerge: // TODO selection controls
    break;
  case spv::Op::OpPhi: // Nothing to do here, phi registers are assigned
                       // elsewhere
    assign_result = false;
    break;
  case spv::Op::OpSwitch: {
    assign_result = false;
    auto select = inst.GetSingleWordOperand(0);
    auto def = inst.GetSingleWordOperand(1);
    src = "switch (" + var_for(select) + "){";
    src += "default: goto " + var_for(def) + ";";
    for (unsigned i = 2; i < inst.NumOperands(); i += 2) {
      auto &val = inst.GetOperand(i);
      auto &target = inst.GetOperand(i + 1);
      src += "case " + std::to_string(val.AsLiteralUint64()) + ": goto " +
             var_for(target.AsId()) + ";";
    }
    src += "}";
    break;
  }
  case spv::Op::OpControlBarrier: {
    auto execution_scope = inst.GetSingleWordOperand(0);
    auto memory_scope = inst.GetSingleWordOperand(1);
    auto memory_semantics = inst.GetSingleWordOperand(2);

    auto cstmgr = m_ir->get_constant_mgr();

    auto exec_scope_cst = cstmgr->FindDeclaredConstant(execution_scope);
    if (exec_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-constant execution scope"
          << std::endl;
      return false;
    }

    if (exec_scope_cst->GetU32() != SpvScopeWorkgroup) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-workgroup execution scope"
          << std::endl;
      return false;
    }

    auto mem_scope_cst = cstmgr->FindDeclaredConstant(memory_scope);
    if (mem_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-constant memory scope"
          << std::endl;
      return false;
    }

    std::string flags;
    switch (mem_scope_cst->GetU32()) {
    case SpvScopeWorkgroup:
      flags = "CLK_LOCAL_MEM_FENCE";
      break;
    case SpvScopeDevice:
      flags = "CLK_GLOBAL_MEM_FENCE";
      break;
    default:
      std::cerr << "UNIMPLEMENTED memory scope in OpControlBarrier "
                << memory_scope << std::endl;
      return false;
    }

    auto mem_sem_cst = cstmgr->FindDeclaredConstant(memory_semantics);
    if (mem_sem_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-constant memory semantics"
          << std::endl;
      return false;
    }

    auto mem_sem = mem_sem_cst->GetU32();
    if ((mem_sem != (SpvMemorySemanticsSequentiallyConsistentMask |
                     SpvMemorySemanticsWorkgroupMemoryMask)) &&
        (mem_sem != (SpvMemorySemanticsSequentiallyConsistentMask |
                     SpvMemorySemanticsCrossWorkgroupMemoryMask))) {
      std::cerr << "UNIMPLEMENTED OpControlBarrier with memory semantics "
                << mem_sem << std::endl;
      return false;
    }

    src = src_function_call("barrier", flags);
    break;
  }
  case spv::Op::OpGroupAsyncCopy: {
    auto execution_scope = inst.GetSingleWordOperand(2);
    auto dst_ptr = inst.GetSingleWordOperand(3);
    auto src_ptr = inst.GetSingleWordOperand(4);
    auto num_elems = inst.GetSingleWordOperand(5);
    auto stride = inst.GetSingleWordOperand(6);
    auto event = inst.GetSingleWordOperand(7);

    auto cstmgr = m_ir->get_constant_mgr();

    auto exec_scope_cst = cstmgr->FindDeclaredConstant(execution_scope);
    if (exec_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpGroupAsyncCopy with non-constant execution scope"
          << execution_scope << std::endl;
      return false;
    }

    if (exec_scope_cst->GetU32() != SpvScopeWorkgroup) {
      std::cerr
          << "UNIMPLEMENTED OpGroupAsyncCopy with non-workgroup execution scope"
          << std::endl;
      return false;
    }

    auto stride_cst = cstmgr->FindDeclaredConstant(stride);

    if ((stride_cst != nullptr) && (stride_cst->GetZeroExtendedValue() == 1)) {
      sval = src_function_call("async_work_group_copy", dst_ptr, src_ptr,
                               num_elems, event);
    } else {
      sval = src_function_call("async_work_group_strided_copy", dst_ptr,
                               src_ptr, num_elems, stride, event);
    }

    break;
  }
  case spv::Op::OpGroupWaitEvents: {
    auto execution_scope = inst.GetSingleWordOperand(0);
    auto num_events = inst.GetSingleWordOperand(1);
    auto event_list = inst.GetSingleWordOperand(2);

    auto cstmgr = m_ir->get_constant_mgr();

    auto exec_scope_cst = cstmgr->FindDeclaredConstant(execution_scope);
    if (exec_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpGroupWaitEvents with non-constant execution scope"
          << std::endl;
      return false;
    }

    if (exec_scope_cst->GetU32() != SpvScopeWorkgroup) {
      std::cerr << "UNIMPLEMENTED OpGroupWaitEvents with non-workgroup "
                   "execution scope"
                << std::endl;
      return false;
    }

    src = src_function_call("wait_group_events", num_events, event_list);
    assign_result = false;
    break;
  }
  case spv::Op::OpExtInst: {
    assign_result = false;
    if (!translate_extended_instruction(inst, src)) {
      return false;
    }
    break;
  }
  default:
    std::cerr << "UNIMPLEMENTED instruction " << opcode << std::endl;
    return false;
  }

  if (boolean_result) {
    m_boolean_src_types[result] = boolean_result_src_type;
  }

  if ((result != 0) && assign_result) {
    src = src_var_decl(result);
    src += " = " + sval;
  }

  return true;
}

std::string translator::translate_binop(const Instruction &inst) const {
  static std::unordered_map<spv::Op, const std::string> binops = {
      {spv::Op::OpFMul, "*"},
      {spv::Op::OpFDiv, "/"},
      {spv::Op::OpFAdd, "+"},
      {spv::Op::OpFSub, "-"},
      {spv::Op::OpISub, "-"},
      {spv::Op::OpIAdd, "+"},
      {spv::Op::OpIMul, "*"},
      {spv::Op::OpUDiv, "/"},
      {spv::Op::OpUMod, "%"},
      {spv::Op::OpULessThan, "<"},
      {spv::Op::OpULessThanEqual, "<="},
      {spv::Op::OpUGreaterThan, ">"},
      {spv::Op::OpUGreaterThanEqual, ">="},
      {spv::Op::OpLogicalEqual, "=="},
      {spv::Op::OpLogicalNotEqual, "!="},
      {spv::Op::OpIEqual, "=="},
      {spv::Op::OpINotEqual, "!="},
      {spv::Op::OpBitwiseOr, "|"},
      {spv::Op::OpBitwiseXor, "^"},
      {spv::Op::OpBitwiseAnd, "&"},
      {spv::Op::OpLogicalOr, "||"},
      {spv::Op::OpLogicalAnd, "&&"},
      {spv::Op::OpVectorTimesScalar, "*"},
      {spv::Op::OpShiftLeftLogical, "<<"},
      {spv::Op::OpShiftRightLogical, ">>"},
      {spv::Op::OpFOrdEqual, "=="},
      {spv::Op::OpFUnordEqual, "=="},
      {spv::Op::OpFOrdNotEqual, "!="},
      {spv::Op::OpFUnordNotEqual, "!="},
      {spv::Op::OpFOrdLessThan, "<"},
      {spv::Op::OpFUnordLessThan, "<"},
      {spv::Op::OpFOrdGreaterThan, ">"},
      {spv::Op::OpFUnordGreaterThan, ">"},
      {spv::Op::OpFOrdLessThanEqual, "<="},
      {spv::Op::OpFUnordLessThanEqual, "<="},
      {spv::Op::OpFOrdGreaterThanEqual, ">="},
      {spv::Op::OpFUnordGreaterThanEqual, ">="},
  };

  auto v1 = inst.GetSingleWordOperand(2);
  auto v2 = inst.GetSingleWordOperand(3);

  auto &srcop = binops.at(inst.opcode());

  return var_for(v1) + " " + srcop + " " + var_for(v2);
}

std::string translator::translate_binop_signed(const Instruction &inst) const {
  static std::unordered_map<spv::Op, const std::string> binops = {
      {spv::Op::OpSDiv, "/"},
      {spv::Op::OpSRem, "%"},
      {spv::Op::OpShiftRightArithmetic, ">>"},
      {spv::Op::OpSLessThan, "<"},
      {spv::Op::OpSLessThanEqual, "<="},
      {spv::Op::OpSGreaterThan, ">"},
      {spv::Op::OpSGreaterThanEqual, ">="},
  };

  auto v1 = inst.GetSingleWordOperand(2);
  auto v2 = inst.GetSingleWordOperand(3);

  auto &srcop = binops.at(inst.opcode());

  return src_as_signed(v1) + " " + srcop + " " + src_as_signed(v2);
}

std::string translator::builtin_vector_extract(uint32_t id, uint32_t idx, bool constant) const {
  std::string arg;
  if (constant) {
    arg = std::to_string(idx);
  } else {
    arg = var_for(idx);
  }

  switch (m_builtin_values.at(id)) {
  case SpvBuiltInGlobalInvocationId:
    return src_function_call("get_global_id", arg);
  case SpvBuiltInGlobalOffset:
    return src_function_call("get_global_offset", arg);
  case SpvBuiltInGlobalSize:
    return src_function_call("get_global_size", arg);
  case SpvBuiltInWorkgroupId:
    return src_function_call("get_group_id", arg);
  case SpvBuiltInWorkgroupSize:
    return src_function_call("get_local_size", arg);
  case SpvBuiltInLocalInvocationId:
    return src_function_call("get_local_id", arg);
  case SpvBuiltInNumWorkgroups:
    return src_function_call("get_num_groups", arg);
  default:
    std::cerr << "UNIMPLEMENTED built-in in builtin_vector_extract" << std::endl;
    return "UNIMPLEMENTED";
  }
}
