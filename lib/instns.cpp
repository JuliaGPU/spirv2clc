
bool translator_impl::emit_access_chain(const Instruction &inst,
                                        bool ptr_variant,
                                        c::expr_ref &result) const {
  auto base = inst.GetSingleWordOperand(2);
  const Type *cty = type_for_val(base)->AsPointer()->pointee_type();
  unsigned i;
  if (ptr_variant) {
    // The Ptr* variants take a leading Element index that does pointer
    // arithmetic on the base (scaled by the pointee size, which is correct now
    // that pointer-to-array is a pointer-to-wrapper). The index is signed, so
    // emit it as such: a -1 here (1-based indexing) must subtract one element,
    // not add 0xFFFF... and overflow the pointer.
    auto elem = inst.GetSingleWordOperand(3);
    if (inst.opcode() == spv::Op::OpInBoundsPtrAccessChain) {
      result = c::address_of(c::index(value(base), signed_index(elem)));
    } else {
      // Only the InBounds variant promises the result stays within the
      // base's object. C pointer arithmetic always claims that (compilers
      // lower it to inbounds GEPs, making an out-of-bounds intermediate --
      // e.g. `p - 1 + i` from 1-based indexing -- poison), so the plain
      // variant must go through integer arithmetic, which claims nothing.
      // Interior struct/array steps below still use typed lvalues: C cannot
      // express an out-of-bounds member walk, and producers don't emit them.
      auto intptr = src_pointer_int_type();
      auto offset = c::binary("*", c::cast(intptr, signed_index(elem)),
                              c::call("sizeof", {c::deref(value(base))}));
      result = cast_to(type_id_for(base),
                       c::binary("+", c::cast(intptr, value(base)), offset));
    }
    i = 4;
  } else {
    // The plain variants' first index walks into the pointee directly.
    result = value(base);
    i = 3;
  }
  bool descended_aggregate = false;
  for (; i < inst.NumOperands(); i++) {
    descended_aggregate = true;
    auto idx = inst.GetSingleWordOperand(i);
    switch (cty->kind()) {
    case Type::Kind::kArray:
      // Arrays are struct-wrapped; index through the 'e' member.
      result = c::address_of(
          c::index(c::member(c::deref(result), "e"), signed_index(idx)));
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
      result = c::address_of(
          c::member(c::deref(result), "m" + std::to_string(struct_idx)));
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
  // slot holds the pointer in storage form (see src_storage_type). Reinterpret
  // the slot's address as the address of a pointer, which the storage form
  // has the size and bit pattern of, so loads and stores through it see the
  // pointer.
  if (descended_aggregate && cty->kind() == Type::Kind::kPointer) {
    result = cast_to(inst.type_id(), result);
  }
  return true;
}

std::string translator_impl::atomic_builtin(const std::string &op,
                                            uint32_t ptr) const {
  auto pointee = type_for_val(ptr)->AsPointer()->pointee_type();
  bool is64 = pointee->kind() == Type::Kind::kInteger &&
              pointee->AsInteger()->width() == 64;
  // atom_* (cl_khr_int64_*_atomics) for 64-bit, atomic_* (core) otherwise.
  return (is64 ? "atom_" : "atomic_") + op;
}

c::expr_ref translator_impl::atomic_c11_pointer(uint32_t ptr,
                                                bool is_signed) const {
  auto ptrty = type_for_val(ptr)->AsPointer();
  auto pointee = ptrty->pointee_type();
  std::string ty;
  if (pointee->kind() == Type::Kind::kFloat) {
    ty = pointee->AsFloat()->width() == 64 ? "atomic_double" : "atomic_float";
  } else {
    bool is64 = pointee->AsInteger()->width() == 64;
    ty = is64 ? (is_signed ? "atomic_long" : "atomic_ulong")
              : (is_signed ? "atomic_int" : "atomic_uint");
  }
  std::string as =
      address_space_qualifier(static_cast<uint32_t>(ptrty->storage_class()));
  return c::cast("volatile " + (as.empty() ? "" : as + " ") + ty + "*",
                 value(ptr));
}

std::optional<uint32_t>
translator_impl::constant_operand(uint32_t id, const char *what) const {
  auto cst = m_ir->get_constant_mgr()->FindDeclaredConstant(id);
  if (cst == nullptr) {
    std::cerr << "UNIMPLEMENTED non-constant " << what << std::endl;
    return std::nullopt;
  }
  return cst->GetU32();
}

c::expr_ref translator_impl::memory_scope(uint32_t scope) const {
  switch (scope) {
  case SpvScopeCrossDevice:
    return c::name("memory_scope_all_svm_devices");
  case SpvScopeDevice:
    return c::name("memory_scope_device");
  case SpvScopeWorkgroup:
  // A wider scope is always correct, and these narrower ones are restricted:
  // memory_scope_work_item to image fences, memory_scope_sub_group to devices
  // with sub-groups.
  case SpvScopeSubgroup:
  case SpvScopeInvocation:
    return c::name("memory_scope_work_group");
  default:
    std::cerr << "UNIMPLEMENTED memory scope " << scope << std::endl;
    return nullptr;
  }
}

namespace {

// The memory orders of OpenCL C 2.0, which match those of SPIR-V.
enum class memory_order { relaxed, acquire, release, acq_rel, seq_cst };

memory_order order_of(uint32_t mem_sem) {
  if (mem_sem & SpvMemorySemanticsSequentiallyConsistentMask) {
    return memory_order::seq_cst;
  }
  if (mem_sem & SpvMemorySemanticsAcquireReleaseMask) {
    return memory_order::acq_rel;
  }
  if (mem_sem & SpvMemorySemanticsReleaseMask) {
    return memory_order::release;
  }
  if (mem_sem & SpvMemorySemanticsAcquireMask) {
    return memory_order::acquire;
  }
  return memory_order::relaxed;
}

c::expr_ref spell(memory_order order) {
  switch (order) {
  case memory_order::relaxed:
    return c::name("memory_order_relaxed");
  case memory_order::acquire:
    return c::name("memory_order_acquire");
  case memory_order::release:
    return c::name("memory_order_release");
  case memory_order::acq_rel:
    return c::name("memory_order_acq_rel");
  case memory_order::seq_cst:
    return c::name("memory_order_seq_cst");
  }
  return nullptr;
}

// OpenCL C rejects orders that don't apply to an operation, e.g. release on a
// load, which SPIR-V allows; drop the half that has no effect.
memory_order without_release(memory_order order) {
  switch (order) {
  case memory_order::release:
    return memory_order::relaxed;
  case memory_order::acq_rel:
    return memory_order::acquire;
  default:
    return order;
  }
}

memory_order without_acquire(memory_order order) {
  switch (order) {
  case memory_order::acquire:
    return memory_order::relaxed;
  case memory_order::acq_rel:
    return memory_order::release;
  default:
    return order;
  }
}

// How many work-items a scope spans, relative to the others.
int scope_width(uint32_t scope) {
  switch (scope) {
  case SpvScopeInvocation:
    return 0;
  case SpvScopeSubgroup:
    return 1;
  case SpvScopeWorkgroup:
    return 2;
  case SpvScopeDevice:
    return 3;
  default: // CrossDevice, and scopes OpenCL C doesn't know
    return 4;
  }
}

// The weakest order at least as strong as both `a` and `b`.
memory_order join(memory_order a, memory_order b) {
  if (a == b || b == memory_order::relaxed) {
    return a;
  }
  if (a == memory_order::relaxed) {
    return b;
  }
  if (a == memory_order::seq_cst || b == memory_order::seq_cst) {
    return memory_order::seq_cst;
  }
  return memory_order::acq_rel;
}

} // namespace

c::expr_ref translator_impl::fence_flags(uint32_t mem_sem) const {
  c::expr_ref flags;
  auto add = [&](const char *f) {
    flags = flags ? c::binary("|", flags, c::name(f)) : c::name(f);
  };
  if (mem_sem & SpvMemorySemanticsWorkgroupMemoryMask) {
    add("CLK_LOCAL_MEM_FENCE");
  }
  if (mem_sem & SpvMemorySemanticsCrossWorkgroupMemoryMask) {
    add("CLK_GLOBAL_MEM_FENCE");
  }
  // CLK_IMAGE_MEM_FENCE is OpenCL C 2.0. Before that, a kernel can't both write
  // and read an image, so there are no image accesses to order.
  if ((mem_sem & SpvMemorySemanticsImageMemoryMask) &&
      m_opencl_c_version >= 200) {
    add("CLK_IMAGE_MEM_FENCE");
  }
  // Semantics with only an ordering (or subgroup memory, which has no OpenCL C
  // fence flag) fence no named address space: a pure execution barrier / no-op
  // fence, spelled with flags 0.
  return flags ? flags : c::literal("0");
}

c::expr_ref translator_impl::dereference(uint32_t ptr,
                                         const MemoryAccess &access) {
  auto ptrty = type_for_val(ptr)->AsPointer();
  auto pointee = m_ir->get_type_mgr()->GetId(ptrty->pointee_type());
  bool vol = access.mask & SpvMemoryAccessVolatileMask;
  if (!vol && !is_underaligned(pointee, access)) {
    return c::deref(value(ptr));
  }
  auto as =
      address_space_qualifier(static_cast<uint32_t>(ptrty->storage_class()));
  return c::deref(c::cast(std::string(vol ? "volatile " : "") +
                              src_access_pointee(pointee, access) + " " + as +
                              "*",
                          value(ptr)));
}

bool translator_impl::translate_instruction(const Instruction &inst,
                                            function_builder &fb) {
  auto opcode = inst.opcode();
  auto rtype = inst.type_id();
  auto result = inst.result_id();

  c::expr_ref val;
  bool assign_result = true;

  switch (opcode) {
  case spv::Op::OpUndef:
    val = null_constant(rtype);
    break;
  case spv::Op::OpUnreachable: // TODO trigger crash? end invocation?
    break;
  case spv::Op::OpReturn:
    fb.statement("return");
    break;
  case spv::Op::OpReturnValue:
    fb.statement("return " + c::print(value(inst.GetSingleWordOperand(0))));
    break;
  case spv::Op::OpFunctionCall: {
    auto func = inst.GetSingleWordOperand(2);
    // The callee's by-value (ByVal) parameters are emitted by value in its
    // signature, so dereference the matching pointer argument at the call site.
    std::vector<uint32_t> callee_params;
    if (auto *callee = m_ir->GetFunction(func)) {
      callee->ForEachParam(
          [&](const Instruction *p) { callee_params.push_back(p->result_id()); });
    }
    std::vector<c::expr_ref> args;
    for (unsigned i = 3; i < inst.NumOperands(); i++) {
      auto param = inst.GetSingleWordOperand(i);
      unsigned argidx = i - 3;
      if (argidx < callee_params.size() &&
          m_byval_params.count(callee_params[argidx])) {
        args.push_back(c::deref(value(param)));
      } else {
        args.push_back(value(param));
      }
    }
    // Pass down the local-pointer parameters the callee expects for the
    // module-scope Workgroup variables it references (see
    // emit_function_signature / compute_workgroup_params). Each is in scope in
    // the caller either as the entry kernel's storage pointer or as the caller's
    // own threaded-through parameter.
    auto wgit = m_function_workgroup_params.find(func);
    if (wgit != m_function_workgroup_params.end()) {
      for (auto wgvar : wgit->second) {
        args.push_back(value(wgvar));
      }
    }
    val = c::call(name_of(func), std::move(args));
    if (type_for(rtype)->kind() == Type::Kind::kVoid) {
      assign_result = false;
      fb.expression(val);
    }
    break;
  }
  case spv::Op::OpCopyObject:
    val = value(inst.GetSingleWordOperand(2));
    break;
  case spv::Op::OpLifetimeStart:
  case spv::Op::OpLifetimeStop:
    break;
  case spv::Op::OpVariable: {
    assign_result = false;
    auto varty = type_for(rtype)->AsPointer()->pointee_type();
    auto &storagename = derived_name(result, "_storage");
    // The variable's value is a pointer to its storage.
    auto tymgr = m_ir->get_type_mgr();
    fb.declare(src_type_memory_object_declaration(tymgr->GetId(varty), result,
                                                  storagename),
               inst.NumOperands() == 4 ? value(inst.GetSingleWordOperand(3))
                                       : nullptr);
    fb.declare(src_var_decl(result), c::address_of(c::name(storagename)));
    break;
  }
  case spv::Op::OpLoad: {
    auto ptr = inst.GetSingleWordOperand(2);
    if (m_builtin_variables.count(ptr)) {
      // Built-in variables are queries, bound to the load rather than
      // declared.
      auto builtin = m_builtin_variables.at(ptr);
      m_builtin_values[result] = builtin;
      if (type_for(rtype)->kind() == Type::Kind::kVector) {
        m_bindings[result] = builtin_vector(result);
      } else {
        m_bindings[result] = builtin_scalar(builtin);
      }
      assign_result = false;
    } else {
      val = dereference(ptr, memory_access_operands(inst, 3));
    }
    break;
  }
  case spv::Op::OpStore: {
    auto ptr = inst.GetSingleWordOperand(0);
    auto stored = inst.GetSingleWordOperand(1);
    fb.assign(dereference(ptr, memory_access_operands(inst, 2)), value(stored));
    break;
  }
  case spv::Op::OpCopyMemory: {
    auto target = inst.GetSingleWordOperand(0);
    auto source = inst.GetSingleWordOperand(1);
    assign_result = false;
    // Same pointee type, so copy by value (arrays are struct-wrapped, so this
    // is a legal aggregate assignment). A cast cannot change address space, so
    // reinterpret the source to the target's pointee within its own space and
    // let the assignment cross address spaces. The first memory operand
    // applies to the target, the second (SPIR-V 1.4+, or the first again when
    // absent) to the source.
    auto src_storage = static_cast<uint32_t>(
        type_for_val(source)->AsPointer()->storage_class());
    auto tgt_pointee = pointee_type_id(target);
    auto tgt_access = memory_access_operands(inst, 2);
    auto src_access = memory_access_operands(inst, tgt_access.next);
    if (inst.NumOperands() <= tgt_access.next) {
      src_access = tgt_access;
    }
    bool src_vol = src_access.mask & SpvMemoryAccessVolatileMask;
    auto src_ptr_type = std::string(src_vol ? "volatile " : "") +
                        src_access_pointee(tgt_pointee, src_access) + " " +
                        address_space_qualifier(src_storage) + "*";
    fb.assign(dereference(target, tgt_access),
              c::deref(c::cast(src_ptr_type, value(source))));
    break;
  }
  case spv::Op::OpCopyMemorySized: {
    auto target = inst.GetSingleWordOperand(0);
    auto source = inst.GetSingleWordOperand(1);
    auto size = inst.GetSingleWordOperand(2);
    assign_result = false;
    // Explicit byte count, and the operands may have differently-sized pointee
    // types, so a typed assignment would copy the wrong amount. Copy byte-wise
    // instead, casting each pointer to a byte pointer in its own address space;
    // optimizers fold the small loop back into an efficient copy. Byte
    // accesses cannot be under-aligned, but a Volatile memory operand must
    // survive (the first applies to the target, a second, if present, to the
    // source).
    auto tgt_access = memory_access_operands(inst, 3);
    auto src_access = memory_access_operands(inst, tgt_access.next);
    if (inst.NumOperands() <= tgt_access.next) {
      src_access = tgt_access;
    }
    auto byte_pointer = [this](uint32_t ptr, const MemoryAccess &access) {
      auto as = address_space_qualifier(static_cast<uint32_t>(
          type_for_val(ptr)->AsPointer()->storage_class()));
      bool vol = access.mask & SpvMemoryAccessVolatileMask;
      return c::cast(std::string(vol ? "volatile " : "") + "uchar " + as + "*",
                     value(ptr));
    };
    // The counter is allocated like any other name, so that an operand can't
    // be shadowed by it.
    auto counter = m_name_allocator.allocate("_i");
    auto i = c::name(counter);
    fb.statement(
        "for (ulong " + counter + " = 0; " + counter + " < " +
        c::print(value(size)) + "; ++" + counter + ") " +
        c::print(c::binary("=", c::index(byte_pointer(target, tgt_access), i),
                           c::index(byte_pointer(source, src_access), i))));
    break;
  }
  case spv::Op::OpConvertPtrToU:
  case spv::Op::OpConvertUToPtr:
  case spv::Op::OpPtrCastToGeneric:
  case spv::Op::OpGenericCastToPtr:
    // The result pointer type already carries the destination address space
    // (e.g. `uint generic*` / `uint global*`), so a plain cast suffices.
    val = cast_to(rtype, value(inst.GetSingleWordOperand(2)));
    break;
  case spv::Op::OpIAddCarry:
  case spv::Op::OpISubBorrow: {
    // Result is a struct { low, carry/borrow } whose members share the operand
    // type. For unsigned add the carry is (sum < a); for sub the borrow is
    // (a < b). Emitted as a struct initializer.
    auto a = inst.GetSingleWordOperand(2);
    auto b = inst.GetSingleWordOperand(3);
    auto mt = type_id_for(a);
    if (opcode == spv::Op::OpIAddCarry) {
      auto sum = cast_to(mt, c::binary("+", value(a), value(b)));
      val = c::init_list({sum, cast_to(mt, c::binary("<", sum, value(a)))});
    } else {
      val = c::init_list({cast_to(mt, c::binary("-", value(a), value(b))),
                          cast_to(mt, c::binary("<", value(a), value(b)))});
    }
    break;
  }
  case spv::Op::OpPtrAccessChain:
  case spv::Op::OpInBoundsPtrAccessChain:
    if (!emit_access_chain(inst, /*ptr_variant=*/true, val)) {
      return false;
    }
    break;
  case spv::Op::OpAccessChain:
  case spv::Op::OpInBoundsAccessChain:
    if (!emit_access_chain(inst, /*ptr_variant=*/false, val)) {
      return false;
    }
    break;
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
    auto coord_val = value(coord);
    if (!is_float_coord) {
      coord_val = c::call("as_int2", {coord_val});
    }
    auto &image_sampler = m_sampled_images.at(sampledimage);
    // FIXME i vs. ui
    val = c::call(
        is_float ? "read_imagef" : "read_imagei",
        {value(image_sampler.first), value(image_sampler.second), coord_val});
    if (!is_float) {
      val = c::call("as_uint4", {val});
    }
    // TODO check Lod
    break;
  }
  case spv::Op::OpImageQuerySizeLod: {
    auto image = inst.GetSingleWordOperand(2);
    // auto lod = inst.GetSingleWordOperand(3); // FIXME validate
    auto dim = type_for_val(image)->AsImage()->dim();
    std::vector<c::expr_ref> dims = {
        c::call("get_image_width", {value(image)})};
    if ((dim == spv::Dim::Dim2D) || (dim == spv::Dim::Dim3D)) {
      dims.push_back(c::call("get_image_height", {value(image)}));
    }
    if (dim == spv::Dim::Dim3D) {
      dims.push_back(c::call("get_image_depth", {value(image)}));
    }
    val = c::vector_literal(src_type(rtype), dims);
    break;
  }
  case spv::Op::OpAtomicIIncrement:
  case spv::Op::OpAtomicIDecrement:
  case spv::Op::OpAtomicAnd:
  case spv::Op::OpAtomicExchange:
  case spv::Op::OpAtomicIAdd:
  case spv::Op::OpAtomicISub:
  case spv::Op::OpAtomicOr:
  case spv::Op::OpAtomicSMax:
  case spv::Op::OpAtomicSMin:
  case spv::Op::OpAtomicUMax:
  case spv::Op::OpAtomicUMin:
  case spv::Op::OpAtomicXor:
  case spv::Op::OpAtomicFAddEXT:
  case spv::Op::OpAtomicFMinEXT:
  case spv::Op::OpAtomicFMaxEXT: {
    auto ptr = inst.GetSingleWordOperand(2);
    auto scope = constant_operand(inst.GetSingleWordOperand(3), "atomic scope");
    auto mem_sem =
        constant_operand(inst.GetSingleWordOperand(4), "atomic semantics");
    if (!scope || !mem_sem) {
      return false;
    }
    bool is_signed =
        opcode == spv::Op::OpAtomicSMax || opcode == spv::Op::OpAtomicSMin;
    // Increment and decrement have no operand: they add or subtract one.
    bool has_operand = opcode != spv::Op::OpAtomicIIncrement &&
                       opcode != spv::Op::OpAtomicIDecrement;
    c::expr_ref operand;
    if (!has_operand) {
      operand = cast_to(rtype, c::literal("1"));
    } else if (is_signed) {
      operand = as_signed(inst.GetSingleWordOperand(5));
    } else {
      operand = value(inst.GetSingleWordOperand(5));
    }
    if (m_opencl_c_version >= 200) {
      static std::unordered_map<spv::Op, const char *> fns{
          {spv::Op::OpAtomicIIncrement, "atomic_fetch_add_explicit"},
          {spv::Op::OpAtomicIDecrement, "atomic_fetch_sub_explicit"},
          {spv::Op::OpAtomicAnd, "atomic_fetch_and_explicit"},
          {spv::Op::OpAtomicExchange, "atomic_exchange_explicit"},
          {spv::Op::OpAtomicIAdd, "atomic_fetch_add_explicit"},
          {spv::Op::OpAtomicISub, "atomic_fetch_sub_explicit"},
          {spv::Op::OpAtomicOr, "atomic_fetch_or_explicit"},
          {spv::Op::OpAtomicSMax, "atomic_fetch_max_explicit"},
          {spv::Op::OpAtomicSMin, "atomic_fetch_min_explicit"},
          {spv::Op::OpAtomicUMax, "atomic_fetch_max_explicit"},
          {spv::Op::OpAtomicUMin, "atomic_fetch_min_explicit"},
          {spv::Op::OpAtomicXor, "atomic_fetch_xor_explicit"},
          {spv::Op::OpAtomicFAddEXT, "atomic_fetch_add_explicit"},
          {spv::Op::OpAtomicFMinEXT, "atomic_fetch_min_explicit"},
          {spv::Op::OpAtomicFMaxEXT, "atomic_fetch_max_explicit"},
      };
      auto scope_arg = memory_scope(*scope);
      if (!scope_arg) {
        return false;
      }
      val =
          c::call(fns.at(opcode), {atomic_c11_pointer(ptr, is_signed), operand,
                                   spell(order_of(*mem_sem)), scope_arg});
      if (is_signed) {
        val = as_type(rtype, val);
      }
      break;
    }
    // The OpenCL C 1.x atomics are relaxed. OpenCL C 1.x has no scopes: they
    // are atomic for every work-item accessing the memory, which is as wide as
    // any scope gets without shared virtual memory.
    if (order_of(*mem_sem) != memory_order::relaxed) {
      std::cerr << "UNIMPLEMENTED: ordered atomics require OpenCL C 2.0 "
                   "(targeting "
                << opencl_c_version_str(m_opencl_c_version) << ").\n";
      return false;
    }
    if (opcode == spv::Op::OpAtomicFAddEXT ||
        opcode == spv::Op::OpAtomicFMinEXT ||
        opcode == spv::Op::OpAtomicFMaxEXT) {
      std::cerr << "UNIMPLEMENTED: floating-point atomics require OpenCL C "
                   "2.0 (targeting "
                << opencl_c_version_str(m_opencl_c_version) << ").\n";
      return false;
    }
    static std::unordered_map<spv::Op, const char *> fns{
        {spv::Op::OpAtomicIIncrement, "inc"},
        {spv::Op::OpAtomicIDecrement, "dec"},
        {spv::Op::OpAtomicAnd, "and"},
        {spv::Op::OpAtomicExchange, "xchg"},
        {spv::Op::OpAtomicIAdd, "add"},
        {spv::Op::OpAtomicISub, "sub"},
        {spv::Op::OpAtomicOr, "or"},
        {spv::Op::OpAtomicSMax, "max"},
        {spv::Op::OpAtomicSMin, "min"},
        {spv::Op::OpAtomicUMax, "max"},
        {spv::Op::OpAtomicUMin, "min"},
        {spv::Op::OpAtomicXor, "xor"},
    };
    auto fn = atomic_builtin(fns.at(opcode), ptr);
    if (!has_operand) {
      val = call_values(fn, {ptr});
    } else if (is_signed) {
      // Integers are spelled unsigned, so the signed comparison needs the
      // signed overload: reinterpret the operands and convert the result back.
      val = as_type(
          rtype,
          c::call(fn, {cast_to_signed(type_id_for(ptr), value(ptr)), operand}));
    } else {
      val = c::call(fn, {value(ptr), operand});
    }
    break;
  }
  // SPIR-V gives the weak exchange the semantics of the strong one.
  case spv::Op::OpAtomicCompareExchangeWeak:
  case spv::Op::OpAtomicCompareExchange: {
    auto ptr = inst.GetSingleWordOperand(2);
    auto scope = constant_operand(inst.GetSingleWordOperand(3), "atomic scope");
    auto sem_equal =
        constant_operand(inst.GetSingleWordOperand(4), "atomic semantics");
    auto sem_unequal =
        constant_operand(inst.GetSingleWordOperand(5), "atomic semantics");
    auto desired = inst.GetSingleWordOperand(6);
    auto cmp = inst.GetSingleWordOperand(7);
    if (!scope || !sem_equal || !sem_unequal) {
      return false;
    }
    if (m_opencl_c_version >= 200) {
      auto scope_arg = memory_scope(*scope);
      if (!scope_arg) {
        return false;
      }
      // OpenCL C requires the failure order to be no stronger than the success
      // order; SPIR-V doesn't (it only forbids a releasing failure order).
      auto failure = order_of(*sem_unequal);
      auto success = join(order_of(*sem_equal), failure);
      // The expected value is passed by address and overwritten with the value
      // found, which is the result: so the result variable can hold it.
      fb.declare(src_var_decl(result), value(cmp));
      fb.expression(c::call(
          "atomic_compare_exchange_strong_explicit",
          {atomic_c11_pointer(ptr), c::address_of(c::name(name_of(result))),
           value(desired), spell(success), spell(failure), scope_arg}));
      assign_result = false;
      break;
    }
    if (order_of(*sem_equal) != memory_order::relaxed ||
        order_of(*sem_unequal) != memory_order::relaxed) {
      std::cerr << "UNIMPLEMENTED: ordered atomics require OpenCL C 2.0 "
                   "(targeting "
                << opencl_c_version_str(m_opencl_c_version) << ").\n";
      return false;
    }
    val = call_values(atomic_builtin("cmpxchg", ptr), {ptr, cmp, desired});
    break;
  }
  case spv::Op::OpAtomicLoad:
  case spv::Op::OpAtomicStore: {
    // OpenCL C 1.x has no atomic loads and stores.
    if (m_opencl_c_version < 200) {
      std::cerr << "UNIMPLEMENTED: atomic load/store require OpenCL C 2.0 "
                   "(targeting "
                << opencl_c_version_str(m_opencl_c_version) << ").\n";
      return false;
    }
    bool is_load = opcode == spv::Op::OpAtomicLoad;
    auto ptr = inst.GetSingleWordOperand(is_load ? 2 : 0);
    auto scope = constant_operand(inst.GetSingleWordOperand(is_load ? 3 : 1),
                                  "atomic scope");
    auto mem_sem = constant_operand(inst.GetSingleWordOperand(is_load ? 4 : 2),
                                    "atomic semantics");
    if (!scope || !mem_sem) {
      return false;
    }
    auto scope_arg = memory_scope(*scope);
    if (!scope_arg) {
      return false;
    }
    if (is_load) {
      val = c::call("atomic_load_explicit",
                    {atomic_c11_pointer(ptr),
                     spell(without_release(order_of(*mem_sem))), scope_arg});
    } else {
      auto stored = inst.GetSingleWordOperand(3);
      fb.expression(
          c::call("atomic_store_explicit",
                  {atomic_c11_pointer(ptr), value(stored),
                   spell(without_acquire(order_of(*mem_sem))), scope_arg}));
    }
    break;
  }
  case spv::Op::OpCompositeExtract: {
    auto comp = inst.GetSingleWordOperand(2);
    if (m_builtin_values.count(comp)) {
      // Built-in values are vectors of scalars, so there is a single index.
      val = builtin_vector_extract(
          comp, c::literal(std::to_string(inst.GetSingleWordOperand(3))));
      break;
    }
    c::expr_ref member;
    uint32_t leaf_tyid, parent_tyid;
    if (!composite_member(inst, 3, type_id_for(comp), value(comp), member,
                          leaf_tyid, parent_tyid)) {
      return false;
    }
    val = from_component(parent_tyid, leaf_tyid, member);
    break;
  }
  case spv::Op::OpCompositeInsert: {
    auto object = inst.GetSingleWordOperand(2);
    auto composite = inst.GetSingleWordOperand(3);

    c::expr_ref member;
    uint32_t leaf_tyid, parent_tyid;
    if (!composite_member(inst, 4, rtype, c::name(name_of(result)), member,
                          leaf_tyid, parent_tyid)) {
      return false;
    }

    assign_result = false;
    fb.declare(src_var_decl(result), value(composite));
    fb.assign(member, to_component(parent_tyid, leaf_tyid, value(object)));
    break;
  }
  case spv::Op::OpCompositeConstruct: {
    std::vector<c::expr_ref> elems;
    for (unsigned i = 2; i < inst.NumOperands(); i++) {
      auto mem = inst.GetSingleWordOperand(i);
      elems.push_back(to_component(rtype, type_id_for(mem), value(mem)));
    }
    val = c::init_list(std::move(elems));
    // Arrays are struct-wrapped, so their elements live in the 'e' member and
    // need an extra brace level: { { e0, e1, ... } }.
    if (type_for(rtype)->kind() == Type::Kind::kArray) {
      val = c::init_list({val});
    }
    break;
  }
  case spv::Op::OpVectorExtractDynamic: {
    // ((elemtype*)&vec)[elem]
    auto vec = inst.GetSingleWordOperand(2);
    auto idx = inst.GetSingleWordOperand(3);
    if (m_builtin_values.count(vec)) {
      val = builtin_vector_extract(vec, value(idx));
    } else {
      val = c::index(c::cast(src_vector_element_type(type_id_for(vec)) + "*",
                             c::address_of(materialize(vec, fb))),
                     value(idx));
    }
    break;
  }
  case spv::Op::OpVectorInsertDynamic: {
    auto vec = inst.GetSingleWordOperand(2);
    auto comp = inst.GetSingleWordOperand(3);
    auto idx = inst.GetSingleWordOperand(4);
    assign_result = false;
    auto element = c::index(c::cast(src_vector_element_type(rtype) + "*",
                                    c::address_of(c::name(name_of(result)))),
                            value(idx));
    fb.declare(src_var_decl(result), value(vec));
    fb.assign(element, to_component(rtype, type_id_for(comp), value(comp)));
    break;
  }
  case spv::Op::OpVectorShuffle: {
    auto v1 = inst.GetSingleWordOperand(2);
    auto v2 = inst.GetSingleWordOperand(3);
    auto n1 = type_for_val(v1)->AsVector()->element_count();
    std::vector<c::expr_ref> comps;
    for (unsigned i = 4; i < inst.NumOperands(); i++) {
      auto comp = inst.GetSingleWordOperand(i);
      if (comp == 0xFFFFFFFFU) {
        comps.push_back(c::literal("0"));
      } else if (comp >= n1) {
        comps.push_back(vector_component(v2, comp - n1));
      } else {
        comps.push_back(vector_component(v1, comp));
      }
    }
    val = c::vector_literal(src_type(rtype), std::move(comps));
    break;
  }
  case spv::Op::OpSDiv:
  case spv::Op::OpSRem:
  case spv::Op::OpShiftRightArithmetic:
    val = as_type(rtype, translate_binop_signed(inst));
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
    val = translate_binop(inst);
    break;
  case spv::Op::OpFRem:
    // OpFRem: remainder with the sign of operand 1 -- exactly C/OpenCL fmod().
    val = call_values(
        "fmod", {inst.GetSingleWordOperand(2), inst.GetSingleWordOperand(3)});
    break;
  case spv::Op::OpFMod: {
    // OpFMod: remainder with the sign of operand 2 (the divisor) -- this is the
    // floored modulo "a - b*floor(a/b)", NOT fmod() (which takes the sign of
    // operand 1). They differ whenever the operands have opposite signs.
    auto a = value(inst.GetSingleWordOperand(2));
    auto b = value(inst.GetSingleWordOperand(3));
    val = c::binary(
        "-", a, c::binary("*", b, c::call("floor", {c::binary("/", a, b)})));
    break;
  }
  case spv::Op::OpSNegate:
  case spv::Op::OpFNegate:
    val = c::unary("-", value(inst.GetSingleWordOperand(2)));
    break;
  case spv::Op::OpLogicalNot:
    val = c::unary("!", value(inst.GetSingleWordOperand(2)));
    break;
  case spv::Op::OpNot:
    val = c::unary("~", value(inst.GetSingleWordOperand(2)));
    break;
  case spv::Op::OpLessOrGreater: {
    auto op1 = inst.GetSingleWordOperand(2);
    auto op2 = inst.GetSingleWordOperand(3);
    val = relational_mask(op1, call_values("islessgreater", {op1, op2}));
    break;
  }
  // Unordered float comparisons (OpFUnord*) are true when either operand is
  // NaN, and OpFOrdNotEqual is false for NaN -- but the C/OpenCL C relational
  // operators "< <= > >= ==" are *ordered* (false for NaN) while only "!=" is
  // unordered. Translating these as plain operators (as translate_binop does
  // for the remaining, correct cases below) drops the NaN semantics, so emit
  // NaN-correct forms built from the OpenCL ordered relational built-ins.
  case spv::Op::OpFOrdNotEqual:
  case spv::Op::OpFUnordEqual:
  case spv::Op::OpFUnordLessThan:
  case spv::Op::OpFUnordGreaterThan:
  case spv::Op::OpFUnordLessThanEqual:
  case spv::Op::OpFUnordGreaterThanEqual: {
    auto op1 = inst.GetSingleWordOperand(2);
    auto op2 = inst.GetSingleWordOperand(3);
    const char *ordered = nullptr;
    switch (opcode) {
    case spv::Op::OpFOrdNotEqual:
      // ordered "!=" is islessgreater (false when either operand is NaN)
      val = call_values("islessgreater", {op1, op2});
      break;
    case spv::Op::OpFUnordEqual:
      ordered = "isequal";
      break;
    case spv::Op::OpFUnordLessThan:
      ordered = "isless";
      break;
    case spv::Op::OpFUnordGreaterThan:
      ordered = "isgreater";
      break;
    case spv::Op::OpFUnordLessThanEqual:
      ordered = "islessequal";
      break;
    case spv::Op::OpFUnordGreaterThanEqual:
      ordered = "isgreaterequal";
      break;
    default:
      break;
    }
    if (ordered) {
      // unordered cmp == isunordered(a,b) | <ordered cmp>(a,b); the bitwise OR
      // is correct for both scalar (1/0) and vector (-1/0) relational results.
      val = c::binary("|", call_values("isunordered", {op1, op2}),
                      call_values(ordered, {op1, op2}));
    }
    val = relational_mask(op1, val);
    break;
  }
  case spv::Op::OpFOrdEqual:
  case spv::Op::OpFOrdLessThan:
  case spv::Op::OpFOrdGreaterThan:
  case spv::Op::OpFOrdLessThanEqual:
  case spv::Op::OpFOrdGreaterThanEqual:
  case spv::Op::OpFUnordNotEqual:
  case spv::Op::OpLogicalOr:
  case spv::Op::OpLogicalAnd:
  case spv::Op::OpULessThan:
  case spv::Op::OpULessThanEqual:
  case spv::Op::OpUGreaterThan:
  case spv::Op::OpUGreaterThanEqual:
  case spv::Op::OpLogicalEqual:
  case spv::Op::OpLogicalNotEqual:
  case spv::Op::OpIEqual:
  case spv::Op::OpINotEqual:
  case spv::Op::OpPtrEqual:
  case spv::Op::OpPtrNotEqual:
    val = relational_mask(inst.GetSingleWordOperand(2), translate_binop(inst));
    break;
  case spv::Op::OpSLessThanEqual:
  case spv::Op::OpSGreaterThan:
  case spv::Op::OpSGreaterThanEqual:
  case spv::Op::OpSLessThan:
    val = relational_mask(inst.GetSingleWordOperand(2),
                          translate_binop_signed(inst));
    break;
  case spv::Op::OpAny:
    val = call_values("any", {inst.GetSingleWordOperand(2)});
    break;
  case spv::Op::OpAll:
    val = call_values("all", {inst.GetSingleWordOperand(2)});
    break;
  case spv::Op::OpIsNan:
  case spv::Op::OpIsInf:
  case spv::Op::OpIsFinite:
  case spv::Op::OpIsNormal:
  case spv::Op::OpSignBitSet: {
    static const std::unordered_map<spv::Op, const char *> fns = {
        {spv::Op::OpIsNan, "isnan"},        {spv::Op::OpIsInf, "isinf"},
        {spv::Op::OpIsFinite, "isfinite"},  {spv::Op::OpIsNormal, "isnormal"},
        {spv::Op::OpSignBitSet, "signbit"},
    };
    auto operand = inst.GetSingleWordOperand(2);
    val = relational_mask(operand, call_values(fns.at(opcode), {operand}));
    break;
  }
  case spv::Op::OpBitCount:
    val = call_values("popcount", {inst.GetSingleWordOperand(2)});
    break;
  case spv::Op::OpOrdered:
  case spv::Op::OpUnordered: {
    auto x = inst.GetSingleWordOperand(2);
    auto y = inst.GetSingleWordOperand(3);
    auto fn = opcode == spv::Op::OpOrdered ? "isordered" : "isunordered";
    val = relational_mask(x, call_values(fn, {x, y}));
    break;
  }
  case spv::Op::OpConvertFToU:
  case spv::Op::OpConvertFToS: {
    auto op = inst.GetSingleWordOperand(2);
    bool sat = m_saturated_conversions.count(result);
    std::string fn = "convert_";
    if (opcode == spv::Op::OpConvertFToU) {
      fn += src_type(rtype);
    } else {
      fn += src_type_signed(rtype);
    }

    if (sat) {
      fn += "_sat";
    }

    if (m_rounding_mode_decorations.count(result)) {
      auto rmode = m_rounding_mode_decorations.at(result);
      fn += "_" + rounding_mode(rmode);
    } else {
      fn += "_" + rounding_mode(SpvFPRoundingModeRTZ);
    }

    val = c::call(fn, {value(op)});

    // SPIR-V requires that NaNs be converted to 0 for saturating conversions
    // but OpenCL C just recommends it (§6.2.3)
    if (sat) {
      val = c::ternary(call_values("isnan", {op}), c::literal("0"), val);
    }

    break;
  }
  case spv::Op::OpDot:
    val = call_values(
        "dot", {inst.GetSingleWordOperand(2), inst.GetSingleWordOperand(3)});
    break;
  case spv::Op::OpConvertUToF:
  case spv::Op::OpConvertSToF: {
    auto op = inst.GetSingleWordOperand(2);
    bool sat = m_saturated_conversions.count(result);
    std::string fn = "convert_" + src_type(rtype);

    if (sat) {
      fn += "_sat";
    }

    if (m_rounding_mode_decorations.count(result)) {
      auto rmode = m_rounding_mode_decorations.at(result);
      fn += "_" + rounding_mode(rmode);
    }

    // OpConvertSToF interprets the operand as a signed integer. Integer
    // variables are emitted as unsigned C types, so a bare convert_float()
    // would perform an unsigned conversion (e.g. -1 -> 1.8e19). Reinterpret
    // the operand as signed first. OpConvertUToF takes it as-is (unsigned).
    val = c::call(
        fn, {opcode == spv::Op::OpConvertSToF ? as_signed(op) : value(op)});
    break;
  }
  case spv::Op::OpSatConvertSToU:
    // Signed source -> unsigned dest, saturating (negatives clamp to 0).
    // Integer variables are emitted as unsigned C types, so reinterpret the
    // operand as signed first; the destination is the (unsigned) result type.
    val = call_signed("convert_" + src_type(rtype) + "_sat",
                      {inst.GetSingleWordOperand(2)});
    break;
  case spv::Op::OpSatConvertUToS:
    // Unsigned source -> signed dest, saturating (clamps to the signed range).
    // The operand is taken as-is (unsigned); the destination is the signed
    // result type (cf. OpConvertFToS, which likewise converts to the signed
    // type name without reinterpreting the operand).
    val = call_values("convert_" + src_type_signed(rtype) + "_sat",
                      {inst.GetSingleWordOperand(2)});
    break;
  case spv::Op::OpBitcast: {
    auto operand = inst.GetSingleWordOperand(2);
    auto dstty = type_for(rtype);
    auto srcty = type_for_val(operand);
    if ((srcty->kind() == Type::Kind::kPointer) ||
        (dstty->kind() == Type::Kind::kPointer)) {
      val = cast_to(rtype, value(operand));
    } else {
      val = as_type(rtype, value(operand));
    }
    break;
  }
  case spv::Op::OpSConvert:
    val = c::call("convert_" + src_type_signed(rtype),
                  {as_signed(inst.GetSingleWordOperand(2))});
    break;
  case spv::Op::OpFConvert:
  case spv::Op::OpUConvert:
    val = call_values("convert_" + src_type(rtype),
                      {inst.GetSingleWordOperand(2)});
    break;
  case spv::Op::OpSelect: {
    auto cond = inst.GetSingleWordOperand(2);
    auto val_true = inst.GetSingleWordOperand(3);
    auto val_false = inst.GetSingleWordOperand(4);
    val = c::ternary(select_condition(cond, rtype), value(val_true),
                     value(val_false));
    break;
  }
  case spv::Op::OpBranch: {
    auto target = inst.GetSingleWordOperand(0);
    assign_result = false;
    fb.statement("goto " + name_of(target));
    break;
  }
  case spv::Op::OpBranchConditional: {
    auto cond = inst.GetSingleWordOperand(0);
    auto label_true = inst.GetSingleWordOperand(1);
    auto label_false = inst.GetSingleWordOperand(2);
    assign_result = false;
    fb.statement("if (" + c::print(value(cond)) + ") { goto " +
                 name_of(label_true) + ";} else { goto " +
                 name_of(label_false) + ";}");
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
    std::string sw = "switch (" + c::print(value(select)) + "){";
    sw += "default: goto " + name_of(def) + ";";
    for (unsigned i = 2; i < inst.NumOperands(); i += 2) {
      auto &case_val = inst.GetOperand(i);
      auto &target = inst.GetOperand(i + 1);
      sw += "case " + std::to_string(case_val.AsLiteralUint64()) + ": goto " +
            name_of(target.AsId()) + ";";
    }
    fb.statement(sw + "}");
    break;
  }
  case spv::Op::OpControlBarrier: {
    auto exec_scope = constant_operand(inst.GetSingleWordOperand(0),
                                       "OpControlBarrier execution scope");
    auto mem_scope = constant_operand(inst.GetSingleWordOperand(1),
                                      "OpControlBarrier memory scope");
    auto mem_sem = constant_operand(inst.GetSingleWordOperand(2),
                                    "OpControlBarrier memory semantics");
    if (!exec_scope || !mem_scope || !mem_sem) {
      return false;
    }

    // The execution scope selects the barrier: work-group (barrier) vs
    // sub-group (sub_group_barrier, cl_khr_subgroups).
    bool work_group;
    if (*exec_scope == SpvScopeWorkgroup) {
      work_group = true;
    } else if (*exec_scope == SpvScopeSubgroup) {
      work_group = false;
    } else {
      std::cerr << "UNIMPLEMENTED OpControlBarrier execution scope "
                << *exec_scope << std::endl;
      return false;
    }

    // The fence flags come from the memory semantics (Workgroup memory -> local
    // fence, CrossWorkgroup memory -> global fence). The barrier fences memory
    // at its own scope unless given a wider one, which takes OpenCL C 2.0.
    if (scope_width(*mem_scope) <= scope_width(*exec_scope)) {
      fb.expression(c::call(work_group ? "barrier" : "sub_group_barrier",
                            {fence_flags(*mem_sem)}));
      break;
    }
    if (m_opencl_c_version < 200) {
      std::cerr << "UNIMPLEMENTED: barriers fencing beyond the work-group "
                   "require OpenCL C 2.0 (targeting "
                << opencl_c_version_str(m_opencl_c_version) << ").\n";
      return false;
    }
    auto scope_arg = memory_scope(*mem_scope);
    if (!scope_arg) {
      return false;
    }
    // work_group_barrier only fences image memory at the scope of the
    // work-group, so that takes a barrier of its own. Only accesses to
    // read-write images need it.
    auto mem_sem_scoped = *mem_sem;
    if (work_group) {
      mem_sem_scoped &= ~SpvMemorySemanticsImageMemoryMask;
      if ((*mem_sem & SpvMemorySemanticsImageMemoryMask) &&
          m_read_write_images) {
        fb.expression(c::call(
            "barrier", {fence_flags(SpvMemorySemanticsImageMemoryMask)}));
      }
    }
    fb.expression(
        c::call(work_group ? "work_group_barrier" : "sub_group_barrier",
                {fence_flags(mem_sem_scoped), scope_arg}));
    break;
  }
  case spv::Op::OpMemoryBarrier: {
    auto scope = constant_operand(inst.GetSingleWordOperand(0),
                                  "OpMemoryBarrier memory scope");
    auto mem_sem = constant_operand(inst.GetSingleWordOperand(1),
                                    "OpMemoryBarrier memory semantics");
    if (!scope || !mem_sem) {
      return false;
    }
    assign_result = false;
    auto order = order_of(*mem_sem);
    // A relaxed fence orders nothing.
    if (order == memory_order::relaxed) {
      break;
    }
    // mem_fence orders memory within the work-group.
    if (m_opencl_c_version < 200) {
      if (scope_width(*scope) > scope_width(SpvScopeWorkgroup)) {
        std::cerr << "UNIMPLEMENTED: fences beyond the work-group require "
                     "OpenCL C 2.0 (targeting "
                  << opencl_c_version_str(m_opencl_c_version) << ").\n";
        return false;
      }
      fb.expression(c::call("mem_fence", {fence_flags(*mem_sem)}));
      break;
    }
    auto scope_arg = memory_scope(*scope);
    if (!scope_arg) {
      return false;
    }
    // OpenCL C only fences image memory within a work-item (across work-items
    // that takes a barrier), with its own fence. That orders accesses to
    // read-write images, and devices without them reject the fence.
    bool image_fence = (*mem_sem & SpvMemorySemanticsImageMemoryMask) &&
                       *scope == SpvScopeInvocation && m_read_write_images;
    if (image_fence) {
      fb.expression(c::call("atomic_work_item_fence",
                            {c::name("CLK_IMAGE_MEM_FENCE"),
                             c::name("memory_order_acq_rel"),
                             c::name("memory_scope_work_item")}));
    }
    // Flags naming no memory are undefined behavior in atomic_work_item_fence,
    // so fence all memory instead.
    auto memory = *mem_sem & (SpvMemorySemanticsWorkgroupMemoryMask |
                              SpvMemorySemanticsCrossWorkgroupMemoryMask);
    if (memory == 0 && !image_fence) {
      memory = SpvMemorySemanticsWorkgroupMemoryMask |
               SpvMemorySemanticsCrossWorkgroupMemoryMask;
    }
    if (memory != 0) {
      fb.expression(c::call("atomic_work_item_fence",
                            {fence_flags(memory), spell(order), scope_arg}));
    }
    break;
  }
  case spv::Op::OpGroupNonUniformShuffle:
  case spv::Op::OpGroupNonUniformShuffleXor: {
    // Sub-group shuffle: operands are <scope> <value> <id|mask>. The scope is
    // Subgroup; map to the cl_khr_subgroups shuffle builtins.
    const char *fn = opcode == spv::Op::OpGroupNonUniformShuffle
                         ? "sub_group_shuffle"
                         : "sub_group_shuffle_xor";
    val = call_values(
        fn, {inst.GetSingleWordOperand(3), inst.GetSingleWordOperand(4)});
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
      val = call_values("async_work_group_copy",
                        {dst_ptr, src_ptr, num_elems, event});
    } else {
      val = call_values("async_work_group_strided_copy",
                        {dst_ptr, src_ptr, num_elems, stride, event});
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

    fb.expression(call_values("wait_group_events", {num_events, event_list}));
    assign_result = false;
    break;
  }
  case spv::Op::OpExtInst: {
    assign_result = false;
    if (!translate_extended_instruction(inst, fb)) {
      return false;
    }
    break;
  }
  default:
    std::cerr << "UNIMPLEMENTED instruction " << opcode << std::endl;
    return false;
  }

  if ((result != 0) && assign_result) {
    fb.declare(src_var_decl(result), val);
  }

  return true;
}

c::expr_ref translator_impl::materialize(uint32_t id, function_builder &fb) {
  auto val = value(id);
  if (val->k == c::expr::kind::name) {
    return val;
  }
  auto tmp = m_name_allocator.allocate(name_of(id) + "_tmp");
  fb.declare(src_var_decl(type_id_for(id), tmp), val);
  return c::name(tmp);
}

c::expr_ref translator_impl::translate_binop(const Instruction &inst) const {
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
      {spv::Op::OpPtrEqual, "=="},
      {spv::Op::OpPtrNotEqual, "!="},
      {spv::Op::OpBitwiseOr, "|"},
      {spv::Op::OpBitwiseXor, "^"},
      {spv::Op::OpBitwiseAnd, "&"},
      {spv::Op::OpLogicalOr, "||"},
      {spv::Op::OpLogicalAnd, "&&"},
      {spv::Op::OpVectorTimesScalar, "*"},
      {spv::Op::OpShiftLeftLogical, "<<"},
      {spv::Op::OpShiftRightLogical, ">>"},
      // Ordered float comparisons map directly to the (ordered) C operators.
      // OpFUnordNotEqual maps to "!=" because C "!=" is itself unordered (true
      // for NaN). The other unordered comparisons and OpFOrdNotEqual need
      // NaN-aware handling and are emitted separately (see
      // translate_instruction).
      {spv::Op::OpFOrdEqual, "=="},
      {spv::Op::OpFUnordNotEqual, "!="},
      {spv::Op::OpFOrdLessThan, "<"},
      {spv::Op::OpFOrdGreaterThan, ">"},
      {spv::Op::OpFOrdLessThanEqual, "<="},
      {spv::Op::OpFOrdGreaterThanEqual, ">="},
  };

  return c::binary(binops.at(inst.opcode()),
                   value(inst.GetSingleWordOperand(2)),
                   value(inst.GetSingleWordOperand(3)));
}

c::expr_ref
translator_impl::translate_binop_signed(const Instruction &inst) const {
  static std::unordered_map<spv::Op, const std::string> binops = {
      {spv::Op::OpSDiv, "/"},
      {spv::Op::OpSRem, "%"},
      {spv::Op::OpShiftRightArithmetic, ">>"},
      {spv::Op::OpSLessThan, "<"},
      {spv::Op::OpSLessThanEqual, "<="},
      {spv::Op::OpSGreaterThan, ">"},
      {spv::Op::OpSGreaterThanEqual, ">="},
  };

  return c::binary(binops.at(inst.opcode()),
                   as_signed(inst.GetSingleWordOperand(2)),
                   as_signed(inst.GetSingleWordOperand(3)));
}

c::expr_ref translator_impl::builtin_scalar(SpvBuiltIn builtin) const {
  switch (builtin) {
  case SpvBuiltInWorkDim:
    return c::call("get_work_dim", {});
  case SpvBuiltInSubgroupSize:
    return c::call("get_sub_group_size", {});
  case SpvBuiltInSubgroupMaxSize:
    return c::call("get_max_sub_group_size", {});
  case SpvBuiltInNumSubgroups:
    return c::call("get_num_sub_groups", {});
  case SpvBuiltInSubgroupId:
    return c::call("get_sub_group_id", {});
  case SpvBuiltInSubgroupLocalInvocationId:
    return c::call("get_sub_group_local_id", {});
  default:
    return c::literal(
        note_unsupported("builtin value " + std::to_string(builtin)));
  }
}

c::expr_ref translator_impl::builtin_vector_extract(uint32_t id,
                                                    c::expr_ref idx) const {
  const char *query;
  switch (m_builtin_values.at(id)) {
  case SpvBuiltInGlobalInvocationId:
    query = "get_global_id";
    break;
  case SpvBuiltInGlobalOffset:
    query = "get_global_offset";
    break;
  case SpvBuiltInGlobalSize:
    query = "get_global_size";
    break;
  case SpvBuiltInWorkgroupId:
    query = "get_group_id";
    break;
  case SpvBuiltInWorkgroupSize:
    query = "get_local_size";
    break;
  case SpvBuiltInLocalInvocationId:
    query = "get_local_id";
    break;
  case SpvBuiltInNumWorkgroups:
    query = "get_num_groups";
    break;
  default:
    return c::literal(note_unsupported("built-in in builtin_vector_extract"));
  }
  return c::call(query, {std::move(idx)});
}

c::expr_ref translator_impl::builtin_vector(uint32_t id) const {
  auto type = type_for_val(id);
  auto vec = type ? type->AsVector() : nullptr;
  if (!vec) {
    return c::literal(note_unsupported("non-vector built-in value"));
  }
  // (typeN)(query(0), query(1), ..., query(N-1))
  std::vector<c::expr_ref> comps;
  for (uint32_t i = 0; i < vec->element_count(); i++) {
    comps.push_back(builtin_vector_extract(id, c::literal(std::to_string(i))));
  }
  return c::vector_literal(src_type(type_id_for(id)), std::move(comps));
}
