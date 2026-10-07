static std::unordered_map<OpenCLLIB::Entrypoints,
                          std::pair<const std::string, bool>>
    gExtendedInstructionsTernary = {
        {OpenCLLIB::Bitselect, {"bitselect", false}},
        {OpenCLLIB::FClamp, {"clamp", false}},
        {OpenCLLIB::SClamp, {"clamp", true}},
        {OpenCLLIB::UClamp, {"clamp", false}},
        {OpenCLLIB::Fma, {"fma", false}},
        {OpenCLLIB::Mad, {"mad", false}},
        {OpenCLLIB::Mix, {"mix", false}},
        {OpenCLLIB::SMad24, {"mad24", true}},
        {OpenCLLIB::UMad24, {"mad24", false}},
        {OpenCLLIB::SMad_hi, {"mad_hi", true}},
        {OpenCLLIB::UMad_hi, {"mad_hi", false}},
        {OpenCLLIB::SMad_sat, {"mad_sat", true}},
        {OpenCLLIB::UMad_sat, {"mad_sat", false}},
        {OpenCLLIB::Select, {"select", false}},
        {OpenCLLIB::Shuffle2, {"shuffle2", false}},
        {OpenCLLIB::Smoothstep, {"smoothstep", false}},
};

c::expr_ref
translator_impl::translate_extended_ternary(const Instruction &inst) const {
  auto rtype = inst.type_id();
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  std::vector<uint32_t> args = {inst.GetSingleWordOperand(4),
                                inst.GetSingleWordOperand(5),
                                inst.GetSingleWordOperand(6)};
  auto fn_signed = gExtendedInstructionsTernary.at(extinst);
  if (fn_signed.second) {
    return as_type(rtype, call_signed(fn_signed.first, args));
  } else {
    return call_values(fn_signed.first, args);
  }
}

static std::unordered_map<OpenCLLIB::Entrypoints,
                          std::pair<const std::string, bool>>
    gExtendedInstructionsBinary = {
        {OpenCLLIB::UAbs_diff, {"abs_diff", false}},
        {OpenCLLIB::SHadd, {"hadd", true}},
        {OpenCLLIB::UHadd, {"hadd", false}},
        {OpenCLLIB::SMul_hi, {"mul_hi", true}},
        {OpenCLLIB::UMul_hi, {"mul_hi", false}},
        {OpenCLLIB::SRhadd, {"rhadd", true}},
        {OpenCLLIB::URhadd, {"rhadd", false}},
        {OpenCLLIB::Rotate, {"rotate", false}},
        {OpenCLLIB::SAdd_sat, {"add_sat", true}},
        {OpenCLLIB::UAdd_sat, {"add_sat", false}},
        {OpenCLLIB::SSub_sat, {"sub_sat", true}},
        {OpenCLLIB::USub_sat, {"sub_sat", false}},
        {OpenCLLIB::SMul24, {"mul24", true}},
        {OpenCLLIB::UMul24, {"mul24", false}},
        {OpenCLLIB::Shuffle, {"shuffle", false}},
        {OpenCLLIB::Atan2, {"atan2", false}},
        {OpenCLLIB::Atan2pi, {"atan2pi", false}},
        {OpenCLLIB::Copysign, {"copysign", false}},
        {OpenCLLIB::Fdim, {"fdim", false}},
        {OpenCLLIB::Fmax, {"fmax", false}},
        {OpenCLLIB::Fmin, {"fmin", false}},
        {OpenCLLIB::Fmod, {"fmod", false}},
        {OpenCLLIB::Hypot, {"hypot", false}},
        {OpenCLLIB::Ldexp, {"ldexp", false}},
        {OpenCLLIB::Maxmag, {"maxmag", false}},
        {OpenCLLIB::Minmag, {"minmag", false}},
        {OpenCLLIB::Modf, {"modf", false}},
        {OpenCLLIB::Nextafter, {"nextafter", false}},
        {OpenCLLIB::Pow, {"pow", false}},
        {OpenCLLIB::Pown, {"pown", false}},
        {OpenCLLIB::Powr, {"powr", false}},
        {OpenCLLIB::Remainder, {"remainder", false}},
        {OpenCLLIB::Rootn, {"rootn", false}},
        {OpenCLLIB::Sincos, {"sincos", false}},
        {OpenCLLIB::Fract, {"fract", false}},
        {OpenCLLIB::Half_divide, {"half_divide", false}},
        {OpenCLLIB::Half_powr, {"half_powr", false}},
        {OpenCLLIB::Cross, {"cross", false}},
        {OpenCLLIB::Distance, {"distance", false}},
        {OpenCLLIB::Fast_distance, {"fast_distance", false}},
        {OpenCLLIB::Step, {"step", false}},
        {OpenCLLIB::S_Upsample, {"upsample", true}},
        {OpenCLLIB::U_Upsample, {"upsample", false}},
        {OpenCLLIB::SMax, {"max", true}},
        {OpenCLLIB::UMax, {"max", false}},
        {OpenCLLIB::SMin, {"min", true}},
        {OpenCLLIB::UMin, {"min", false}},
        {OpenCLLIB::Vload_half, {"vload_half", false}},
};

c::expr_ref
translator_impl::translate_extended_binary(const Instruction &inst) const {
  auto rtype = inst.type_id();
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  auto x = inst.GetSingleWordOperand(4);
  auto y = inst.GetSingleWordOperand(5);
  // pown/rootn/ldexp take a *signed* integer as their second argument, while
  // the first argument and the result stay floating-point. The all-or-nothing
  // signed handling below would reinterpret every operand and the result, so
  // cast just the exponent to the matching signed integer type.
  if (extinst == OpenCLLIB::Pown || extinst == OpenCLLIB::Rootn ||
      extinst == OpenCLLIB::Ldexp) {
    return c::call(gExtendedInstructionsBinary.at(extinst).first,
                   {value(x), as_signed(y)});
  }
  auto fn_signed = gExtendedInstructionsBinary.at(extinst);
  if (fn_signed.second) {
    return as_type(rtype, call_signed(fn_signed.first, {x, y}));
  } else {
    return call_values(fn_signed.first, {x, y});
  }
}

static std::unordered_map<OpenCLLIB::Entrypoints, const std::string>
    gExtendedInstructionsUnary = {
        {OpenCLLIB::UAbs, "abs"},
        {OpenCLLIB::Acos, "acos"},
        {OpenCLLIB::Acosh, "acosh"},
        {OpenCLLIB::Acospi, "acospi"},
        {OpenCLLIB::Asin, "asin"},
        {OpenCLLIB::Asinh, "asinh"},
        {OpenCLLIB::Asinpi, "asinpi"},
        {OpenCLLIB::Atan, "atan"},
        {OpenCLLIB::Atanh, "atanh"},
        {OpenCLLIB::Atanpi, "atanpi"},
        {OpenCLLIB::Cbrt, "cbrt"},
        {OpenCLLIB::Ceil, "ceil"},
        {OpenCLLIB::Clz, "clz"},
        {OpenCLLIB::Cos, "cos"},
        {OpenCLLIB::Cosh, "cosh"},
        {OpenCLLIB::Cospi, "cospi"},
        {OpenCLLIB::Degrees, "degrees"},
        {OpenCLLIB::Erf, "erf"},
        {OpenCLLIB::Erfc, "erfc"},
        {OpenCLLIB::Exp, "exp"},
        {OpenCLLIB::Exp2, "exp2"},
        {OpenCLLIB::Exp10, "exp10"},
        {OpenCLLIB::Expm1, "expm1"},
        {OpenCLLIB::Fabs, "fabs"},
        {OpenCLLIB::Fast_length, "fast_length"},
        {OpenCLLIB::Fast_normalize, "fast_normalize"},
        {OpenCLLIB::Floor, "floor"},
        {OpenCLLIB::Half_cos, "half_cos"},
        {OpenCLLIB::Half_exp, "half_exp"},
        {OpenCLLIB::Half_exp2, "half_exp2"},
        {OpenCLLIB::Half_exp10, "half_exp10"},
        {OpenCLLIB::Half_log, "half_log"},
        {OpenCLLIB::Half_log2, "half_log2"},
        {OpenCLLIB::Half_log10, "half_log10"},
        {OpenCLLIB::Half_recip, "half_recip"},
        {OpenCLLIB::Half_rsqrt, "half_rsqrt"},
        {OpenCLLIB::Half_sin, "half_sin"},
        {OpenCLLIB::Half_sqrt, "half_sqrt"},
        {OpenCLLIB::Half_tan, "half_tan"},
        {OpenCLLIB::Length, "length"},
        {OpenCLLIB::Lgamma, "lgamma"},
        {OpenCLLIB::Log, "log"},
        {OpenCLLIB::Log2, "log2"},
        {OpenCLLIB::Log10, "log10"},
        {OpenCLLIB::Log1p, "log1p"},
        {OpenCLLIB::Logb, "logb"},
        {OpenCLLIB::Nan, "nan"},
        {OpenCLLIB::Normalize, "normalize"},
        {OpenCLLIB::Radians, "radians"},
        {OpenCLLIB::Rint, "rint"},
        {OpenCLLIB::Round, "round"},
        {OpenCLLIB::Rsqrt, "rsqrt"},
        {OpenCLLIB::Sign, "sign"},
        {OpenCLLIB::Sin, "sin"},
        {OpenCLLIB::Sinh, "sinh"},
        {OpenCLLIB::Sinpi, "sinpi"},
        {OpenCLLIB::Sqrt, "sqrt"},
        {OpenCLLIB::Tan, "tan"},
        {OpenCLLIB::Tanh, "tanh"},
        {OpenCLLIB::Tanpi, "tanpi"},
        {OpenCLLIB::Tgamma, "tgamma"},
        {OpenCLLIB::Trunc, "trunc"},
};

c::expr_ref
translator_impl::translate_extended_unary(const Instruction &inst) const {
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  return call_values(gExtendedInstructionsUnary.at(extinst),
                     {inst.GetSingleWordOperand(4)});
}

bool translator_impl::translate_extended_instruction(const Instruction &inst,
                                                     function_builder &fb) {
  auto result = inst.result_id();
  auto instruction =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  auto operand = [&inst](unsigned i) { return inst.GetSingleWordOperand(i); };

  c::expr_ref val;
  // The vstore family returns nothing; it is a statement of its own.
  c::expr_ref stmt;

  if (gExtendedInstructionsUnary.count(instruction)) {
    val = translate_extended_unary(inst);
  } else if (gExtendedInstructionsBinary.count(instruction)) {
    val = translate_extended_binary(inst);
  } else if (gExtendedInstructionsTernary.count(instruction)) {
    val = translate_extended_ternary(inst);
  } else {
    switch (instruction) {
    case OpenCLLIB::Ctz:
      if (m_opencl_c_version >= 200) {
        val = call_values("ctz", {operand(4)});
      } else {
        // ctz is OpenCL C 2.0. `(x & -x) - 1` sets exactly the trailing zero
        // bits, and all bits for x == 0, where ctz is the bit width. The cast
        // undoes the promotion of char and short operands to int.
        auto x = value(operand(4));
        auto ones = c::binary("-", c::binary("&", x, c::unary("-", x)),
                              c::literal("1"));
        val = c::call("popcount", {cast_to(inst.type_id(), ones)});
      }
      break;
    case OpenCLLIB::Ilogb:
      // ilogb returns a signed intn; reinterpret to the (unsigned) result type
      // so a vector assignment type-checks (uintN = intN is not implicit).
      val = as_type(inst.type_id(), call_values("ilogb", {operand(4)}));
      break;
    case OpenCLLIB::Vloadn:
      val = call_values("vload" + std::to_string(operand(6)),
                        {operand(4), operand(5)});
      break;
    case OpenCLLIB::Vload_halfn:
      val = call_values("vload_half" + std::to_string(operand(6)),
                        {operand(4), operand(5)});
      break;
    case OpenCLLIB::Vloada_halfn:
      val = call_values("vloada_half" + std::to_string(operand(6)),
                        {operand(4), operand(5)});
      break;
    case OpenCLLIB::Vstoren: {
      auto n = type_for_val(operand(4))->AsVector()->element_count();
      stmt = call_values("vstore" + std::to_string(n),
                         {operand(4), operand(5), operand(6)});
      break;
    }
    case OpenCLLIB::Vstore_half:
      stmt = call_values("vstore_half", {operand(4), operand(5), operand(6)});
      break;
    case OpenCLLIB::Vstore_half_r: {
      auto mode = rounding_mode(static_cast<SpvFPRoundingMode>(operand(7)));
      stmt = call_values("vstore_half_" + mode,
                         {operand(4), operand(5), operand(6)});
      break;
    }
    case OpenCLLIB::Vstore_halfn: {
      auto n = type_for_val(operand(4))->AsVector()->element_count();
      stmt = call_values("vstore_half" + std::to_string(n),
                         {operand(4), operand(5), operand(6)});
      break;
    }
    case OpenCLLIB::Vstorea_halfn: {
      auto n = type_for_val(operand(4))->AsVector()->element_count();
      stmt = call_values("vstorea_half" + std::to_string(n),
                         {operand(4), operand(5), operand(6)});
      break;
    }
    case OpenCLLIB::Vstorea_halfn_r: {
      auto mode = rounding_mode(static_cast<SpvFPRoundingMode>(operand(7)));
      auto n = type_for_val(operand(4))->AsVector()->element_count();
      stmt = call_values("vstorea_half" + std::to_string(n) + "_" + mode,
                         {operand(4), operand(5), operand(6)});
      break;
    }
    case OpenCLLIB::SAbs:
      val = call_signed("abs", {operand(4)});
      break;
    case OpenCLLIB::SAbs_diff:
      val = call_signed("abs_diff", {operand(4), operand(5)});
      break;
    case OpenCLLIB::Frexp: {
      auto exp = operand(5);
      val = c::call("frexp", {value(operand(4)),
                              cast_to_signed(type_id_for(exp), value(exp))});
      break;
    }
    case OpenCLLIB::Lgamma_r: {
      auto signp = operand(5);
      val = c::call("lgamma_r",
                    {value(operand(4)),
                     cast_to_signed(type_id_for(signp), value(signp))});
      break;
    }
    case OpenCLLIB::Remquo: {
      auto quo = operand(6);
      val = c::call("remquo", {value(operand(4)), value(operand(5)),
                               cast_to_signed(type_id_for(quo), value(quo))});
      break;
    }
    case OpenCLLIB::Printf: {
      auto format = operand(4);
      c::expr_ref format_arg;

      // Check if we have cached string data for this variable
      auto string_literal = string_literal_for(format);
      if (string_literal) {
        format_arg = c::literal(*string_literal);
      } else {
        format_arg = value(format);
        auto pointee_type = type_for_val(format)->AsPointer()->pointee_type();

        // When dealing with an array, make sure to pass a pointer. Arrays are
        // struct-wrapped, so decay through the 'e' member.
        if (pointee_type->kind() == Type::Kind::kArray) {
          format_arg = c::address_of(
              c::index(c::member(c::deref(format_arg), "e"), c::literal("0")));
        }
      }

      std::vector<c::expr_ref> args = {format_arg};
      for (unsigned op = 5; op < inst.NumOperands(); op++) {
        args.push_back(value(operand(op)));
      }
      val = c::call("printf", std::move(args));
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED extended instruction " << instruction
                << std::endl;
      return false;
    }
  }

  if (stmt) {
    fb.expression(stmt);
  } else if (result != 0) {
    fb.declare(src_var_decl(result), val);
  }

  return true;
}
