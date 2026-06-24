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

std::string
translator::translate_extended_ternary(const Instruction &inst) const {
  auto rtype = inst.type_id();
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  auto a = inst.GetSingleWordOperand(4);
  auto b = inst.GetSingleWordOperand(5);
  auto c = inst.GetSingleWordOperand(6);
  auto fn_signed = gExtendedInstructionsTernary.at(extinst);
  if (fn_signed.second) {
    return src_as(rtype, src_function_call_signed(fn_signed.first, a, b, c));
  } else {
    return src_function_call(fn_signed.first, a, b, c);
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

std::string
translator::translate_extended_binary(const Instruction &inst) const {
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
    return gExtendedInstructionsBinary.at(extinst).first + "(" + var_for(x) +
           ", " + src_as_signed(y) + ")";
  }
  auto fn_signed = gExtendedInstructionsBinary.at(extinst);
  if (fn_signed.second) {
    return src_as(rtype, src_function_call_signed(fn_signed.first, x, y));
  } else {
    return src_function_call(fn_signed.first, x, y);
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

std::string
translator::translate_extended_unary(const Instruction &inst) const {
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  auto val = inst.GetSingleWordOperand(4);
  return src_function_call(gExtendedInstructionsUnary.at(extinst), val);
}

bool translator::translate_extended_instruction(const Instruction &inst,
                                                std::string &src) {
  auto result = inst.result_id();
  auto instruction =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));

  std::string sval;
  bool assign_result = true;

  if (gExtendedInstructionsUnary.count(instruction)) {
    sval = translate_extended_unary(inst);
  } else if (gExtendedInstructionsBinary.count(instruction)) {
    sval = translate_extended_binary(inst);
  } else if (gExtendedInstructionsTernary.count(instruction)) {
    sval = translate_extended_ternary(inst);
  } else {
    switch (instruction) {
    case OpenCLLIB::Ilogb: {
      // ilogb returns a signed intn; reinterpret to the (unsigned) result type
      // so a vector assignment type-checks (uintN = intN is not implicit).
      auto x = inst.GetSingleWordOperand(4);
      sval = src_as(inst.type_id(), src_function_call("ilogb", x));
      break;
    }
    case OpenCLLIB::Vloadn: {
      auto offset = inst.GetSingleWordOperand(4);
      auto ptr = inst.GetSingleWordOperand(5);
      auto n = inst.GetSingleWordOperand(6);
      sval = src_function_call("vload" + std::to_string(n), offset, ptr);
      break;
    }
    case OpenCLLIB::Vload_halfn: {
      auto offset = inst.GetSingleWordOperand(4);
      auto ptr = inst.GetSingleWordOperand(5);
      auto n = inst.GetSingleWordOperand(6);
      sval = src_function_call("vload_half" + std::to_string(n), offset, ptr);
      break;
    }
    case OpenCLLIB::Vloada_halfn: {
      auto offset = inst.GetSingleWordOperand(4);
      auto ptr = inst.GetSingleWordOperand(5);
      auto n = inst.GetSingleWordOperand(6);
      sval = src_function_call("vloada_half" + std::to_string(n), offset, ptr);
      break;
    }
    case OpenCLLIB::Vstoren: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src = src_function_call("vstore" + std::to_string(n), data, offset, ptr);
      break;
    }
    case OpenCLLIB::Vstore_half: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      src = src_function_call("vstore_half", data, offset, ptr);
      break;
    }
    case OpenCLLIB::Vstore_half_r: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      auto mode = inst.GetSingleWordOperand(7);
      std::string mode_str =
          rounding_mode(static_cast<SpvFPRoundingMode>(mode));
      assign_result = false;
      src = src_function_call("vstore_half_" + mode_str, data, offset, ptr);
      break;
    }
    case OpenCLLIB::Vstore_halfn: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src = src_function_call("vstore_half" + std::to_string(n), data, offset,
                              ptr);
      break;
    }
    case OpenCLLIB::Vstorea_halfn: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src = src_function_call("vstorea_half" + std::to_string(n), data, offset,
                              ptr);
      break;
    }
    case OpenCLLIB::Vstorea_halfn_r: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      auto mode = inst.GetSingleWordOperand(7);
      std::string mode_str =
          rounding_mode(static_cast<SpvFPRoundingMode>(mode));
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src =
          src_function_call("vstorea_half" + std::to_string(n) + "_" + mode_str,
                            data, offset, ptr);
      break;
    }
    case OpenCLLIB::SAbs: {
      auto val = inst.GetSingleWordOperand(4);
      sval = src_function_call_signed("abs", val);
      break;
    }
    case OpenCLLIB::SAbs_diff: {
      auto a = inst.GetSingleWordOperand(4);
      auto b = inst.GetSingleWordOperand(5);
      sval = src_function_call_signed("abs_diff", a, b);
      break;
    }
    case OpenCLLIB::Frexp: {
      auto x = inst.GetSingleWordOperand(4);
      auto exp = inst.GetSingleWordOperand(5);
      sval = src_function_call(
          "frexp", var_for(x) + ", " + src_cast_signed(type_id_for(exp), exp));
      break;
    }
    case OpenCLLIB::Lgamma_r: {
      auto x = inst.GetSingleWordOperand(4);
      auto signp = inst.GetSingleWordOperand(5);
      sval = src_function_call("lgamma_r",
                               var_for(x) + ", " +
                                   src_cast_signed(type_id_for(signp), signp));
      break;
    }
    case OpenCLLIB::Remquo: {
      auto x = inst.GetSingleWordOperand(4);
      auto y = inst.GetSingleWordOperand(5);
      auto quo = inst.GetSingleWordOperand(6);
      sval = src_function_call("remquo",
                               var_for(x) + ", " + var_for(y) + ", " +
                                   src_cast_signed(type_id_for(quo), quo));
      break;
    }
    case OpenCLLIB::Printf: {
      auto format = inst.GetSingleWordOperand(4);
      std::string format_arg;

      // Check if we have cached string data for this variable
      auto string_literal = string_literal_for(format);
      if (string_literal) {
        format_arg = *string_literal;
      } else {
        format_arg = var_for(format);
        auto format_type = type_for_val(format);
        auto ptr_type = format_type->AsPointer();
        auto pointee_type = ptr_type->pointee_type();

        // When dealing with an array, make sure to pass a pointer. Arrays are
        // struct-wrapped, so decay through the 'e' member.
        if (pointee_type->kind() == Type::Kind::kArray) {
          format_arg = "&(" + format_arg + "->e[0])";
        }
      }

      std::string src_args = format_arg;
      for (unsigned op = 5; op < inst.NumOperands(); op++) {
        auto arg = inst.GetSingleWordOperand(op);
        src_args += ", " + var_for(arg);
      }
      sval = src_function_call("printf", src_args);
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED extended instruction " << instruction
                << std::endl;
      return false;
    }
  }

  if ((result != 0) && assign_result) {
    src = src_var_decl(result) + " = " + sval;
  }

  return true;
}
