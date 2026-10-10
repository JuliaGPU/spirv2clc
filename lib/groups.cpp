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

// OpenCL builtin mappings follow Khronos SPIRV-LLVM-Translator's
// SPIRVToOCL.cpp (groupOCToOCLBuiltinName / visitCallSPIRVGroupBuiltin).
std::optional<bool>
translator_impl::translate_group_instruction(const Instruction &inst,
                                             c::expr_ref &result) {
  struct builtin {
    const char *name;
    bool arithmetic;
    bool signed_operand;
    bool predicate;
    bool boolean_result;
  };
  static const std::unordered_map<spv::Op, builtin> builtins = {
      {spv::Op::OpGroupAll, {"all", false, false, true, true}},
      {spv::Op::OpGroupAny, {"any", false, false, true, true}},
      {spv::Op::OpGroupBroadcast, {"broadcast", false, false, false, false}},
      {spv::Op::OpGroupNonUniformRotateKHR,
       {"rotate", false, false, false, false}},
      {spv::Op::OpGroupNonUniformElect, {"elect", false, false, false, true}},
      {spv::Op::OpGroupNonUniformAll,
       {"non_uniform_all", false, false, true, true}},
      {spv::Op::OpGroupNonUniformAny,
       {"non_uniform_any", false, false, true, true}},
      {spv::Op::OpGroupNonUniformAllEqual,
       {"non_uniform_all_equal", false, false, false, true}},
      {spv::Op::OpGroupNonUniformBroadcast,
       {"non_uniform_broadcast", false, false, false, false}},
      {spv::Op::OpGroupNonUniformBroadcastFirst,
       {"broadcast_first", false, false, false, false}},
      {spv::Op::OpGroupNonUniformBallot, {"ballot", false, false, true, false}},
      {spv::Op::OpGroupNonUniformInverseBallot,
       {"inverse_ballot", false, false, false, true}},
      {spv::Op::OpGroupNonUniformBallotBitExtract,
       {"ballot_bit_extract", false, false, false, true}},
      {spv::Op::OpGroupNonUniformBallotFindLSB,
       {"ballot_find_lsb", false, false, false, false}},
      {spv::Op::OpGroupNonUniformBallotFindMSB,
       {"ballot_find_msb", false, false, false, false}},
      {spv::Op::OpGroupNonUniformShuffle,
       {"shuffle", false, false, false, false}},
      {spv::Op::OpGroupNonUniformShuffleXor,
       {"shuffle_xor", false, false, false, false}},
      {spv::Op::OpGroupNonUniformShuffleUp,
       {"shuffle_up", false, false, false, false}},
      {spv::Op::OpGroupNonUniformShuffleDown,
       {"shuffle_down", false, false, false, false}},
      {spv::Op::OpGroupIAdd, {"add", true, false, false, false}},
      {spv::Op::OpGroupFAdd, {"add", true, false, false, false}},
      {spv::Op::OpGroupFMin, {"min", true, false, false, false}},
      {spv::Op::OpGroupUMin, {"min", true, false, false, false}},
      {spv::Op::OpGroupSMin, {"min", true, true, false, false}},
      {spv::Op::OpGroupFMax, {"max", true, false, false, false}},
      {spv::Op::OpGroupUMax, {"max", true, false, false, false}},
      {spv::Op::OpGroupSMax, {"max", true, true, false, false}},
      {spv::Op::OpGroupNonUniformIAdd, {"add", true, false, false, false}},
      {spv::Op::OpGroupNonUniformFAdd, {"add", true, false, false, false}},
      {spv::Op::OpGroupNonUniformIMul, {"mul", true, false, false, false}},
      {spv::Op::OpGroupNonUniformFMul, {"mul", true, false, false, false}},
      {spv::Op::OpGroupNonUniformSMin, {"min", true, true, false, false}},
      {spv::Op::OpGroupNonUniformUMin, {"min", true, false, false, false}},
      {spv::Op::OpGroupNonUniformFMin, {"min", true, false, false, false}},
      {spv::Op::OpGroupNonUniformSMax, {"max", true, true, false, false}},
      {spv::Op::OpGroupNonUniformUMax, {"max", true, false, false, false}},
      {spv::Op::OpGroupNonUniformFMax, {"max", true, false, false, false}},
      {spv::Op::OpGroupNonUniformBitwiseAnd,
       {"and", true, false, false, false}},
      {spv::Op::OpGroupNonUniformBitwiseOr, {"or", true, false, false, false}},
      {spv::Op::OpGroupNonUniformBitwiseXor,
       {"xor", true, false, false, false}},
      {spv::Op::OpGroupNonUniformLogicalAnd,
       {"logical_and", true, false, true, true}},
      {spv::Op::OpGroupNonUniformLogicalOr,
       {"logical_or", true, false, true, true}},
      {spv::Op::OpGroupNonUniformLogicalXor,
       {"logical_xor", true, false, true, true}},
      {spv::Op::OpGroupNonUniformBallotBitCount,
       {"ballot", true, false, false, false}},
  };
  auto it = builtins.find(inst.opcode());
  if (it == builtins.end()) {
    return std::nullopt;
  }
  const auto &fn = it->second;
  auto scope =
      constant_operand(inst.GetSingleWordOperand(2), "group execution scope");
  if (!scope) {
    return false;
  }
  const bool uniform = inst.opcode() >= spv::Op::OpGroupAll &&
                       inst.opcode() <= spv::Op::OpGroupSMax;
  if (*scope != SpvScopeSubgroup && !(uniform && *scope == SpvScopeWorkgroup)) {
    std::cerr << "UNIMPLEMENTED group execution scope " << *scope << std::endl;
    return false;
  }
  if (m_opencl_c_version < 200) {
    std::cerr << "UNIMPLEMENTED: group operations require OpenCL C 2.0\n";
    return false;
  }
  if (*scope == SpvScopeSubgroup) {
    enable_extension("cl_khr_subgroups");
  }
  // Elect only requires GroupNonUniform in SPIR-V, but its OpenCL builtin
  // belongs to the non-uniform vote extension.
  if (inst.opcode() == spv::Op::OpGroupNonUniformElect) {
    enable_extension("cl_khr_subgroup_non_uniform_vote");
  }
  std::string name = *scope == SpvScopeSubgroup ? "sub_group_" : "work_group_";
  unsigned first_arg = 3;
  if (fn.arithmetic) {
    auto operation = inst.GetSingleWordOperand(3);
    const bool ballot =
        inst.opcode() == spv::Op::OpGroupNonUniformBallotBitCount;
    if (!uniform && !ballot) {
      enable_extension(operation == SpvGroupOperationClusteredReduce
                           ? "cl_khr_subgroup_clustered_reduce"
                           : "cl_khr_subgroup_non_uniform_arithmetic");
    }
    if (ballot) {
      name += "ballot_";
    } else if (!uniform && operation != SpvGroupOperationClusteredReduce) {
      name += "non_uniform_";
    }
    switch (operation) {
    case SpvGroupOperationReduce:
      name += ballot ? "bit_count" : "reduce";
      break;
    case SpvGroupOperationInclusiveScan:
      name += ballot ? "inclusive_scan" : "scan_inclusive";
      break;
    case SpvGroupOperationExclusiveScan:
      name += ballot ? "exclusive_scan" : "scan_exclusive";
      break;
    case SpvGroupOperationClusteredReduce:
      if (uniform || ballot) {
        std::cerr << "UNIMPLEMENTED clustered uniform or ballot operation\n";
        return false;
      }
      name += "clustered_reduce";
      break;
    default:
      std::cerr << "UNIMPLEMENTED group operation " << operation << std::endl;
      return false;
    }
    if (!ballot) {
      name += std::string("_") + fn.name;
    }
    first_arg = 4;
    if (!ballot) {
      auto ty = type_for_val(inst.GetSingleWordOperand(first_arg));
      // OpenCL arithmetic collectives have scalar overloads. Uniform integer
      // work-group collectives additionally require at least 32 bits.
      if (ty->AsVector() ||
          (uniform && *scope == SpvScopeWorkgroup && ty->AsInteger() &&
           ty->AsInteger()->width() < 32)) {
        std::cerr << "UNIMPLEMENTED group arithmetic operand type\n";
        return false;
      }
      if (uniform && *scope == SpvScopeSubgroup && ty->AsInteger() &&
          ty->AsInteger()->width() < 32) {
        enable_extension("cl_khr_subgroup_extended_types");
      }
    }
  } else {
    if (inst.opcode() == spv::Op::OpGroupNonUniformRotateKHR &&
        inst.NumOperands() == 6) {
      name += "clustered_";
    }
    name += fn.name;
    if (inst.opcode() == spv::Op::OpGroupBroadcast) {
      auto ty = type_for_val(inst.GetSingleWordOperand(3));
      if (*scope == SpvScopeSubgroup &&
          (ty->AsVector() ||
           (ty->AsInteger() && ty->AsInteger()->width() < 32))) {
        enable_extension("cl_khr_subgroup_extended_types");
      } else if (*scope == SpvScopeWorkgroup &&
                 (ty->AsVector() ||
                  (ty->AsInteger() && ty->AsInteger()->width() < 32))) {
        std::cerr << "UNIMPLEMENTED work-group broadcast operand type\n";
        return false;
      }
    }
  }
  const bool ballot = inst.opcode() >= spv::Op::OpGroupNonUniformBallot &&
                      inst.opcode() <= spv::Op::OpGroupNonUniformBallotFindMSB;
  const bool has_value = inst.opcode() != spv::Op::OpGroupNonUniformElect;
  if (has_value && !ballot && !fn.arithmetic) {
    auto ty = type_for_val(inst.GetSingleWordOperand(first_arg));
    if (ty->AsVector() && inst.opcode() != spv::Op::OpGroupBroadcast &&
        inst.opcode() != spv::Op::OpGroupNonUniformBroadcast) {
      std::cerr << "UNIMPLEMENTED subgroup operand type\n";
      return false;
    }
  }
  std::vector<c::expr_ref> args;
  for (unsigned i = first_arg; i < inst.NumOperands(); ++i) {
    auto id = inst.GetSingleWordOperand(i);
    const bool boolean = type_for_val(id)->kind() == Type::Kind::kBool;
    if (i == first_arg && fn.signed_operand) {
      args.push_back(as_signed(id));
    } else if (inst.opcode() == spv::Op::OpGroupNonUniformRotateKHR && i == 4) {
      args.push_back(as_signed(id));
    } else if (i == first_arg && (fn.predicate || boolean)) {
      args.push_back(c::cast("int", value(id)));
    } else if (inst.opcode() == spv::Op::OpGroupBroadcast && i == 4 &&
               type_for_val(id)->AsVector()) {
      // OpenCL takes a separate coordinate for each work-group dimension.
      auto vec = type_for_val(id)->AsVector();
      for (unsigned j = 0; j < vec->element_count(); ++j) {
        args.push_back(vector_component(id, j));
      }
    } else {
      args.push_back(value(id));
    }
  }
  result = c::call(name, std::move(args));
  if (fn.boolean_result ||
      type_for(inst.type_id())->kind() == Type::Kind::kBool) {
    // OpenCL returns any nonzero int, whereas SPIR-V returns a boolean.
    result = c::binary("!=", result, c::literal("0"));
  } else if (fn.signed_operand) {
    result = cast_to(inst.type_id(), result);
  }
  return true;
}
