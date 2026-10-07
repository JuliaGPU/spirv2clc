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

// C expressions as trees, printed with exactly the parentheses that precedence
// and tokenization require. Building expressions from parts instead of pasting
// strings means an operand never has to be known to be "atomic" to be safe to
// splice in: `-x` of a negative literal prints as `-(-1.5f)`, not `--1.5f`.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace spirv2clc {
namespace c {

struct expr;
using expr_ref = std::shared_ptr<const expr>;

struct expr {
  enum class kind {
    name,     // text
    literal,  // text: a number, string or macro
    unary,    // text ops[0], text one of - ! ~ * &
    binary,   // ops[0] text ops[1]
    ternary,  // ops[0] ? ops[1] : ops[2]
    cast,     // (text)ops[0]
    call,     // text(ops...)
    member,   // ops[0].text
    arrow,    // ops[0]->text
    index,    // ops[0][ops[1]]
    vector,   // (text)(ops...), an OpenCL C vector literal
    compound, // (text){ops...}, a compound literal
    init,     // {ops...}, a brace initializer; not an expression of its own
  };

  kind k;
  std::string text;
  std::vector<expr_ref> ops;
};

expr_ref name(const std::string &identifier);
expr_ref literal(const std::string &text);
expr_ref unary(const std::string &op, expr_ref operand);
expr_ref binary(const std::string &op, expr_ref lhs, expr_ref rhs);
expr_ref ternary(expr_ref cond, expr_ref if_true, expr_ref if_false);
expr_ref cast(const std::string &type, expr_ref operand);
expr_ref call(const std::string &fn, std::vector<expr_ref> args);
expr_ref index(expr_ref base, expr_ref subscript);
expr_ref vector_literal(const std::string &type, std::vector<expr_ref> elems);
expr_ref compound_literal(const std::string &type, std::vector<expr_ref> inits);
expr_ref init_list(std::vector<expr_ref> elems);

// These fold away a dereference next to an address-of, and turn a member of a
// dereference into `->`, so the address arithmetic of access chains reads
// naturally.
expr_ref deref(expr_ref pointer);
expr_ref address_of(expr_ref lvalue);
expr_ref member(expr_ref base, const std::string &field);

std::string print(const expr_ref &e);

} // namespace c
} // namespace spirv2clc
