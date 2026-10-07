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

#include "cexpr.h"

#include <cassert>
#include <unordered_map>

namespace spirv2clc {
namespace c {

namespace {

expr_ref make(expr::kind k, std::string text, std::vector<expr_ref> ops = {}) {
  return std::make_shared<const expr>(expr{k, std::move(text), std::move(ops)});
}

// C precedence levels, loosest first.
enum prec : int {
  comma = 1,
  assignment,
  conditional,
  logical_or,
  logical_and,
  bitwise_or,
  bitwise_xor,
  bitwise_and,
  equality,
  relational,
  shift,
  additive,
  multiplicative,
  prefix, // unary operators and casts
  postfix,
};

int binary_precedence(const std::string &op) {
  static const std::unordered_map<std::string, int> table = {
      {"*", multiplicative}, {"/", multiplicative}, {"%", multiplicative},
      {"+", additive},       {"-", additive},       {"<<", shift},
      {">>", shift},         {"<", relational},     {"<=", relational},
      {">", relational},     {">=", relational},    {"==", equality},
      {"!=", equality},      {"&", bitwise_and},    {"^", bitwise_xor},
      {"|", bitwise_or},     {"&&", logical_and},   {"||", logical_or},
      {"=", assignment},
  };
  auto it = table.find(op);
  assert(it != table.end() && "unknown binary operator");
  return it->second;
}

int precedence(const expr &e) {
  switch (e.k) {
  case expr::kind::name:
  case expr::kind::call:
  case expr::kind::member:
  case expr::kind::arrow:
  case expr::kind::index:
  case expr::kind::compound:
  case expr::kind::init:
    return postfix;
  case expr::kind::literal:
    return !e.text.empty() && e.text[0] == '-' ? prefix : postfix;
  case expr::kind::unary:
  case expr::kind::cast:
  case expr::kind::vector: // syntactically a cast of a parenthesized list
    return prefix;
  case expr::kind::binary:
    return binary_precedence(e.text);
  case expr::kind::ternary:
    return conditional;
  }
  return comma;
}

std::string print_at(const expr_ref &e, int min_prec);

std::string print_list(const std::vector<expr_ref> &ops) {
  std::string s;
  for (size_t i = 0; i < ops.size(); i++) {
    s += (i ? ", " : "") + print_at(ops[i], assignment);
  }
  return s;
}

std::string print_unparenthesized(const expr &e) {
  switch (e.k) {
  case expr::kind::name:
  case expr::kind::literal:
    return e.text;
  case expr::kind::unary: {
    auto operand = print_at(e.ops[0], prefix);
    // Keep the operator from fusing with the operand's leading one into a
    // different token: `- -x` must not become `--x`, nor `& &x` `&&x`.
    if (!operand.empty() && operand[0] == e.text.back() &&
        (operand[0] == '-' || operand[0] == '+' || operand[0] == '&')) {
      operand = "(" + operand + ")";
    }
    return e.text + operand;
  }
  case expr::kind::binary: {
    // Left-associative, except assignment.
    int p = binary_precedence(e.text);
    bool right_assoc = p == assignment;
    return print_at(e.ops[0], right_assoc ? p + 1 : p) + " " + e.text + " " +
           print_at(e.ops[1], right_assoc ? p : p + 1);
  }
  case expr::kind::ternary:
    return print_at(e.ops[0], logical_or) + " ? " +
           print_at(e.ops[1], assignment) + " : " +
           print_at(e.ops[2], conditional);
  case expr::kind::cast:
    return "(" + e.text + ")" + print_at(e.ops[0], prefix);
  case expr::kind::call:
    return e.text + "(" + print_list(e.ops) + ")";
  case expr::kind::member:
    return print_at(e.ops[0], postfix) + "." + e.text;
  case expr::kind::arrow:
    return print_at(e.ops[0], postfix) + "->" + e.text;
  case expr::kind::index:
    return print_at(e.ops[0], postfix) + "[" + print_at(e.ops[1], comma) + "]";
  case expr::kind::vector:
    return "(" + e.text + ")(" + print_list(e.ops) + ")";
  case expr::kind::compound:
    return "(" + e.text + "){" + print_list(e.ops) + "}";
  case expr::kind::init:
    return "{" + print_list(e.ops) + "}";
  }
  return "";
}

std::string print_at(const expr_ref &e, int min_prec) {
  auto s = print_unparenthesized(*e);
  return precedence(*e) < min_prec ? "(" + s + ")" : s;
}

} // namespace

expr_ref name(const std::string &identifier) {
  return make(expr::kind::name, identifier);
}

expr_ref literal(const std::string &text) {
  return make(expr::kind::literal, text);
}

expr_ref unary(const std::string &op, expr_ref operand) {
  return make(expr::kind::unary, op, {std::move(operand)});
}

expr_ref binary(const std::string &op, expr_ref lhs, expr_ref rhs) {
  return make(expr::kind::binary, op, {std::move(lhs), std::move(rhs)});
}

expr_ref ternary(expr_ref cond, expr_ref if_true, expr_ref if_false) {
  return make(expr::kind::ternary, "",
              {std::move(cond), std::move(if_true), std::move(if_false)});
}

expr_ref cast(const std::string &type, expr_ref operand) {
  return make(expr::kind::cast, type, {std::move(operand)});
}

expr_ref call(const std::string &fn, std::vector<expr_ref> args) {
  return make(expr::kind::call, fn, std::move(args));
}

expr_ref index(expr_ref base, expr_ref subscript) {
  return make(expr::kind::index, "", {std::move(base), std::move(subscript)});
}

expr_ref vector_literal(const std::string &type, std::vector<expr_ref> elems) {
  return make(expr::kind::vector, type, std::move(elems));
}

expr_ref compound_literal(const std::string &type,
                          std::vector<expr_ref> inits) {
  return make(expr::kind::compound, type, std::move(inits));
}

expr_ref init_list(std::vector<expr_ref> elems) {
  return make(expr::kind::init, "", std::move(elems));
}

expr_ref deref(expr_ref pointer) {
  if (pointer->k == expr::kind::unary && pointer->text == "&") {
    return pointer->ops[0];
  }
  return unary("*", std::move(pointer));
}

expr_ref address_of(expr_ref lvalue) {
  if (lvalue->k == expr::kind::unary && lvalue->text == "*") {
    return lvalue->ops[0];
  }
  return unary("&", std::move(lvalue));
}

expr_ref member(expr_ref base, const std::string &field) {
  if (base->k == expr::kind::unary && base->text == "*") {
    return make(expr::kind::arrow, field, {base->ops[0]});
  }
  return make(expr::kind::member, field, {std::move(base)});
}

std::string print(const expr_ref &e) { return print_at(e, comma); }

} // namespace c
} // namespace spirv2clc
