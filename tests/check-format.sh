#!/usr/bin/env bash

GIT_CLANG_FORMAT=${GIT_CLANG_FORMAT:-git-clang-format}

# Revision to check changes against (default: origin/main)
BASE=${1:-origin/main}

# Run git-clang-format to check for violations
CLANG_FORMAT_OUTPUT=$(mktemp /tmp/clang-format-output-XXXXXXXX.txt)
${GIT_CLANG_FORMAT} --diff "$BASE" --extensions cpp,hpp,h >$CLANG_FORMAT_OUTPUT

# Check for no-ops
grep '^no modified files to format$' "$CLANG_FORMAT_OUTPUT" && exit 0
grep '^clang-format did not modify any files$' "$CLANG_FORMAT_OUTPUT" && exit 0

# Dump formatting diff and signal failure
echo -e "\n==== FORMATTING VIOLATIONS DETECTED ====\n"
cat "$CLANG_FORMAT_OUTPUT"
exit 1
