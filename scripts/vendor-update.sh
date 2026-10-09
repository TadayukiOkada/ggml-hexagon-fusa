#!/usr/bin/env bash
# Import the upstream ggml-hexagon backend at <commit> into the vendor branch.
# The vendor branch holds unmodified upstream files only; merge it into main
# afterwards and move the llama.cpp submodule to the same commit.
#
# usage: scripts/vendor-update.sh <llama.cpp commit>
set -euo pipefail

commit=${1:?usage: $0 <llama.cpp commit>}
top=$(git rev-parse --show-toplevel)
sub=$top/llama.cpp
paths=(ggml/src/ggml-hexagon ggml/include/ggml-hexagon.h)

git -C "$sub" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$sub" fetch origin "$commit"
full=$(git -C "$sub" rev-parse "$commit^{commit}")

wt=$(mktemp -d)
trap 'git -C "$top" worktree remove --force "$wt" >/dev/null 2>&1 || true; rm -rf "$wt"' EXIT
git -C "$top" worktree add -q "$wt" vendor

git -C "$wt" rm -rq --ignore-unmatch "${paths[@]}"
git -C "$sub" archive "$full" "${paths[@]}" | tar -x -C "$wt"
git -C "$wt" add -A "${paths[@]}"

if git -C "$wt" diff --cached --quiet; then
    echo "vendor already matches $full"
    exit 0
fi

git -C "$wt" commit -q -m "vendor: ggml-hexagon from llama.cpp ${full:0:9}

Upstream: https://github.com/ggml-org/llama.cpp/commit/$full
Date: $(git -C "$sub" show -s --format=%cs "$full")"
echo "vendor updated to $full: $(git -C "$top" rev-parse --short vendor)"
echo "next: git merge vendor; git -C llama.cpp checkout $full; update VENDOR.md"
