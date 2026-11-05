#!/usr/bin/env bash
# Import an official llama.cpp release (and optionally new NeMo-Speech.cpp / SenseVoice pins) into
# an import branch in its own worktree, so the main checkout and its work stay untouched.
#
#   scripts/llama-release.sh import <llama tag> <worktree> <work dir> [--nemo <rev>] [--sensevoice <rev>]
#   scripts/llama-release.sh review-ops <old llama rev> <new llama rev> [<tree>]
#
# import, run again after each stop until it prints "import ready":
#   1. resolves the tag to its commit (fetching tags into the llama.cpp submodule)
#   2. scripts/vendor-update.sh: the vendor branch gets the upstream backend unmodified
#   3. creates branch import/llama-<tag> from main in <worktree> and merges vendor without
#      squashing (skipped when the backend did not change); a conflict stops here: resolve,
#      `git add`, then run import again (it commits the merge)
#   4. moves the submodule pins in the worktree (submodules are cloned with --reference to the
#      main checkout's)
#   5. moves the patch series onto the new pins with scripts/rebase-patches.sh, in <work dir>:
#      tests/patches (llama.cpp, conflicts resolved by union: the patches add test cases),
#      asr/patches (NeMo), sensevoice/patches (SenseVoice); a conflict in the latter two stops
#   6. review-ops: the ops the HTP newly takes, and whether the functional-safety op budget or a
#      yield point covers each one (<work dir>/review-ops.txt)
# Nothing is committed except the vendor branch and the merge; nothing is pushed.
#
# review-ops lists the GGML ops that htp_op_remap() maps in <new> but not in <old>, and for each
# whether htp-budget.cpp has a case for it, scripts/op-budget.py counts it as yielding, or its DSP
# source yields. An op with none of these may hold VTCM/HMX for a whole long op.
set -euo pipefail

die() { echo "llama-release: $*" >&2; exit 1; }
top=$(git rev-parse --show-toplevel)

remap_ops() { # rev in llama.cpp -> GGML op names the HTP maps
    git -C "$top/llama.cpp" show "$1:ggml/src/ggml-hexagon/htp-kparams.cpp" 2>/dev/null |
        sed -n 's/^ *case GGML_OP_\([A-Z0-9_]*\): *return HTP_OP_.*/\1/p' | sort -u
}

review_ops() { # old new tree
    local old=$1 new=$2 tree=${3:-$top}
    local budget=$tree/ggml/src/ggml-hexagon/htp-budget.cpp yielding=$tree/scripts/op-budget.py
    echo "# HTP ops in llama.cpp ${new:0:9} that ${old:0:9} did not map"
    local added
    added=$(comm -13 <(remap_ops "$old") <(remap_ops "$new"))
    [ -n "$added" ] || { echo "(none)"; return 0; }
    printf '%-22s %-7s %-9s %s\n' op budget yielding note
    for op in $added; do
        local b=no y=no note=""
        grep -q "case GGML_OP_$op:" "$budget" && b=yes
        grep -q "\"$op[\":]" "$yielding" && y=yes
        if [ $b = no ] && [ $y = no ]; then note="REVIEW: no budget estimate and no yield point"; fi
        printf '%-22s %-7s %-9s %s\n' "$op" "$b" "$y" "$note"
    done
    echo
    echo "# DSP sources upstream changed between the two revisions, with yield points in this tree"
    for f in $(git -C "$top/llama.cpp" diff --name-only "$old" "$new" -- ggml/src/ggml-hexagon/htp | grep '\.c$'); do
        local p=$tree/$f
        [ -f "$p" ] || continue
        printf '%-48s work_queue_run %2d  yield %d\n' "${f#ggml/src/ggml-hexagon/}" \
            "$(grep -c work_queue_run "$p" || true)" "$(grep -c 'htp_vtcm_yield' "$p" || true)"
    done
}

cmd=${1:-}
case "$cmd" in
review-ops)
    [ $# -ge 3 ] || die "usage: $0 review-ops <old rev> <new rev> [<tree>]"
    review_ops "$2" "$3" "${4:-$top}"
    exit 0 ;;
import) ;;
*) die "usage: $0 import <tag> <worktree> <work dir> [--nemo <rev>] [--sensevoice <rev>] | review-ops ..." ;;
esac

[ $# -ge 4 ] || die "usage: $0 import <llama tag> <worktree> <work dir> [--nemo <rev>] [--sensevoice <rev>]"
tag=$2 wt=$3 work=$4; shift 4
nemo="" sv=""
while [ $# -gt 0 ]; do
    case $1 in
        --nemo) nemo=${2:?}; shift ;;
        --sensevoice) sv=${2:?}; shift ;;
        *) die "unknown option $1" ;;
    esac
    shift
done
mkdir -p "$work"; work=$(cd "$work" && pwd)
branch=import/llama-$tag

# 1. the release
[[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || echo "llama-release: warning: $tag is not an official vX.Y.Z tag" >&2
git -C "$top/llama.cpp" rev-parse -q --verify "refs/tags/$tag" > /dev/null || git -C "$top/llama.cpp" fetch -q --tags origin
new=$(git -C "$top/llama.cpp" rev-parse "$tag^{commit}")
old=$(git -C "$top" ls-tree main -- llama.cpp | awk '{print $3}')
echo "llama.cpp: ${old:0:9} (main) -> ${new:0:9} ($tag), $(git -C "$top/llama.cpp" rev-list --count "$old..$new") commits"

# 2. vendor
if [ "$(git -C "$top" log -1 --format=%s vendor)" != "vendor: ggml-hexagon from llama.cpp ${new:0:9}" ]; then
    "$top/scripts/vendor-update.sh" "$new"
fi

# 3. import branch, merge
if [ ! -e "$wt/.git" ]; then
    git -C "$top" worktree add -b "$branch" "$wt" main
fi
[ "$(git -C "$wt" branch --show-current)" = "$branch" ] || die "$wt is not on $branch"
if git -C "$wt" rev-parse -q --verify MERGE_HEAD > /dev/null; then
    if [ -n "$(git -C "$wt" diff --name-only --diff-filter=U)" ]; then
        echo "merge conflicts left in $wt:" >&2; git -C "$wt" diff --name-only --diff-filter=U >&2; exit 2
    fi
    git -C "$wt" commit -q --no-edit -m "Merge vendor: ggml-hexagon from llama.cpp ${new:0:9} ($tag)"
    echo "merge committed: $(git -C "$wt" rev-parse --short HEAD)"
elif ! git -C "$wt" merge-base --is-ancestor vendor HEAD; then
    if ! git -C "$wt" merge -q --no-ff --no-commit vendor; then
        echo "merge conflicts in $wt (resolve, git add, run import again):" >&2
        git -C "$wt" diff --name-only --diff-filter=U >&2
        exit 2
    fi
    git -C "$wt" commit -q --no-edit -m "Merge vendor: ggml-hexagon from llama.cpp ${new:0:9} ($tag)"
    echo "merge committed: $(git -C "$wt" rev-parse --short HEAD)"
else
    echo "vendor already merged"
fi

# 4. submodule pins
for s in llama.cpp NeMo-Speech.cpp SenseVoice; do
    if [ ! -e "$wt/$s/.git" ]; then
        git -C "$wt" submodule update -q --init --reference "$top/$s" -- "$s"
    fi
done
pin() { # submodule rev
    git -C "$wt/$1" cat-file -e "$2^{commit}" 2>/dev/null || git -C "$wt/$1" fetch -q "$top/$1" "$2" 2>/dev/null ||
        git -C "$wt/$1" fetch -q --tags origin
    git -C "$wt/$1" checkout -q "$(git -C "$wt/$1" rev-parse "$2^{commit}")"
}
pin llama.cpp "$new"
[ -z "$nemo" ] || { git -C "$top/NeMo-Speech.cpp" cat-file -e "$nemo^{commit}" 2>/dev/null || git -C "$top/NeMo-Speech.cpp" fetch -q --tags origin; pin NeMo-Speech.cpp "$nemo"; }
[ -z "$sv" ] || { git -C "$top/SenseVoice" cat-file -e "$sv^{commit}" 2>/dev/null || git -C "$top/SenseVoice" fetch -q origin; pin SenseVoice "$sv"; }

# 5. patch series
series() { # name submodule patch-dir union?
    local name=$1 sub=$2 pdir=$3 union=$4 w=$work/rebase-$1
    local want; want=$(git -C "$wt/$sub" rev-parse HEAD)
    [ "$(git -C "$wt" ls-tree HEAD -- "$sub" | awk '{print $3}')" != "$want" ] || return 0
    if [ -f "$w/exported" ] && [ "$(cat "$w/exported")" = "$want" ]; then
        return 0
    fi
    if [ ! -d "$w/repo" ]; then
        (cd "$wt" && scripts/rebase-patches.sh start "$sub" "$pdir" "$want" "$w") || {
            [ "$union" = union ] || { echo "resolve the $name series in $w/repo, then run import again" >&2; exit 2; }
        }
    fi
    if [ -d "$(git -C "$w/repo" rev-parse --path-format=absolute --git-path rebase-merge)" ]; then
        if [ "$union" = union ]; then
            (cd "$wt" && scripts/rebase-patches.sh continue "$w" --union)
        else
            (cd "$wt" && scripts/rebase-patches.sh continue "$w") || exit 2
        fi
    fi
    (cd "$wt" && scripts/rebase-patches.sh export "$w")
    echo "$want" > "$w/exported"
}
series tests llama.cpp tests/patches union
series asr NeMo-Speech.cpp asr/patches no
series sensevoice SenseVoice sensevoice/patches no

# 6. ops review
review_ops "$old" "$new" "$wt" | tee "$work/review-ops.txt"
echo
echo "import ready in $wt ($branch): build and test, then commit the pins, patches and VENDOR.md"
grep -q REVIEW "$work/review-ops.txt" && echo "note: ops marked REVIEW in $work/review-ops.txt need a budget estimate or a yield point"
exit 0
