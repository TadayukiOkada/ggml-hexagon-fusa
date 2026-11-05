#!/usr/bin/env bash
# Move a patch series (asr/patches, sensevoice/patches) to a new upstream
# revision of its submodule, without touching the submodule.
#
#   scripts/rebase-patches.sh start  <submodule> <patch-dir> <new-rev> <work-dir>
#   scripts/rebase-patches.sh continue <work-dir> [--union]
#   scripts/rebase-patches.sh export <work-dir> [<patch-dir>]
#
# start clones the submodule into <work-dir>/repo, applies each patch of
# <patch-dir> (in sorted order) as one commit on top of the revision the
# superproject pins now, and rebases the series onto <new-rev>. On a conflict
# it stops: resolve in <work-dir>/repo and run continue (repeat until done).
# continue --union resolves each conflict as "ours, then the patch's", which
# is right for series that only add code at the same places (test cases).
# export writes every commit of the series back to its patch file, as
# `git diff` output in the same file names, and compares the lines the whole
# series adds and removes with the old series (line-check.txt): a resolution
# that dropped or duplicated a line shows up there.
set -euo pipefail

die() { echo "rebase-patches: $*" >&2; exit 1; }

cmd=${1:-}
case "$cmd" in
start)
    [ $# -eq 5 ] || die "usage: $0 start <submodule> <patch-dir> <new-rev> <work-dir>"
    sub=$2 pdir=$(cd "$3" && pwd) new=$4 work=$5
    top=$(git rev-parse --show-toplevel)
    old=$(git -C "$top" ls-tree HEAD -- "$sub" | awk '$2 == "commit" { print $3 }')
    [ -n "$old" ] || die "$sub is not a submodule of $top"
    [ ! -e "$work/repo" ] || die "$work/repo exists"
    mkdir -p "$work"
    git clone -q --shared --no-checkout "$top/$sub" "$work/repo"
    repo=$work/repo
    git -C "$repo" cat-file -e "$new^{commit}" 2>/dev/null || die "$new is not in $sub (fetch it there first)"
    newfull=$(git -C "$repo" rev-parse "$new^{commit}")
    git -C "$repo" checkout -q -b patches "$old"
    git -C "$repo" -c user.name=rebase-patches -c user.email=rebase-patches@localhost commit -q --allow-empty -m "base $old"
    base=$(git -C "$repo" rev-parse HEAD)
    for p in $(ls "$pdir"/*.patch | sort); do
        git -C "$repo" apply --index --whitespace=nowarn "$p" || die "$(basename "$p") does not apply to the pinned $old"
        git -C "$repo" -c user.name=rebase-patches -c user.email=rebase-patches@localhost commit -q -m "$(basename "$p")"
    done
    tip=$(git -C "$repo" rev-parse HEAD)
    printf 'sub=%s\npdir=%s\nold=%s\nnew=%s\nbase=%s\ntip=%s\n' "$sub" "$pdir" "$old" "$newfull" "$base" "$tip" > "$work/state"
    echo "series of $(ls "$pdir"/*.patch | wc -l) patches applied on ${old:0:9}; rebasing onto ${newfull:0:9}"
    # the empty base commit is left behind: only the patch commits move
    if git -C "$repo" -c user.name=rebase-patches -c user.email=rebase-patches@localhost \
           rebase -q --onto "$newfull" "$base" patches; then
        echo "rebased without conflicts; next: $0 export $work"
    else
        echo "conflict: resolve in $repo, then $0 continue $work [--union]" >&2
        exit 2
    fi
    ;;
continue)
    [ $# -ge 2 ] || die "usage: $0 continue <work-dir> [--union]"
    work=$2 union=${3:-}
    repo=$work/repo
    while [ -d "$(git -C "$repo" rev-parse --path-format=absolute --git-path rebase-merge)" ]; do
        files=$(git -C "$repo" diff --name-only --diff-filter=U)
        if [ -n "$files" ]; then
            [ "$union" = --union ] || die "unresolved: $files"
            for f in $files; do
                python3 - "$repo/$f" <<'PY'
# Ours, then the patch's. Git leaves the lines both sides end with (closing braces) out of the
# conflict, after it: when our side opens more braces than it closes, those following lines up to
# the balance (and a blank line) are repeated between the two sides, so the patch's code does
# not end up inside our last block.
import sys
p = sys.argv[1]
lines = open(p).read().split("\n")
out, i, n = [], 0, 0
bal = lambda ls: sum(l.count("{") - l.count("}") for l in ls)
while i < len(lines):
    if not lines[i].startswith("<<<<<<< "):
        out.append(lines[i]); i += 1; continue
    j = lines.index("=======", i)
    k = next(x for x in range(j, len(lines)) if lines[x].startswith(">>>>>>> "))
    ours, theirs = lines[i + 1:j], lines[j + 1:k]
    b, close, x, acc = bal(ours), [], k + 1, 0
    if b > 0:
        while x < len(lines) and acc > -b:
            close.append(lines[x]); acc += bal([lines[x]]); x += 1
        if acc == -b:
            if x < len(lines) and lines[x].strip() == "":
                close.append("")
        else:
            close = []
    out += ours + close + theirs
    i, n = k + 1, n + 1
open(p, "w").write("\n".join(out))
print(f"union: {p}: {n} conflicts")
PY
                git -C "$repo" add -- "$f"
            done
        fi
        if GIT_EDITOR=true git -C "$repo" -c user.name=rebase-patches -c user.email=rebase-patches@localhost \
               rebase --continue >/dev/null 2>"$work/rebase.err"; then
            :
        elif [ -z "$(git -C "$repo" diff --name-only --diff-filter=U)" ]; then
            cat "$work/rebase.err" >&2; die "rebase --continue failed"
        fi
    done
    echo "rebase done; next: $0 export $work"
    ;;
export)
    [ $# -ge 2 ] || die "usage: $0 export <work-dir> [<patch-dir>]"
    work=$2
    . "$work/state"
    [ $# -ge 3 ] && pdir=$(cd "$3" && pwd)
    repo=$work/repo
    [ ! -d "$(git -C "$repo" rev-parse --path-format=absolute --git-path rebase-merge)" ] || die "rebase still in progress in $repo"
    [ "$(git -C "$repo" merge-base "$new" refs/heads/patches)" = "$new" ] || die "series in $repo is not on $new"
    # braces a patch opens minus those it closes, in C/C++ files: a conflict resolution that put
    # code at the wrong place usually changes it
    braces() { git -C "$repo" diff --no-color -U0 "$1" "$2" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' |
        python3 -c 'import sys
b = 0
for l in sys.stdin:
    if l[:1] in "+-" and l[:3] not in ("+++", "---"):
        d = l.count("{") - l.count("}")
        b += d if l[0] == "+" else -d
print(b)'; }
    declare -A oldb
    for c in $(git -C "$repo" rev-list --reverse "$base..$tip"); do
        oldb[$(git -C "$repo" show -s --format=%s "$c")]=$(braces "$c^" "$c")
    done
    n=0 bad=0
    for c in $(git -C "$repo" rev-list --reverse "$new..refs/heads/patches"); do
        name=$(git -C "$repo" show -s --format=%s "$c")
        case "$name" in *.patch) ;; *) die "commit $c has no patch file name as subject" ;; esac
        b=$(braces "$c^" "$c")
        if [ "$b" != "${oldb[$name]:-$b}" ]; then
            echo "brace check: $name opens $b more braces than it closes, ${oldb[$name]} before" >&2
            bad=1
        fi
        git -C "$repo" diff --no-color --no-ext-diff "$c^" "$c" > "$pdir/$name"
        [ -s "$pdir/$name" ] || die "$name became empty (already upstream?): remove it by hand"
        n=$((n + 1))
    done
    [ $bad -eq 0 ] || die "fix the series in $repo (code moved into another block?) and export again"
    # lines added (+) and removed (-) by the whole series, old and new, as sorted multisets
    lines() { git -C "$repo" diff --no-color -U0 "$1" "$2" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---) ' | sort; }
    diff <(lines "$base" "$tip") <(lines "$new" refs/heads/patches) > "$work/line-check.txt" || true
    if [ -s "$work/line-check.txt" ]; then
        echo "line check: $(grep -c '^[<>]' "$work/line-check.txt") lines differ from the old series, review $work/line-check.txt"
    else
        echo "line check: the series adds and removes the same lines as before"
    fi
    echo "wrote $n patches to $pdir for ${new:0:9}; next: move the submodule to $new and build"
    ;;
*)
    die "usage: $0 start|export ..."
    ;;
esac
