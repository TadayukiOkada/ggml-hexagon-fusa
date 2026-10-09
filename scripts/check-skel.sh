#!/usr/bin/env bash
# Check an HTP skel for functional-safety firmware: it must carry an SWIV CRC
# section, and if a firmware shell is given, every symbol it imports must be
# exported by that shell.
#
# usage: scripts/check-skel.sh <libggml-htp-vNN.so> <hexagon tools root> [fastrpc_shell_N]
set -euo pipefail

skel=$1; tools=$2; shell=${3:-}
nm=$tools/Tools/bin/hexagon-nm
readelf=$tools/Tools/bin/hexagon-readelf
for f in "$skel" "$nm" "$readelf"; do
    [ -f "$f" ] || { echo "check-skel: not found: $f" >&2; exit 2; }
done

rc=0
if "$readelf" -S -W "$skel" | grep -q 'LOUSER+0x53574956'; then
    echo "check-skel: $(basename "$skel"): SWIV CRC section present"
else
    echo "check-skel: $(basename "$skel"): no SWIV CRC section" >&2; rc=1
fi

if [ -n "$shell" ]; then
    [ -f "$shell" ] || { echo "check-skel: not found: $shell" >&2; exit 2; }
    missing=$(comm -23 <("$nm" -D --undefined-only "$skel" | awk '{print $NF}' | sort -u) \
                       <("$nm" -D --defined-only "$shell" | awk '{print $NF}' | sort -u))
    if [ -n "$missing" ]; then
        echo "check-skel: $(basename "$skel"): imports not exported by $(basename "$shell"):" >&2
        echo "$missing" | sed 's/^/  /' >&2; rc=1
    else
        echo "check-skel: $(basename "$skel"): all imports resolve against $(basename "$shell")"
    fi
else
    echo "check-skel: no firmware shell given (GGML_HEXAGON_FUSA_SHELL), skipping the import check"
fi
exit $rc
