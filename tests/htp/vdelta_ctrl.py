"""Control vectors of the Q8_0 pair decode for vdelta / vrdelta (HVX v73 PRM 80-N2040-54 AA p.203:
vdelta runs the stages offset = 64, 32, ..., 1 and vrdelta 1, 2, ..., 64; in each stage
Vd.ub[k] = (Vv.ub[k] & offset) ? Vu.ub[k ^ offset] : Vu.ub[k]).

A mapping dest k <- src s has one path: after the stages with the offsets in `done`, the byte sits
at (k & done) | (s & ~done), and the control bit of that stage at that position is (s ^ k) & offset.
The network can do the mapping (broadcasts included) unless two paths need different bits there.

usage: vdelta_ctrl.py [--c]   (--c prints the C tables)
"""
import sys

DELTA, RDELTA = [64, 32, 16, 8, 4, 2, 1], [1, 2, 4, 8, 16, 32, 64]


def solve(src, offs):
    ctrl, need = [0] * 128, {}
    for k, s in enumerate(src):
        if s is None:
            continue
        done = 0
        for o in offs:
            done |= o
            x = (k & done) | (s & ~done & 127)
            bit = (s ^ k) & o
            if need.get((o, x), bit) != bit:
                return None
            need[(o, x)] = bit
            if bit:
                ctrl[x] |= o
    return ctrl


def apply(v, ctrl, offs):
    v = list(v)
    for o in offs:
        v = [v[k ^ o] if ctrl[k] & o else v[k] for k in range(128)]
    return v


# One unaligned vector load at the first block of a pair: d0 at bytes 0..1, qs0 at 2..33,
# d1 at 34..35, qs1 at 36..67.
#   codes: bytes 0..31 = qs0, 32..63 = qs1 (the lower half of vunpack gives the 64 halfwords)
#   scale: halfwords 0..31 = d0, 32..63 = d1
MAPS = {
    'codes': ([2 + k if k < 32 else 4 + k if k < 64 else None for k in range(128)], RDELTA, 'vrdelta'),
    'scale': ([(k & 1) if k < 64 else 34 + (k & 1) for k in range(128)], DELTA, 'vdelta'),
}

if __name__ == '__main__':
    out = {}
    for name, (src, offs, op) in MAPS.items():
        c = solve(src, offs)
        if c is None:
            sys.exit(f'{name}: no {op} control')
        got = apply(range(128), c, offs)
        assert all(s is None or got[k] == s for k, s in enumerate(src)), name
        out[name] = (c, op)
        print(f'{name}: {op} ok', file=sys.stderr)
    if '--c' in sys.argv:
        for name, (c, op) in out.items():
            print(f'// {op} control (vdelta_ctrl.py)')
            print(f'static const uint8_t hmx_q8_{name}_ctrl[128] __attribute__((aligned(128))) = {{')
            for i in range(0, 128, 16):
                print('    ' + ', '.join(f'0x{x:02x}' for x in c[i:i + 16]) + ',')
            print('};')
