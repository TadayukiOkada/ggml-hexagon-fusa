#!/usr/bin/env python3
"""Byte-permutation controls of the KV cache kernels (vdelta/vrdelta stage rule of the v73 HVX PRM,
80-N2040-54 AA p.203; solver in vdelta_ctrl.py). Prints the C tables:
  kv-cache-quant.h: kv_pack_<type> (vrdelta, four 32-byte staging slots -> four consecutive blocks) and
                    kv_splat_w<k> (vdelta, every word <- word k)
  hmx-kv-prep.h:    hmx_kvg_<type>_{codes,high,scale0,scale1,min0,min1} (vdelta, four-block group decoder)
usage: gen-kv-ctrl.py pack|dec|splat
"""
import sys
from vdelta_ctrl import solve, apply, DELTA, RDELTA


def table(name, c, op=None):
    out = (f'// {op}\n' if op else '') + f'static const uint8_t {name}[128] __attribute__((aligned(128))) = {{\n'
    for i in range(0, 128, 16):
        out += '    ' + ', '.join(f'0x{x:02x}' for x in c[i:i + 16]) + ',\n'
    return out + '};'


def pack():
    L = {
        'q4_0': (18, [('d', 0, 2), ('qs', 2, 16)]),
        'q4_1': (20, [('d', 0, 2), ('m', 2, 2), ('qs', 4, 16)]),
        'q5_0': (22, [('d', 0, 2), ('qh', 2, 4), ('qs', 6, 16)]),
        'q5_1': (24, [('d', 0, 2), ('m', 2, 2), ('qh', 4, 4), ('qs', 8, 16)]),
    }
    for name, (size, fields) in L.items():
        m = [None] * 128
        for k in range(4):
            dm = 16 + 4 * ((k + 1) & 3)
            pos = {'qs': 0, 'qh': 16 + 4 * k, 'd': dm, 'm': dm + 2}
            for f, off, ln in fields:
                for j in range(ln):
                    m[k * size + off + j] = 32 * k + pos[f] + j
        c = solve(m, RDELTA)
        assert c, name
        got = apply(range(128), c, RDELTA)
        assert all(s is None or got[k] == s for k, s in enumerate(m))
        print(table(f'kv_pack_{name}', c))


def dec():
    L = {'q4_0': (18, 0, None, None, 2), 'q4_1': (20, 0, 2, None, 4), 'q5_0': (22, 0, None, 2, 6), 'q5_1': (24, 0, 2, 4, 8)}
    for name, (bs, od, om, oh, oq) in L.items():
        maps = {'codes': [bs * (i // 32) + oq + (i % 32) % 16 for i in range(128)]}
        if oh is not None:
            maps['high'] = [bs * (i // 32) + oh + (i % 32) // 8 for i in range(128)]
        for vec in (0, 1):
            maps[f'scale{vec}'] = [bs * (2 * vec + (k >= 64)) + od + (k & 1) for k in range(128)]
            if om is not None:
                maps[f'min{vec}'] = [bs * (2 * vec + (k >= 64)) + om + (k & 1) for k in range(128)]
        for mname, m in maps.items():
            c = solve(m, DELTA)
            assert c, (name, mname)
            got = apply(range(128), c, DELTA)
            assert all(s is None or got[k] == s for k, s in enumerate(m))
            print(table(f'hmx_kvg_{name}_{mname}', c))


def splat():
    for k in range(4):
        m = [4 * k + (i % 4) for i in range(128)]
        c = solve(m, DELTA)
        assert c and apply(range(128), c, DELTA) == m
        print(table(f'kv_splat_w{k}', c, f'vdelta: every word <- word {k}'))


if __name__ == '__main__':
    {'pack': pack, 'dec': dec, 'splat': splat}[sys.argv[1] if len(sys.argv) > 1 else 'pack']()
