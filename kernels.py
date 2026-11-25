"""Benchmark kernels for the ASUNDER study.

Each kernel is a loop body written as a small data-flow graph over fixed-width
words. Operand references:
  ('in', name)                  scalar input port
  ('arr', name, stride, off)    element (iter*stride+off) mod len of input array
  ('carr', name, stride, off)   element of a constant array
  ('car', var)                  loop-carried variable
  ('op', dst)                   result of a body operation
  ('const', value, width)
  ('iter',)                     loop iteration index
  ('free', fop, ref)            wiring-only transform of another reference
Every kernel also carries a bit-exact Python golden model.
"""
import random

M = lambda w: (1 << w) - 1


def _aes_sbox():
    def gmul(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return p
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if gmul(a, b) == 1:
                inv[a] = b
                break
    s = []
    for a in range(256):
        x = inv[a]
        y = x
        for i in range(1, 5):
            y ^= ((x << i) | (x >> (8 - i))) & 0xFF
        s.append(y ^ 0x63)
    return s


AES_SBOX = _aes_sbox()
PRESENT_SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
# 16-bit PRESENT-style bit permutation P(i) = 4*i mod 15, P(15)=15
PERM16 = [(4 * i) % 15 if i != 15 else 15 for i in range(16)]
ATAN12 = [256, 151, 80, 41, 20, 10]  # round(atan(2^-i)/(pi/4)*256), 12-bit angle scale
FIR_H = [23, 87, 87, 23]


def xtime(a):
    return ((a << 1) & 0xFF) ^ (0x1B if a & 0x80 else 0)


def crcstep(c):
    return ((c << 1) & 0xFFFF) ^ (0x1021 if c & 0x8000 else 0)


def sbox4x4(v):
    r = 0
    for n in range(4):
        r |= PRESENT_SBOX[(v >> (4 * n)) & 0xF] << (4 * n)
    return r


def perm16(v):
    r = 0
    for i in range(16):
        if (v >> i) & 1:
            r |= 1 << PERM16[i]
    return r


def sra(v, s, w):
    if v & (1 << (w - 1)):
        v = v - (1 << w)
    return (v >> s) & M(w)


def addsub(a, b, sel, w):
    return ((a - b) if sel else (a + b)) & M(w)


KERNELS = {}


def kernel(fn):
    k = fn()
    KERNELS[k['name']] = k
    return fn


@kernel
def _aes_ark():
    def golden(I):
        return {'o': [AES_SBOX[I['p'][i] ^ I['k'][i]] for i in range(4)]}
    return dict(name='aes_ark', domain='crypto', N=4,
                inputs={}, arrays={'p': (8, 4), 'k': (8, 4)}, carr={}, carried={},
                body=[('t1', 'xor', [('arr', 'p', 1, 0), ('arr', 'k', 1, 0)], 8),
                      ('t2', 'sbox8', [('op', 't1')], 8)],
                next={}, outarr={'o': (('op', 't2'), 8)}, outs={}, golden=golden)


@kernel
def _aes_mix():
    def golden(I):
        a = I['a']
        return {'o': [xtime(a[i]) ^ xtime(a[(i + 1) % 4]) ^ a[(i + 1) % 4] ^ a[(i + 2) % 4] ^ a[(i + 3) % 4]
                      for i in range(4)]}
    A = lambda o: ('arr', 'a', 1, o)
    return dict(name='aes_mix', domain='crypto', N=4,
                inputs={}, arrays={'a': (8, 4)}, carr={}, carried={},
                body=[('x1', 'xtime', [A(0)], 8), ('x2', 'xtime', [A(1)], 8),
                      ('t', 'xor', [('op', 'x2'), A(1)], 8), ('u', 'xor', [('op', 'x1'), ('op', 't')], 8),
                      ('v', 'xor', [A(2), A(3)], 8), ('w', 'xor', [('op', 'u'), ('op', 'v')], 8)],
                next={}, outarr={'o': (('op', 'w'), 8)}, outs={}, golden=golden)


@kernel
def _present():
    def golden(I):
        s = I['p0']
        for i in range(4):
            s = perm16(sbox4x4(s ^ I['k'][i]))
        return {'S': s}
    return dict(name='present', domain='crypto', N=4,
                inputs={'p0': 16}, arrays={'k': (16, 4)}, carr={}, carried={'s': (('in', 'p0'), 16)},
                body=[('t1', 'xor', [('car', 's'), ('arr', 'k', 1, 0)], 16),
                      ('t2', 'sbox4x4', [('op', 't1')], 16)],
                next={'s': ('free', 'perm16', ('op', 't2'))}, outarr={}, outs={'S': 's'}, golden=golden)


@kernel
def _speck():
    def golden(I):
        x, y = I['x0'], I['y0']
        for i in range(4):
            x = ((((x >> 7) | (x << 9)) & 0xFFFF) + y) & 0xFFFF
            x ^= I['k'][i]
            y = (((y << 2) | (y >> 14)) & 0xFFFF) ^ x
        return {'X': x, 'Y': y}
    return dict(name='speck', domain='crypto', N=4,
                inputs={'x0': 16, 'y0': 16}, arrays={'k': (16, 4)}, carr={},
                carried={'x': (('in', 'x0'), 16), 'y': (('in', 'y0'), 16)},
                body=[('b', 'add', [('free', 'ror:7', ('car', 'x')), ('car', 'y')], 16),
                      ('c', 'xor', [('op', 'b'), ('arr', 'k', 1, 0)], 16),
                      ('e', 'xor', [('free', 'rol:2', ('car', 'y')), ('op', 'c')], 16)],
                next={'x': ('op', 'c'), 'y': ('op', 'e')}, outarr={}, outs={'X': 'x', 'Y': 'y'},
                golden=golden)


@kernel
def _crc16():
    def golden(I):
        c = 0xFFFF
        for i in range(4):
            c ^= I['d'][i] << 8
            for _ in range(8):
                c = crcstep(c)
        return {'C': c}
    body = [('t0', 'xor', [('car', 'c'), ('free', 'shl8w16', ('arr', 'd', 1, 0))], 16)]
    for j in range(1, 9):
        body.append((f't{j}', 'crcstep', [('op', f't{j-1}')], 16))
    return dict(name='crc16', domain='dsp', N=4,
                inputs={}, arrays={'d': (8, 4)}, carr={}, carried={'c': (('const', 0xFFFF, 16), 16)},
                body=body, next={'c': ('op', 't8')}, outarr={}, outs={'C': 'c'}, golden=golden)


@kernel
def _fir4():
    def golden(I):
        acc = 0
        for i in range(4):
            acc = (acc + I['x'][i] * FIR_H[i]) & 0xFFFF
        return {'Y': acc}
    return dict(name='fir4', domain='dsp', N=4,
                inputs={}, arrays={'x': (8, 4)}, carr={'h': (8, FIR_H)},
                carried={'acc': (('const', 0, 16), 16)},
                body=[('p', 'mul', [('free', 'zext16', ('arr', 'x', 1, 0)),
                                    ('free', 'zext16', ('carr', 'h', 1, 0))], 16),
                      ('a', 'add', [('car', 'acc'), ('op', 'p')], 16)],
                next={'acc': ('op', 'a')}, outarr={}, outs={'Y': 'acc'}, golden=golden)


@kernel
def _dct4():
    W = 10

    def golden(I):
        x = I['x']
        out = {f'y{j}': [] for j in range(4)}
        for b in range(2):
            x0, x1, x2, x3 = x[4 * b:4 * b + 4]
            s0, s1 = (x0 + x3) & M(W), (x1 + x2) & M(W)
            d0, d1 = (x0 - x3) & M(W), (x1 - x2) & M(W)
            out['y0'].append((s0 + s1) & M(W))
            out['y2'].append((s0 - s1) & M(W))
            out['y1'].append(((d0 << 1) + d1) & M(W))
            out['y3'].append((d0 - (d1 << 1)) & M(W))
        return out
    X = lambda o: ('free', 'zext10', ('arr', 'x', 4, o))
    return dict(name='dct4', domain='dsp', N=2,
                inputs={}, arrays={'x': (8, 8)}, carr={}, carried={},
                body=[('s0', 'add', [X(0), X(3)], W), ('s1', 'add', [X(1), X(2)], W),
                      ('d0', 'sub', [X(0), X(3)], W), ('d1', 'sub', [X(1), X(2)], W),
                      ('y0', 'add', [('op', 's0'), ('op', 's1')], W),
                      ('y2', 'sub', [('op', 's0'), ('op', 's1')], W),
                      ('y1', 'add', [('free', 'shl1w10', ('op', 'd0')), ('op', 'd1')], W),
                      ('y3', 'sub', [('op', 'd0'), ('free', 'shl1w10', ('op', 'd1'))], W)],
                next={}, outarr={f'y{j}': (('op', f'y{j}'), W) for j in range(4)}, outs={},
                golden=golden)


@kernel
def _cordic():
    W = 12

    def golden(I):
        x, y, z = I['x0'], I['y0'], I['z0']
        for i in range(6):
            d = (z >> (W - 1)) & 1
            xs, ys = sra(y, i, W), sra(x, i, W)
            x, y, z = addsub(x, xs, 1 - d, W), addsub(y, ys, d, W), addsub(z, ATAN12[i], 1 - d, W)
        return {'X': x, 'Y': y, 'Z': z}
    nd = ('free', 'notmsb12', ('car', 'z'))
    dd = ('free', 'msb12', ('car', 'z'))
    return dict(name='cordic', domain='dsp', N=6,
                inputs={'x0': W, 'y0': W, 'z0': W}, arrays={}, carr={'at': (W, ATAN12)},
                carried={'x': (('in', 'x0'), W), 'y': (('in', 'y0'), W), 'z': (('in', 'z0'), W)},
                body=[('xs', 'sra', [('car', 'y'), ('iter',)], W),
                      ('ys', 'sra', [('car', 'x'), ('iter',)], W),
                      ('xn', 'addsub', [('car', 'x'), ('op', 'xs'), nd], W),
                      ('yn', 'addsub', [('car', 'y'), ('op', 'ys'), dd], W),
                      ('zn', 'addsub', [('car', 'z'), ('carr', 'at', 1, 0), nd], W)],
                next={'x': ('op', 'xn'), 'y': ('op', 'yn'), 'z': ('op', 'zn')}, outarr={},
                outs={'X': 'x', 'Y': 'y', 'Z': 'z'}, golden=golden)


def random_inputs(k, rng):
    I = {}
    for n, w in k['inputs'].items():
        I[n] = rng.getrandbits(w)
    for n, (w, L) in k['arrays'].items():
        I[n] = [rng.getrandbits(w) for _ in range(L)]
    return I


def output_layout(k):
    """Ordered list of (name, index or None, width) for the packed output vector."""
    lay = []
    for n, (_, w) in k['outarr'].items():
        for j in range(k['N']):
            lay.append((n, j, w))
    for n, var in k['outs'].items():
        lay.append((n, None, k['carried'][var][1]))
    return lay


def input_layout(k):
    lay = []
    for n, w in k['inputs'].items():
        lay.append((n, None, w))
    for n, (w, L) in k['arrays'].items():
        for j in range(L):
            lay.append((n, j, w))
    return lay


def pack(layout, D):
    v, sh = 0, 0
    for n, j, w in layout:
        x = D[n] if j is None else D[n][j]
        v |= (x & M(w)) << sh
        sh += w
    return v, sh


if __name__ == '__main__':
    rng = random.Random(0)
    for name, k in KERNELS.items():
        I = random_inputs(k, rng)
        print(name, k['golden'](I))
