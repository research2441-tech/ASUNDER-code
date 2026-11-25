"""Synthesis to a LUT4/DFF netlist with Yosys and a bit-parallel gate-level simulator."""
import json, os, re, subprocess, hashlib
import numpy as np

YOSYS = os.environ.get('YOSYS', 'yowasp-yosys')
SCRIPT = """read_verilog -sv {v}
synth -flatten -top {top}
dfflegalize -cell $_DFF_P_ 01
abc -lut 4
opt_clean -purge
setundef -zero
write_json {j}
"""


def synthesize(vtext, top, workdir, tag):
    os.makedirs(workdir, exist_ok=True)
    v = os.path.join(workdir, f"{tag}.v")
    j = os.path.join(workdir, f"{tag}.json")
    if os.path.exists(j):
        return json.load(open(j))
    open(v, 'w').write(vtext)
    ys = os.path.join(workdir, f"{tag}.ys")
    open(ys, 'w').write(SCRIPT.format(v=v, top=top, j=j))
    r = subprocess.run([YOSYS, '-q', '-s', ys], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:] + r.stdout[-2000:])
    return json.load(open(j))


class Netlist:
    """Flattened netlist with LUT and FF cells over integer nets.
    Net 0 is constant 0, net 1 is constant 1."""

    def __init__(self):
        self.n = 2
        self.luts = []   # (inputs list, init int, out net, owner, name)
        self.ffs = []    # (d net, q net, init, owner, name)
        self.ports = {}  # (owner, port) -> list of nets

    def new(self):
        self.n += 1
        return self.n - 1

    def add_module(self, js, top, owner, bind=None):
        """Import a Yosys JSON module. `bind` maps port name -> list of existing nets to reuse."""
        m = js['modules'][top]
        bind = bind or {}
        local = {}

        def net(b):
            if isinstance(b, str):
                return 1 if b == '1' else 0
            if b not in local:
                local[b] = self.new()
            return local[b]
        for pn, p in m['ports'].items():
            if pn in bind:
                for b, g in zip(p['bits'], bind[pn]):
                    if isinstance(b, str):
                        continue
                    local[b] = g
        for pn, p in m['ports'].items():
            self.ports[(owner, pn)] = [net(b) for b in p['bits']]
        cand = {}
        for nn, d in m['netnames'].items():
            if d.get('hide_name', 0):
                continue
            for i, b in enumerate(d['bits']):
                if not isinstance(b, str):
                    cand.setdefault(b, []).append((nn, i))

        def rank(t):
            nn = t[0]
            for r, p in enumerate(['R', 'C_', 'O_', 'state', 'cnt', 'valid', 'alarm', 'q']):
                if nn == p or nn.startswith(p) and (p != 'R' or nn[1:].isdigit()):
                    return (r, nn)
            return (99, nn)
        names = {b: "%s[%d]" % min(v, key=rank) for b, v in cand.items()}
        for cn, c in m['cells'].items():
            t = c['type']
            if t == '$lut':
                ins = [net(b) for b in c['connections']['A']]
                init = int(c['parameters']['LUT'], 2) if isinstance(c['parameters']['LUT'], str) else int(c['parameters']['LUT'])
                y = c['connections']['Y'][0]
                self.luts.append((ins, init, net(y), owner, names.get(y, '')))
            elif t == '$_DFF_P_':
                q = c['connections']['Q'][0]
                init = 0
                a = c.get('attributes', {}).get('init')
                if a is not None:
                    init = int(a, 2) & 1 if isinstance(a, str) else int(a) & 1
                self.ffs.append((net(c['connections']['D'][0]), net(q), init, owner, names.get(q, '')))
            else:
                raise ValueError(f"unexpected cell {t}")
        return local

    # -------------------------------------------------------------- compile
    def compile(self):
        """Levelise LUTs and build vectorised evaluation tables."""
        nl = len(self.luts)
        drv = {}
        for i, (ins, init, y, o, n) in enumerate(self.luts):
            drv[y] = i
        level = [-1] * nl
        import sys
        sys.setrecursionlimit(100000)

        def lev(i):
            if level[i] >= 0:
                return level[i]
            if level[i] == -2:
                raise RuntimeError('combinational loop')
            level[i] = -2
            l = 0
            for a in self.luts[i][0]:
                if a in drv:
                    l = max(l, lev(drv[a]) + 1)
            level[i] = l
            return l
        for i in range(nl):
            lev(i)
        self.lut_level = np.array(level)
        self.levels = []
        for L in range(max(level, default=-1) + 1):
            idx = np.where(self.lut_level == L)[0]
            ins = np.zeros((len(idx), 4), dtype=np.int64)
            tab = np.zeros((len(idx), 16), dtype=np.uint64)
            outs = np.zeros(len(idx), dtype=np.int64)
            for r, i in enumerate(idx):
                a, init, y, _, _ = self.luts[i]
                w = len(a)
                aa = a + [0] * (4 - w)
                ins[r] = aa
                for m in range(16):
                    # inputs beyond WIDTH are tied to 0, so entry m maps to m & (2^w-1)
                    bit = (init >> (m & ((1 << w) - 1))) & 1
                    tab[r, m] = np.uint64(0xFFFFFFFFFFFFFFFF) if bit else np.uint64(0)
                outs[r] = y
            self.levels.append((idx, ins, tab, outs))
        self.ff_d = np.array([f[0] for f in self.ffs], dtype=np.int64)
        self.ff_q = np.array([f[1] for f in self.ffs], dtype=np.int64)
        self.ff_init = np.array([f[2] for f in self.ffs], dtype=np.uint64)

    def simulate(self, inputs_packed, n_cycles, ff_flip=None, lut_flip=None, t_inj=None,
                 probes=None):
        """Bit-parallel simulation over lanes packed in uint64 words.

        inputs_packed: dict net -> uint64 array [W] (constant over time)
        ff_flip: uint64 [n_ff, W] lanes in which each FF is upset at its lane's injection cycle
        lut_flip: uint64 [n_lut, W] lanes in which each LUT output is inverted during that cycle
        t_inj: int array [n_lanes] injection cycle per lane
        probes: list of nets; returns dict cycle -> values only at the final cycle
        Returns the final net-value matrix and per-cycle probe history."""
        W = next(iter(inputs_packed.values())).shape[0]
        V = np.zeros((self.n, W), dtype=np.uint64)
        V[1] = np.uint64(0xFFFFFFFFFFFFFFFF)
        for nnet, val in inputs_packed.items():
            V[nnet] = val
        ffs = np.repeat((self.ff_init * np.uint64(0xFFFFFFFFFFFFFFFF))[:, None], W, axis=1)
        hist = []
        tmask = None
        if t_inj is not None:
            lanes = np.arange(W * 64)
        for cyc in range(n_cycles):
            if t_inj is not None:
                sel = (t_inj == cyc)
                tmask = lanes_to_words(sel[None, :])[0] if sel.any() else None
                if tmask is not None and ff_flip is not None:
                    ffs ^= ff_flip & tmask
            V[self.ff_q] = ffs
            for L, (idx, ins, tab, outs) in enumerate(self.levels):
                a = V[ins]                     # [n,4,W]
                x0, x1, x2, x3 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
                t = tab[:, :, None]
                m = [(t[:, 2 * j + 1] & x0) | (t[:, 2 * j] & ~x0) for j in range(8)]
                m = [(m[2 * j + 1] & x1) | (m[2 * j] & ~x1) for j in range(4)]
                m = [(m[2 * j + 1] & x2) | (m[2 * j] & ~x2) for j in range(2)]
                y = (m[1] & x3) | (m[0] & ~x3)
                if tmask is not None and lut_flip is not None:
                    y ^= lut_flip[idx] & tmask
                V[outs] = y
            if probes is not None:
                hist.append(V[probes].copy())
            ffs = V[self.ff_d].copy()
        V[self.ff_q] = ffs
        return V, hist


def lanes_to_words(bits_bool):
    """bool array [..., L] -> uint64 [..., L/64] with lane l at bit (l % 64) of word l//64."""
    L = bits_bool.shape[-1]
    b = bits_bool.reshape(bits_bool.shape[:-1] + (L // 64, 64))
    w = np.zeros(b.shape[:-1], dtype=np.uint64)
    for i in range(64):
        w |= b[..., i].astype(np.uint64) << np.uint64(i)
    return w


def words_to_lanes(words):
    W = words.shape[-1]
    out = np.zeros(words.shape[:-1] + (W, 64), dtype=bool)
    for i in range(64):
        out[..., i] = ((words >> np.uint64(i)) & np.uint64(1)).astype(bool)
    return out.reshape(words.shape[:-1] + (W * 64,))
