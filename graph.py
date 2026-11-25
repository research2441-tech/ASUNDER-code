"""Cross-replica graph construction for a placed DMR design."""
import re
import numpy as np
from scipy.spatial import cKDTree
from kernels import output_layout

ROLES = ['fsm', 'cnt', 'val', 'car', 'out', 'wrap', 'none']
R_PROX = 2.5     # tile units; covers the largest radius studied
K_PROX = 8
K_CORR = 4
F_DIM = 3 + 2 + 16 + 1 + len(ROLES) + 2 + 2 + 1 + 1


def ff_register(name, k):
    """Map a flip-flop net name to (register, bit, role)."""
    m = re.match(r'(.+)\[(\d+)\]$', name or '')
    if not m:
        return None, None, 'none'
    base, b = m.group(1), int(m.group(2))
    if base in ('state', 'done') or base.startswith('st_'):
        return 'state', b, 'fsm'
    if base == 'cnt':
        return 'cnt', b, 'cnt'
    if re.match(r'R\d+$', base):
        return base, b, 'val'
    if base.startswith('C_'):
        return base, b, 'car'
    if base.startswith('O_'):
        return base, b, 'out'
    if base == 'dout':
        off = 0
        for n, j, w in output_layout(k):
            if off <= b < off + w:
                if j is None:
                    return f"C_{k['outs'][n]}", b - off, 'car'
                return f"O_{n}_{j}", b - off, 'out'
            off += w
    if base in ('q', 'valid', 'alarm'):
        return base, b, 'wrap'
    return None, None, 'none'


def build_graph(design, pl):
    nl, k = design.nl, design.k
    cells = pl['cells']
    n = len(cells)
    own_map = {'A': 0, 'B': 1, 'W': 2}
    owner = np.array([own_map[c['owner']] for c in cells])
    pos = pl['pos'].astype(np.float32)
    X = np.zeros((n, F_DIM), dtype=np.float32)
    X[np.arange(n), owner] = 1
    fan_out = np.zeros(n)
    fan_in = np.zeros(n)
    depth = np.zeros(n)
    maxlev = max(1, int(nl.lut_level.max()) if len(nl.lut_level) else 1)
    lat = float(design.lat)
    tl = {'A': design.rA.timelines(), 'B': design.rB.timelines()}
    live = np.zeros(n)
    role_idx = np.full(n, ROLES.index('none'))
    corr_items = {}
    for ci, c in enumerate(cells):
        if c['lut'] is not None:
            ins, init, y, _, _ = nl.luts[c['lut']]
            X[ci, 3] = 1
            w = len(ins)
            for mm in range(16):
                X[ci, 5 + mm] = (init >> (mm & ((1 << w) - 1))) & 1
            X[ci, 21] = w / 4.0
            depth[ci] = nl.lut_level[c['lut']] / maxlev
        if c['ff'] is not None:
            X[ci, 4] = 1
            d, q, _, o, name = nl.ffs[c['ff']]
            reg, bit, role = ff_register(name, k)
            role_idx[ci] = ROLES.index(role)
            if o in tl and reg in tl[o]:
                for key, w, t0, t1 in tl[o][reg]:
                    if bit < w:
                        live[ci] += (t1 - t0 + 1) / lat
                        corr_items.setdefault((repr(key), bit), []).append((ci, t0, t1, o))
    X[np.arange(n), 22 + role_idx] = 1
    # netlist edges between logic cells
    src, dst = [], []
    for net, pins in pl['nets'].items():
        drv = None
        # the driver is the cell whose LUT output or FF Q is this net
        for p in pins:
            c = cells[p]
            if (c['lut'] is not None and nl.luts[c['lut']][2] == net and c['ff'] is None) or \
               (c['ff'] is not None and nl.ffs[c['ff']][1] == net):
                drv = p
                break
        if drv is None:
            continue
        for p in pins:
            if p != drv:
                src.append(drv)
                dst.append(p)
    src, dst = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)
    np.add.at(fan_out, src, 1)
    np.add.at(fan_in, dst, 1)
    b = 22 + len(ROLES)
    X[:, b] = np.log1p(fan_out) / 4
    X[:, b + 1] = np.log1p(fan_in) / 4
    X[:, b + 2] = pos[:, 0] / pl['W']
    X[:, b + 3] = pos[:, 1] / pl['H']
    X[:, b + 4] = np.minimum(live, 4.0) / 4
    X[:, b + 5] = depth
    # proximity edges across owners
    tree = cKDTree(pos)
    dd, ii = tree.query(pos, k=min(n, 4 * K_PROX + 1), distance_upper_bound=R_PROX)
    ps, pd_, pa = [], [], []
    for i in range(n):
        cnt = 0
        for d_, j in zip(dd[i], ii[i]):
            if j >= n or j == i or owner[j] == owner[i]:
                continue
            ps.append(j); pd_.append(i); pa.append(d_)
            cnt += 1
            if cnt >= K_PROX:
                break
    # correspondence edges (A <-> B) weighted by temporal overlap of identical value bits
    off = pl.get('offB', pl['W'] / 2.0)
    pair_w = {}
    for key, lst in corr_items.items():
        As = [t for t in lst if t[3] == 'A']
        Bs = [t for t in lst if t[3] == 'B']
        for ca, a0, a1, _ in As:
            for cb, b0, b1, _ in Bs:
                ov = min(a1, b1) - max(a0, b0) + 1
                if ov > 0:
                    pair_w[(ca, cb)] = pair_w.get((ca, cb), 0.0) + ov / lat
    cs, cd, cat = [], [], []
    byA = {}
    for (ca, cb), w in pair_w.items():
        byA.setdefault(ca, []).append((w, cb))
    for ca, lst in byA.items():
        lst.sort(reverse=True)
        for w, cb in lst[:K_CORR]:
            dv = np.hypot(*(pos[cb] - pos[ca]))
            dt = np.hypot(pos[cb, 0] - pos[ca, 0] - off, pos[cb, 1] - pos[ca, 1])
            for s_, d__ in ((ca, cb), (cb, ca)):
                cs.append(s_); cd.append(d__); cat.append((w, dv, dt))
    return dict(x=X, pos=pos, owner=owner,
                e_net=np.stack([src, dst]) if len(src) else np.zeros((2, 0), np.int64),
                e_prox=np.array([ps, pd_], dtype=np.int64).reshape(2, -1),
                a_prox=np.array(pa, dtype=np.float32).reshape(-1, 1),
                e_corr=np.array([cs, cd], dtype=np.int64).reshape(2, -1),
                a_corr=np.array(cat, dtype=np.float32).reshape(-1, 3),
                W=float(pl['W']), H=float(pl['H']), offB=float(off), lat=lat)
