"""Packing, quadratic placement with floorplan regions, and recursive-bisection legalisation."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

CAP = 8          # logic cells per tile
UTIL = 0.70      # target utilisation inside a placement region
FLOORPLANS = ['none', 'split', 'g1', 'g2']


def pack_cells(nl):
    """Return logic cells: list of dict(lut=idx or None, ff=idx or None, owner)."""
    fanout = {}
    for i, (ins, _, _, _, _) in enumerate(nl.luts):
        for a in ins:
            fanout[a] = fanout.get(a, 0) + 1
    for d, q, _, _, _ in nl.ffs:
        fanout[d] = fanout.get(d, 0) + 1
    drv = {l[2]: i for i, l in enumerate(nl.luts)}
    cells, lut_cell, ff_cell = [], {}, {}
    for j, (d, q, _, owner, _) in enumerate(nl.ffs):
        i = drv.get(d)
        if i is not None and i not in lut_cell and fanout.get(d, 0) == 1 and nl.luts[i][3] == owner:
            lut_cell[i] = len(cells)
            ff_cell[j] = len(cells)
            cells.append(dict(lut=i, ff=j, owner=owner))
        else:
            ff_cell[j] = len(cells)
            cells.append(dict(lut=None, ff=j, owner=owner))
    for i, l in enumerate(nl.luts):
        if i not in lut_cell:
            lut_cell[i] = len(cells)
            cells.append(dict(lut=i, ff=None, owner=l[3]))
    return cells, lut_cell, ff_cell


def nets_of(nl, lut_cell, ff_cell):
    """Hyperedges over logic cells (driver + sinks), excluding constants and internal LUT->FF links."""
    pins = {}
    for i, (ins, _, y, _, _) in enumerate(nl.luts):
        pins.setdefault(y, set()).add(lut_cell[i])
        for a in ins:
            if a > 1:
                pins.setdefault(a, set()).add(lut_cell[i])
    for j, (d, q, _, _, _) in enumerate(nl.ffs):
        pins.setdefault(q, set()).add(ff_cell[j])
        if d > 1:
            pins.setdefault(d, set()).add(ff_cell[j])
    return {n: sorted(s) for n, s in pins.items() if len(s) > 1}, pins


def _quadratic(n, nets, anchors, seed):
    """Solve min sum w (xi-xj)^2 with clique (<=8 pins) or star models and anchor springs."""
    rows, cols, vals = [], [], []
    nstar = 0
    extra = []
    for net in nets:
        p = len(net)
        if p <= 8:
            w = 2.0 / p
            for a in range(p):
                for b in range(a + 1, p):
                    extra.append((net[a], net[b], w))
        else:
            s = n + nstar
            nstar += 1
            for a in net:
                extra.append((a, s, 1.0))
    N = n + nstar
    deg = np.zeros(N)
    for a, b, w in extra:
        rows += [a, b]
        cols += [b, a]
        vals += [-w, -w]
        deg[a] += w
        deg[b] += w
    bx, by = np.zeros(N), np.zeros(N)
    for c, (x, y, w) in anchors.items():
        deg[c] += w
        bx[c] += w * x
        by[c] += w * y
    deg += 1e-3                                   # weak pull to origin offset below
    rows += list(range(N))
    cols += list(range(N))
    vals += list(deg)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    rng = np.random.default_rng(seed)
    x = spl.spsolve(A.tocsc(), bx)
    y = spl.spsolve(A.tocsc(), by)
    x += rng.normal(0, 1e-3, N)
    y += rng.normal(0, 1e-3, N)
    return x[:n], y[:n]


def _bisect(ids, x, y, rect, assign):
    x0, y0, x1, y1 = rect            # tile index ranges [x0,x1) x [y0,y1)
    w, h = x1 - x0, y1 - y0
    if len(ids) == 0:
        return
    if w * h == 1:
        assign[ids] = np.array([x0, y0])
        return
    if w >= h:
        m = x0 + w // 2
        r1, r2 = (x0, y0, m, y1), (m, y0, x1, y1)
        key = x[ids]
    else:
        m = y0 + h // 2
        r1, r2 = (x0, y0, x1, m), (x0, m, x1, y1)
        key = y[ids]
    c1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
    c2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
    order = ids[np.argsort(key, kind='stable')]
    k = int(round(len(ids) * c1 / (c1 + c2)))
    k = min(max(k, len(ids) - c2 * CAP), c1 * CAP)
    _bisect(order[:k], x, y, r1, assign)
    _bisect(order[k:], x, y, r2, assign)


def place(nl, floorplan='none', seed=0):
    cells, lut_cell, ff_cell = pack_cells(nl)
    nets, pins = nets_of(nl, lut_cell, ff_cell)
    n = len(cells)
    owner = np.array([c['owner'] for c in cells])
    nA, nB, nW = (owner == 'A').sum(), (owner == 'B').sum(), (owner == 'W').sum()
    tiles = int(np.ceil(n / (CAP * UTIL)))
    H = max(4, int(np.ceil(np.sqrt(tiles))))
    if floorplan == 'none':
        Wd = int(np.ceil(tiles / H))
        g = 0
    else:
        g = {'split': 0, 'g1': 1, 'g2': 2}[floorplan]
        if g > 0:
            g = max(g, int(np.ceil(nW / (CAP * UTIL * H))))
        wA = int(np.ceil(nA / (CAP * UTIL * H)))
        wB = int(np.ceil(nB / (CAP * UTIL * H)))
        Wd = wA + g + wB
        if g == 0:
            wA = max(wA, int(np.ceil((nA + nW) / (CAP * UTIL * H))))
            Wd = wA + wB
    # IO anchors: shared inputs on the left edge, wrapper outputs on the right edge
    din = nl.ports[('A', 'din')]
    anchors = {}
    for i, net in enumerate(din):
        for c in pins.get(net, []):
            yv = (i + 0.5) / len(din) * H
            anchors[c] = (-1.0, yv, 0.5)
    for i, net in enumerate(nl.ports[('W', 'q')] + nl.ports[('W', 'alarm')]):
        for c in pins.get(net, []):
            anchors[c] = (Wd, (i + 0.5) / (len(nl.ports[('W', 'q')]) + 1) * H, 0.5)
    x, y = _quadratic(n, list(nets.values()), anchors, seed)
    assign = np.zeros((n, 2), dtype=np.int64)
    if floorplan == 'none':
        _bisect(np.arange(n), x, y, (0, 0, Wd, H), assign)
    else:
        if g == 0:
            _bisect(np.where(owner != 'B')[0], x, y, (0, 0, wA, H), assign)
            _bisect(np.where(owner == 'B')[0], x, y, (wA, 0, Wd, H), assign)
        else:
            _bisect(np.where(owner == 'A')[0], x, y, (0, 0, wA, H), assign)
            _bisect(np.where(owner == 'B')[0], x, y, (wA + g, 0, Wd, H), assign)
            _bisect(np.where(owner == 'W')[0], x, y, (wA, 0, wA + g, H), assign)
    # slot offsets inside tiles
    pos = assign.astype(float)
    key = assign[:, 0] * 10000 + assign[:, 1]
    order = np.argsort(key, kind='stable')
    slot = np.zeros(n, dtype=int)
    last, k = None, 0
    for i in order:
        if key[i] != last:
            last, k = key[i], 0
        slot[i] = k
        k += 1
    if slot.max() >= CAP:
        raise RuntimeError('tile overflow')
    pos[:, 0] += (slot % 2) * 0.5 - 0.25
    pos[:, 1] += (slot // 2) * 0.25 - 0.375
    hp = 0.0
    for net in nets.values():
        p = pos[net]
        hp += np.ptp(p[:, 0]) + np.ptp(p[:, 1])
    offB = {'none': Wd / 2.0, 'split': float(Wd - wB) if floorplan != 'none' else 0.0}.get(floorplan, float(Wd - wB) if floorplan != 'none' else 0.0)
    return dict(offB=offB, cells=cells, lut_cell=lut_cell, ff_cell=ff_cell, pos=pos, owner=owner,
                W=Wd, H=H, area=Wd * H, hpwl=hp, nets=nets, floorplan=floorplan)
