"""Fault-injection campaigns: spatially correlated spot model and placement-blind double bit flips."""
import numpy as np
from netlist import lanes_to_words

OUTCOMES = ['S', 'D', 'C', 'H']   # silent (masked), detected, critical, hang


def twin_offset(pl):
    """Horizontal offset between the origins of the two replica regions as seen by an attacker."""
    return pl.get('offB', pl['W'] / 2.0)


def spot_masks(design, pl, n_lanes, r, m, rng):
    """Lanes x cells hit matrix for spots of radius r (tile units).
    m = 1: one spot; m = 2: twin spot, the second translated by the replica-region offset."""
    pos = pl['pos']
    cx = rng.uniform(-0.5, pl['W'] - 0.5, size=(1, n_lanes))
    cy = rng.uniform(-0.5, pl['H'] - 0.5, size=(1, n_lanes))
    if m == 3:   # two independent random spots (attacker without layout knowledge)
        cx = rng.uniform(-0.5, pl['W'] - 0.5, size=(2, n_lanes))
        cy = rng.uniform(-0.5, pl['H'] - 0.5, size=(2, n_lanes))
    elif m == 2:
        off = twin_offset(pl)
        cx0 = rng.uniform(-0.5, pl['W'] - 0.5 - off, size=n_lanes)
        cx = np.stack([cx0, cx0 + off])
        cy = np.repeat(cy, 2, 0)
    hit = np.zeros((len(pos), n_lanes), dtype=bool)
    for s in range(cx.shape[0]):
        d2 = (pos[:, 0:1] - cx[s][None]) ** 2 + (pos[:, 1:2] - cy[s][None]) ** 2
        hit |= d2 <= r * r
    return hit


def cell_to_elem(design, pl, hit):
    nl = design.nl
    ffh = np.zeros((len(nl.ffs), hit.shape[1]), dtype=bool)
    luth = np.zeros((len(nl.luts), hit.shape[1]), dtype=bool)
    for ci, c in enumerate(pl['cells']):
        if c['ff'] is not None:
            ffh[c['ff']] = hit[ci]
        if c['lut'] is not None:
            luth[c['lut']] = hit[ci]
    return ffh, luth


def classify(design, res, gold):
    q, alarm, valid, oa, ob = res
    wrongA = (oa != gold).any(0)
    wrongB = (ob != gold).any(0)
    same = (oa == ob).all(0)
    hang = ~valid
    det = valid & alarm
    crit = valid & ~alarm & (q != gold).any(0)
    sil = valid & ~alarm & ~crit
    out = np.stack([sil, det, crit, hang], 0)
    cm = wrongA & wrongB & same
    anyw = wrongA | wrongB
    return out, cm, anyw


def campaign(design, pl, r, m, n_lanes=2048, seed=0, inputs=None, model='spot'):
    rng = np.random.default_rng(seed)
    if inputs is None:
        inputs = design.make_inputs(n_lanes, seed + 7)
    inp, gold = inputs
    t_inj = rng.integers(1, design.lat, size=n_lanes)
    if model == 'spot':
        hit = spot_masks(design, pl, n_lanes, r, m, rng)
        ffh, luth = cell_to_elem(design, pl, hit)
    else:   # placement-blind double bit flip: one FF in each replica, same cycle
        nl = design.nl
        own = np.array([f[3] for f in nl.ffs])
        ia, ib = np.where(own == 'A')[0], np.where(own == 'B')[0]
        ffh = np.zeros((len(nl.ffs), n_lanes), dtype=bool)
        ffh[rng.choice(ia, n_lanes), np.arange(n_lanes)] = True
        ffh[rng.choice(ib, n_lanes), np.arange(n_lanes)] = True
        luth = np.zeros((len(nl.luts), n_lanes), dtype=bool)
    res = design.run(inp, lanes_to_words(ffh), lanes_to_words(luth), t_inj)
    out, cm, anyw = classify(design, res, gold)
    counts = out.sum(1)
    return dict(counts=counts, n=n_lanes, cm=int(cm.sum()), anyw=int(anyw.sum()),
                rates=counts / n_lanes, cmi=cm.sum() / max(anyw.sum(), 1))
