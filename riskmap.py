import sys, json
import numpy as np, torch
from dmr import Design
from place import place
from fi import cell_to_elem, classify
from netlist import lanes_to_words
from graph import build_graph
from train import to_torch, make_model
from exp_pred import SPECS
from scipy.stats import spearmanr
torch.set_num_threads(1)
REF = dict(u=1, alloc=1, share='le', fsm='bin', sched='asap')
DIV = dict(u=2, alloc=1, share='ded', fsm='onehot', sched='alap')
def grid_spots(pl, m, nx, ny):
    off = pl['offB']; W, H = pl['W'], pl['H']
    span = (W - off) if m == 2 else W
    xs = -0.5 + (np.arange(nx) + 0.5) / nx * span
    ys = -0.5 + (np.arange(ny) + 0.5) / ny * H
    C = np.array([(x, y) for y in ys for x in xs])
    return C, xs, ys
def empirical(d, pl, C, r, m, lanes=512, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    pos = pl['pos']
    for i, c in enumerate(C):
        inp = d.make_inputs(lanes, 100 + i)
        t = rng.integers(1, d.lat, size=lanes)
        cs = [c] if m == 1 else [c, c + np.array([pl['offB'], 0])]
        hit = np.zeros(len(pos), bool)
        for cc in cs:
            hit |= ((pos - cc) ** 2).sum(1) <= r * r
        H = np.repeat(hit[:, None], lanes, 1)
        ffh, luth = cell_to_elem(d, pl, H)
        res = d.run(inp[0], lanes_to_words(ffh), lanes_to_words(luth), t)
        o, cm, anyw = classify(d, res, inp[1])
        out.append(o[2].mean())
    return np.array(out)
@torch.no_grad()
def predicted(g, C, r, m, fold=3, seeds=(0,)):
    gt = to_torch(g)
    sp = torch.tensor(np.stack([C, C + (np.array([g['offB'], 0]) if m == 2 else 0)], 1), dtype=torch.float32)
    res = []
    for s in seeds:
        mdl = make_model(SPECS['asunder']); mdl.load_state_dict(torch.load(f'models/asunder_f{fold}_s{s}.pt')); mdl.eval()
        h = mdl.encode(gt)
        _, _, _, per = mdl.readout(gt, h, [(r, m)], [sp])
        res.append(per[0][:, 2].numpy())
    return np.mean(res, 0)
if __name__ == '__main__':
    kn = 'speck'; r = 1.2
    out = {}
    for name, a, b, fp in [('ident_g2', REF, REF, 'g2'), ('div_g2', REF, DIV, 'g2'), ('ident_none', REF, REF, 'none')]:
        d = Design(kn, a, b); pl = place(d.nl, fp, 0); g = build_graph(d, pl)
        for m in (1, 2):
            C, xs, ys = grid_spots(pl, m, 12 if m == 1 else 6, 10)
            e = empirical(d, pl, C, r, m)
            p = predicted(g, C, r, m)
            rho = spearmanr(e, p).correlation if e.std() > 0 else float('nan')
            out[f'{name}_m{m}'] = dict(C=C.tolist(), emp=e.tolist(), pred=p.tolist(), xs=xs.tolist(), ys=ys.tolist(),
                                     W=pl['W'], H=pl['H'], off=pl['offB'], rho=rho, emp_mean=float(e.mean()), pred_mean=float(p.mean()))
            print(name, m, 'rho', rho, 'emp mean', e.mean(), 'pred mean', p.mean(), flush=True)
    json.dump(out, open('analysis/riskmap.json', 'w'))
