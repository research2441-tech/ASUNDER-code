import sys, json, os, glob
import numpy as np, torch
from scipy.stats import spearmanr
from train import load, FOLDS, KERNEL_ORDER, THREATS, to_torch, make_model, truth
from model import collate
from exp_pred import SPECS
from gbm import features, logit
from sklearn.ensemble import HistGradientBoostingRegressor
torch.set_num_threads(1)
NEW = [(0.9, 1), (0.9, 2), (2.6, 1), (2.6, 2), (1.2, 3)]

def spots(g, t, G, rng):
    W, H, off = g['W'], g['H'], g['offB']; r, m = t
    s = int(np.sqrt(G))
    if m == 2:
        span = W - off
    else:
        span = W
    u = (np.arange(s)[:, None] + rng.uniform(0, 1, (s, s))) / s
    v = (np.arange(s)[None, :] + rng.uniform(0, 1, (s, s))) / s
    cx = -0.5 + u.ravel() * span; cy = -0.5 + v.ravel() * H
    c1 = np.stack([cx, cy], 1)
    if m == 1:
        c2 = c1
    elif m == 2:
        c2 = c1 + np.array([off, 0])
    else:
        c2 = np.stack([rng.uniform(-0.5, W - 0.5, len(cx)), rng.uniform(-0.5, H - 0.5, len(cx))], 1)
    return torch.tensor(np.stack([c1, c2], 1), dtype=torch.float32)

@torch.no_grad()
def asunder_pred(fold, seed, gs):
    m = make_model(SPECS['asunder']); m.load_state_dict(torch.load(f'models/asunder_f{fold}_s{seed}.pt')); m.eval()
    rng = np.random.default_rng(5)
    P = np.zeros((len(gs), len(NEW)))
    for b0 in range(0, len(gs), 16):
        gl = [to_torch(g) for g in gs[b0:b0 + 16]]; B = collate(gl); H = m.encode(B)
        for j, g in enumerate(gl):
            h = H[B['ptr'][j]:B['ptr'][j + 1]]
            thr = [(r, min(mm, 2)) for r, mm in NEW]
            p, _, _, _ = m.readout(g, h, thr, [spots(g, t, 144, rng) for t in NEW])
            P[b0 + j] = p[:, 2].numpy()
    return P

def gbm_pred(fold, gs, mode):
    rng = np.random.default_rng(1000 + fold); tr = []
    for k in KERNEL_ORDER:
        if k in FOLDS[fold]: continue
        g2 = load([k]); idx = rng.permutation(len(g2))[:180]; tr += [g2[i] for i in idx]
    R, _ = truth(tr)
    X = np.array([features(g, t) for g in tr for t in THREATS])
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X, logit(R[:, :, 2].reshape(-1)))
    P = np.zeros((len(gs), len(NEW)))
    for j, (r, mm) in enumerate(NEW):
        mf = mm if mm < 3 else mode
        Xp = np.array([features(g, (r, mf)) for g in gs])
        P[:, j] = 1 / (1 + np.exp(-mdl.predict(Xp)))
    return P

if __name__ == '__main__':
    res = {}
    for fold, held in enumerate(FOLDS):
        for kn in held:
            lab = json.load(open(f'data/transfer_{kn}.json'))
            gs = [g for g in load([kn]) if str(g['meta']['idx']) in lab]
            Y = np.array([[lab[str(g['meta']['idx'])][f'{r}_{m}']['counts'][2] / lab[str(g['meta']['idx'])][f'{r}_{m}']['n'] for r, m in NEW] for g in gs])
            preds = {f'asunder_s{s}': asunder_pred(fold, s, gs) for s in range(3) if os.path.exists(f'models/asunder_f{fold}_s{s}.pt')}
            preds['gbm_m1'] = gbm_pred(fold, gs, 1); preds['gbm_m2'] = gbm_pred(fold, gs, 2)
            res[kn] = {name: {f'{r}_{m}': dict(rho=float(spearmanr(P[:, j], Y[:, j]).correlation), mae=float(100 * np.abs(P[:, j] - Y[:, j]).mean()))
                              for j, (r, m) in enumerate(NEW)} for name, P in preds.items()}
            print(kn, {n: {t: round(v['rho'], 2) for t, v in d.items()} for n, d in res[kn].items()}, flush=True)
    json.dump(res, open('analysis/transfer.json', 'w'), indent=1)
