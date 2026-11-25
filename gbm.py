"""Feature-engineered gradient-boosting baseline with an analytical exposure feature."""
import os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from train import load, truth, FOLDS, KERNEL_ORDER, THREATS
from hls import DIRECTIVE_SPACE
from place import FLOORPLANS


def cfg_vec(c):
    v = []
    for f, opts in DIRECTIVE_SPACE.items():
        v += [float(c[f] == o) for o in opts]
    return v


def exposure(g, r, m):
    """Correspondence-weighted geometric exposure: overlap weight of matched value bits whose
    cells can be covered together by one spot (m=1) or by the translated twin spot (m=2)."""
    a = g['a_corr']
    if a.shape[0] == 0:
        return 0.0, 0.0
    w, dv, dt = a[:, 0], a[:, 1], a[:, 2]
    d = dv if m == 1 else dt
    return float((w * (d <= 2 * r)).sum() / 2), float((w * np.exp(-(d / r) ** 2)).sum() / 2)


def features(g, t, exposure_on=True):
    r, m = t
    me = g['meta']
    a, b = me['cfgA'], me['cfgB']
    ham = sum(a[f] != b[f] for f in DIRECTIVE_SPACE)
    own = g['owner']
    pa = g['a_prox'][:, 0]
    e1, e2 = exposure(g, r, m) if exposure_on else (0.0, 0.0)
    f = cfg_vec(a) + cfg_vec(b) + [float(ham), float(ham == 0)] + [float(me['fp'] == x) for x in FLOORPLANS]
    f += [me['n_lut'], me['n_ff'], me['latA'], me['latB'], me['lat'], me['W'], me['H'], me['area'], me['hpwl'],
          float((pa <= r).sum()) if exposure_on else 0.0, float((pa <= 2 * r).sum()) if exposure_on else 0.0, e1, e2, r, m]
    return f


def logit(p, eps=2e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def run(fold, seed=0, exposure_on=True):
    out = f'results/pred_gbm{"" if exposure_on else "_noexp"}_f{fold}_s{seed}.npz'
    if os.path.exists(out):
        return
    held = FOLDS[fold]
    rng = np.random.default_rng(1000 + fold)
    tr = []
    for k in KERNEL_ORDER:
        if k in held:
            continue
        gs = load([k])
        idx = rng.permutation(len(gs))[:int(os.environ.get('ASUNDER_NTRAIN', 180))]
        tr += [gs[i] for i in idx]
    te = load(list(held))
    Rtr, Ctr = truth(tr)
    Xtr = np.array([features(g, t, exposure_on) for g in tr for t in THREATS])
    Xte = np.array([features(g, t, exposure_on) for g in te for t in THREATS])
    P = np.zeros((len(te), len(THREATS), 4))
    for o in range(4):
        y = logit(Rtr[:, :, o].reshape(-1))
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed).fit(Xtr, y)
        P[:, :, o] = (1 / (1 + np.exp(-mdl.predict(Xte)))).reshape(len(te), len(THREATS))
    P = P / P.sum(-1, keepdims=True)
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed).fit(Xtr, Ctr.reshape(-1))
    cmi = np.clip(mdl.predict(Xte).reshape(len(te), len(THREATS)), 0, 1)
    R, CMI = truth(te)
    ker = np.array([g['meta']['kernel'] for g in te])
    np.savez(out, P=P, CM=cmi, AN=np.ones_like(cmi), R=R, CMI=CMI, ker=ker,
             idx=np.array([g['meta']['idx'] for g in te]), ttrain=0, tinf=0, hist=np.zeros((1, 2)), nparam=0)
    print('gbm', fold, flush=True)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    for f in range(4):
        run(f)
        run(f, exposure_on=False)
