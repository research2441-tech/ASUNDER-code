"""E5: budgeted search for threat-robust DMR designs (directives of both replicas + floorplan)."""
import os, sys, json, time, random, pickle
import numpy as np
import torch
from kernels import KERNELS
from hls import all_configs, cfg_key, DIRECTIVE_SPACE
from dmr import Design
from place import place, FLOORPLANS
from fi import campaign
from graph import build_graph
from train import FOLDS, KERNEL_ORDER, load, truth, to_torch, make_model, THREATS
from model import stratified_spots, collate
from exp_pred import SPECS

REF = dict(u=1, alloc=1, share='le', fsm='bin', sched='asap')
OBJ_T = THREATS
AREA_CAP, LAT_CAP = 1.30, 1.50
B_FI = int(os.environ.get('ASUNDER_BUDGET', 16))
POOL = int(os.environ.get('ASUNDER_POOL', 300))
EPS = 1e-4


def fold_of(kn):
    return [i for i, f in enumerate(FOLDS) if kn in f][0]


def build(kn, a, b, fp):
    d = Design(kn, a, b)
    pl = place(d.nl, fp, 0)
    return d, pl


def ref_costs(kn):
    d, pl = build(kn, REF, REF, 'none')
    return pl['area'], d.lat


def feasible(d, pl, ref):
    return pl['area'] <= AREA_CAP * ref[0] and d.lat <= LAT_CAP * ref[1]


def make_pool(kn, seed=0):
    fn = f'search/pool_{kn}.pkl'
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    cf = all_configs(KERNELS[kn])
    ref = ref_costs(kn)
    rng = random.Random(seed + sum(map(ord, kn)))
    pool, seen = [], set()
    tries = 0
    while len(pool) < POOL and tries < 20 * POOL:
        tries += 1
        a = rng.choice(cf)
        b = a if rng.random() < 0.15 else rng.choice(cf)
        fp = rng.choice(FLOORPLANS)
        key = (cfg_key(a), cfg_key(b), fp)
        if key in seen:
            continue
        seen.add(key)
        d, pl = build(kn, a, b, fp)
        if not feasible(d, pl, ref):
            continue
        g = build_graph(d, pl)
        g['meta'] = dict(kernel=kn, cfgA=a, cfgB=b, fp=fp, area=pl['area'], lat=d.lat, n_lut=len(d.nl.luts),
                         n_ff=len(d.nl.ffs), latA=d.rA.latency(), latB=d.rB.latency(), W=pl['W'], H=pl['H'],
                         hpwl=pl['hpwl'])
        pool.append(g)
    os.makedirs('search', exist_ok=True)
    pickle.dump((pool, ref), open(fn, 'wb'))
    return pool, ref


def true_J(kn, meta, n_lanes=4096, seed=0):
    d, pl = build(kn, meta['cfgA'], meta['cfgB'], meta['fp'])
    seed = int(seed)
    inp = d.make_inputs(n_lanes, 50000 + seed)
    c = [campaign(d, pl, r, m, n_lanes, 70000 + seed * 7 + i, inp)['rates'][2] for i, (r, m) in enumerate(OBJ_T)]
    return max(c), c


# ------------------------------------------------------------------ surrogates
@torch.no_grad()
def ensemble_scores(kn, graphs, seeds=(0, 1, 2)):
    f = fold_of(kn)
    preds = []
    for s in seeds:
        m = make_model(SPECS['asunder'])
        m.load_state_dict(torch.load(f'models/asunder_f{f}_s{s}.pt'))
        m.eval()
        rng = np.random.default_rng(11)
        out = np.zeros((len(graphs), len(OBJ_T)))
        for b0 in range(0, len(graphs), 16):
            gl = [to_torch(g) for g in graphs[b0:b0 + 16]]
            Bt = collate(gl)
            H = m.encode(Bt)
            for j, g in enumerate(gl):
                h = H[Bt['ptr'][j]:Bt['ptr'][j + 1]]
                P, _, _, _ = m.readout(g, h, OBJ_T, stratified_spots(g, OBJ_T, 100, rng))
                out[b0 + j] = P[:, 2].numpy()
        preds.append(out.max(1))
    return np.array(preds)            # [S, n]


def gbm_scores(kn, graphs):
    from gbm import features, logit
    from sklearn.ensemble import HistGradientBoostingRegressor
    f = fold_of(kn)
    tr = []
    rng = np.random.default_rng(1000 + f)
    for k in KERNEL_ORDER:
        if k in FOLDS[f]:
            continue
        gs = load([k])
        idx = rng.permutation(len(gs))[:int(os.environ.get('ASUNDER_NTRAIN', 180))]
        tr += [gs[i] for i in idx]
    R, _ = truth(tr, OBJ_T)
    X = np.array([features(g, t) for g in tr for t in OBJ_T])
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X, logit(R[:, :, 2].reshape(-1)))
    Xp = np.array([features(g, t) for g in graphs for t in OBJ_T])
    p = 1 / (1 + np.exp(-mdl.predict(Xp)))
    return p.reshape(len(graphs), len(OBJ_T)).max(1)


# ---------------------------------------------------------------------- encodings
def encode(meta):
    v = []
    for c in (meta['cfgA'], meta['cfgB']):
        for f, opts in DIRECTIVE_SPACE.items():
            v += [float(c[f] == o) for o in opts]
    v += [float(meta['fp'] == x) for x in FLOORPLANS]
    return np.array(v)


# -------------------------------------------------------------------------- methods
def m_random(kn, pool, rng, Jcache):
    idx = rng.choice(len(pool), B_FI, replace=False)
    return list(idx)


def m_diversity(kn, pool, rng, Jcache):
    """Design-diversity rule of thumb: maximum directive Hamming distance, widest isolation."""
    order = {'g2': 3, 'g1': 2, 'split': 1, 'none': 0}
    sc = []
    for i, g in enumerate(pool):
        a, b = g['meta']['cfgA'], g['meta']['cfgB']
        ham = sum(a[f] != b[f] for f in DIRECTIVE_SPACE)
        sc.append((ham, order[g['meta']['fp']], rng.random(), i))
    sc.sort(reverse=True)
    return [sc[0][3]]


def m_screen(scores):
    def f(kn, pool, rng, Jcache):
        return list(np.argsort(scores)[:B_FI])
    return f


def m_gpbo(kn, pool, rng, Jcache, n_init=4):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import norm
    X = np.array([encode(g['meta']) for g in pool])
    chosen = list(rng.choice(len(pool), n_init, replace=False))
    while len(chosen) < B_FI:
        y = np.log(np.array([Jcache(i) for i in chosen]) + EPS)
        gp = GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=2.0, nu=2.5) + WhiteKernel(0.1),
                                      normalize_y=True, random_state=0).fit(X[chosen], y)
        mu, sd = gp.predict(X, return_std=True)
        best = y.min()
        z = (best - mu) / (sd + 1e-9)
        ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        ei[chosen] = -1
        chosen.append(int(np.argmax(ei)))
    return chosen


def m_reinforce(kn, pool, rng, Jcache, batch=4, lr=0.5):
    """Factorised categorical policy over the 11 design factors, restricted to the feasible pool:
    each sample is the pool member with the highest policy log-probability among 32 random draws."""
    X = np.array([encode(g['meta']) for g in pool])
    logits = np.zeros(X.shape[1])
    groups = []
    k = 0
    for rep in range(2):
        for f, opts in DIRECTIVE_SPACE.items():
            groups.append(list(range(k, k + len(opts))))
            k += len(opts)
    groups.append(list(range(k, k + len(FLOORPLANS))))
    chosen, base = [], None
    while len(chosen) < B_FI:
        lp = np.zeros(len(pool))
        for gidx in groups:
            l = logits[gidx] - np.log(np.exp(logits[gidx]).sum())
            lp += X[:, gidx] @ l
        batch_idx = []
        for _ in range(batch):
            cand = rng.choice(len(pool), min(32, len(pool)), replace=False)
            p = np.exp(lp[cand] - lp[cand].max())
            p /= p.sum()
            i = int(rng.choice(cand, p=p))
            if i not in chosen and i not in batch_idx:
                batch_idx.append(i)
        batch_idx = batch_idx[:B_FI - len(chosen)]
        if not batch_idx:
            continue
        rew = -np.log(np.array([Jcache(i) for i in batch_idx]) + EPS)
        base = rew.mean() if base is None else 0.7 * base + 0.3 * rew.mean()
        for i, rw in zip(batch_idx, rew):
            for gidx in groups:
                pr = np.exp(logits[gidx]) / np.exp(logits[gidx]).sum()
                logits[gidx] += lr * (rw - base) * (X[i, gidx] - pr)
        chosen += batch_idx
    return chosen


def m_asunder(ens, kappa=1.0, lam=2.0):
    """Ensemble-guided acquisition with few-shot log-linear recalibration on the target kernel."""
    lmu = np.log(ens + EPS).mean(0)
    lsd = np.log(ens + EPS).std(0)

    def f(kn, pool, rng, Jcache):
        chosen = [int(np.argmin(lmu))]
        while len(chosen) < B_FI:
            x = lmu[chosen]
            y = np.log(np.array([Jcache(i) for i in chosen]) + EPS)
            # ridge fit of y = a + b x shrunk toward (0, 1)
            A = np.stack([np.ones_like(x), x], 1)
            P0 = np.array([0.0, 1.0])
            reg = lam * np.eye(2)
            th = np.linalg.solve(A.T @ A + reg, A.T @ y + reg @ P0)
            res = y - A @ th
            s = np.sqrt((res ** 2).sum() / max(len(y) - 1, 1)) if len(y) > 2 else 0.5
            pred = th[0] + th[1] * lmu
            lcb = pred - kappa * (abs(th[1]) * lsd + s)
            lcb[chosen] = np.inf
            chosen.append(int(np.argmin(lcb)))
        return chosen
    return f


def run_kernel(kn, seeds=None):
    seeds = seeds or range(int(os.environ.get('ASUNDER_SEARCH_SEEDS', 5)))
    os.makedirs('search', exist_ok=True)
    out = f'search/res_{kn}.json'
    if os.path.exists(out):
        return
    t0 = time.time()
    pool, ref = make_pool(kn)
    tpool = time.time() - t0
    t0 = time.time()
    ens = ensemble_scores(kn, pool)
    tens = (time.time() - t0) / (len(pool) * ens.shape[0])
    gb = gbm_scores(kn, pool)
    ref_meta = dict(cfgA=REF, cfgB=REF, fp='none')
    J_ref, _ = true_J(kn, ref_meta, 16384, 999)
    cache = {}

    def Jc(i):
        if i not in cache:
            cache[int(i)] = true_J(kn, pool[i]["meta"], 4096, int(i))[0]
            return cache[int(i)]
        return cache[i]
    t0 = time.time()
    for i in range(len(pool)):
        Jc(i)
    t_fi = (time.time() - t0) / len(pool)
    res_tfi = t_fi
    methods = {'random': m_random, 'diversity': m_diversity, 'gp_bo': m_gpbo, 'reinforce': m_reinforce,
               'gbm_screen': m_screen(gb), 'asunder_screen': m_screen(ens.mean(0)), 'asunder': m_asunder(ens)}
    res = dict(t_fi=res_tfi, kernel=kn, ref_area=ref[0], ref_lat=ref[1], J_ref=J_ref, t_pool=tpool, t_surrogate=tens,
               pool_size=len(pool), runs=[])
    final_cache = {}
    for s in seeds:
        for name, meth in methods.items():
            rng = np.random.default_rng(100 * s + sum(map(ord, name)))
            chosen = meth(kn, pool, rng, Jc)
            obs = [Jc(i) for i in chosen]
            best = chosen[int(np.argmin(obs))]
            curve = list(np.minimum.accumulate(obs))
            if best not in final_cache:
                final_cache[best] = true_J(kn, pool[best]['meta'], 16384, 777)
            Jf, comps = final_cache[best]
            me = pool[best]['meta']
            res['runs'].append(dict(seed=s, method=name, chosen=[int(c) for c in chosen], obs=obs, curve=curve,
                                    best=int(best), J_final=Jf, C_single=comps[0], C_twin=comps[1],
                                    area=me['area'], lat=me['lat'], fp=me['fp'], A=cfg_key(me['cfgA']),
                                    B=cfg_key(me['cfgB'])))
        print(kn, 'seed', s, {r['method']: round(r['J_final'], 5) for r in res['runs'] if r['seed'] == s}, flush=True)
    # pool oracle (every pool member was evaluated once with 4096 faults)
    res['oracle_seen'] = min(cache.values())
    res['n_evaluated'] = len(cache)
    res['ens_mu'] = ens.mean(0).tolist()
    res['ens_sd'] = ens.std(0).tolist()
    res['gbm'] = gb.tolist()
    res['J_cache'] = {int(k): v for k, v in cache.items()}
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    for kn in sys.argv[1].split(','):
        run_kernel(kn)
