"""ASUNDER-GP: Gaussian-process search whose prior mean is the recalibrated ensemble surrogate.
Uses the already measured pool objectives stored by search.py, so no new pool campaigns are needed;
the returned design is re-measured with the same 16,384-run protocol as every other method."""
import sys, json, pickle, os
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings; warnings.filterwarnings('ignore')
import search
from hls import cfg_key
EPS = search.EPS

def m_asunder_gp(ens, lam=2.0):
    lmu = np.log(ens + EPS).mean(0); lsd = np.log(ens + EPS).std(0)
    def f(kn, pool, rng, Jcache):
        X = np.array([search.encode(g['meta']) for g in pool])
        chosen = [int(np.argmin(lmu))]
        while len(chosen) < search.B_FI:
            x = lmu[chosen]; y = np.log(np.array([Jcache(i) for i in chosen]) + EPS)
            A = np.stack([np.ones_like(x), x], 1); P0 = np.array([0.0, 1.0]); reg = lam * np.eye(2)
            th = np.linalg.solve(A.T @ A + reg, A.T @ y + reg @ P0)
            prior = th[0] + th[1] * lmu
            res = y - prior[chosen]
            gp = GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=2.0, nu=2.5) + WhiteKernel(0.1),
                                          normalize_y=False, random_state=0).fit(X[chosen], res)
            rm, rs = gp.predict(X, return_std=True)
            mu = prior + rm
            sd = np.sqrt(rs ** 2 + (th[1] * lsd) ** 2)
            best = y.min(); z = (best - mu) / (sd + 1e-9)
            ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
            ei[chosen] = -1
            chosen.append(int(np.argmax(ei)))
        return chosen
    return f

def run(kn):
    fn = f'search/res_{kn}.json'
    res = json.load(open(fn))
    if any(r['method'] == 'asunder_gp' for r in res['runs']):
        return
    pool, ref = pickle.load(open(f'search/pool_{kn}.pkl', 'rb'))
    ens = search.ensemble_scores(kn, pool)
    cache = {int(k): v for k, v in res['J_cache'].items()}
    Jc = lambda i: cache[int(i)]
    meth = m_asunder_gp(ens)
    final = {}
    for s in range(int(os.environ.get('ASUNDER_SEARCH_SEEDS', 5))):
        rng = np.random.default_rng(100 * s + 7)
        chosen = meth(kn, pool, rng, Jc)
        obs = [Jc(i) for i in chosen]; best = chosen[int(np.argmin(obs))]
        if best not in final:
            final[best] = search.true_J(kn, pool[best]['meta'], 16384, 777)
        Jf, comps = final[best]; me = pool[best]['meta']
        res['runs'].append(dict(seed=s, method='asunder_gp', chosen=[int(c) for c in chosen], obs=obs,
                                curve=list(np.minimum.accumulate(obs)), best=int(best), J_final=Jf, C_single=comps[0], C_twin=comps[1],
                                area=me['area'], lat=me['lat'], fp=me['fp'], A=cfg_key(me['cfgA']), B=cfg_key(me['cfgB'])))
    json.dump(res, open(fn, 'w'))
    print(kn, 'asunder_gp', [round(100 * r['J_final'], 3) for r in res['runs'] if r['method'] == 'asunder_gp'], flush=True)


if __name__ == '__main__':
    for kn in sys.argv[1].split(','):
        run(kn)
