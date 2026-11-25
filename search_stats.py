import json, glob, collections, numpy as np
from scipy.stats import wilcoxon
METHODS = ['random', 'diversity', 'gp_bo', 'reinforce', 'gbm_screen', 'asunder_screen', 'asunder', 'asunder_gp']
def cliff(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] > y[None]).sum() - (x[:, None] < y[None]).sum()) / (len(x) * len(y))
def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj
res = {k.split('res_')[1][:-5]: json.load(open(k)) for k in glob.glob('search/res_*.json')}
J = collections.defaultdict(dict); REG = collections.defaultdict(dict); PCT = collections.defaultdict(dict); CURVE = collections.defaultdict(list)
OK = collections.defaultdict(dict); AREA = collections.defaultdict(dict); LAT = collections.defaultdict(dict)
for kn, r in res.items():
    pool_J = np.array(list(r['J_cache'].values()))
    Jmin, Jref = r['oracle_seen'], r['J_ref']
    for run in r['runs']:
        key = (kn, run['seed'])
        J[run['method']][key] = run['J_final']
        REG[run['method']][key] = (run['J_final'] - Jmin) / max(Jref - Jmin, 1e-9)
        PCT[run['method']][key] = float((pool_J < run['obs'][int(np.argmin(run['obs']))]).mean())
        c = list(run['curve']) + [run['curve'][-1]] * (16 - len(run['curve']))
        CURVE[run['method']].append([(x - Jmin) / max(Jref - Jmin, 1e-9) for x in c[:16]])
        AREA[run['method']][key] = run['area'] / r['ref_area']; LAT[run['method']][key] = run['lat'] / r['ref_lat']
out = {}
for m in METHODS:
    if m not in J: continue
    v = np.array(list(J[m].values())) * 100
    out[m] = dict(J_mean=float(v.mean()), J_median=float(np.median(v)), J_sd=float(v.std()), regret_mean=float(np.mean(list(REG[m].values()))),
                  regret_median=float(np.median(list(REG[m].values()))), pct_mean=float(np.mean(list(PCT[m].values()))),
                  area=float(np.mean(list(AREA[m].values()))), lat=float(np.mean(list(LAT[m].values()))),
                  curve=np.mean(CURVE[m], 0).tolist(), n=len(v))
kernels = sorted({k for k, _ in J['asunder']})
km = {m: np.array([np.mean([J[m][(kn, sd)] for sd in range(5) if (kn, sd) in J[m]]) for kn in kernels]) for m in METHODS if m in J}
tests = []
for m in METHODS:
    if m == 'asunder' or m not in km: continue
    a, b = km['asunder'], km[m]
    diff = a - b
    p = wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
    tests.append((m, p, float(cliff(b, a)), int((diff < 0).sum()), int((diff > 0).sum()), int((diff == 0).sum())))
adj = holm([t[1] for t in tests])
out['_tests'] = {t[0]: dict(p=float(t[1]), p_holm=float(pa), cliff_other_vs_asunder=t[2], asunder_better=t[3], asunder_worse=t[4], ties=t[5]) for t, pa in zip(tests, adj)}
out['_per_kernel'] = {kn: dict(J_ref=100 * r['J_ref'], J_min=100 * r['oracle_seen'], pool=r['pool_size'], t_fi=r['t_fi'], t_surr=r['t_surrogate'],
                               **{m: 100 * float(np.mean([J[m][(kn, s)] for s in range(5) if (kn, s) in J[m]])) for m in METHODS if m in J}) for kn, r in res.items()}
json.dump(out, open('analysis/search_stats.json', 'w'), indent=1)
for m in METHODS:
    if m in out: print(f"{m:15s} J={out[m]['J_mean']:.3f}% med={out[m]['J_median']:.3f} regret={out[m]['regret_mean']:.3f} pct={out[m]['pct_mean']:.3f} area={out[m]['area']:.2f} lat={out[m]['lat']:.2f}")
for k, v in out['_tests'].items(): print(k, v)
for k, v in out['_per_kernel'].items(): print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
