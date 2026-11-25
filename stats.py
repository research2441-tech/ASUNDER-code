"""Aggregate prediction results: per-model means/SD over held-out kernels, paired Wilcoxon (kernel-seed pairs), Holm, Cliff's delta."""
import sys, glob, json, collections, numpy as np
from evalpred import metrics
from scipy.stats import wilcoxon
KEYS = ['rho_J', 'rho_cer', 'rho_single', 'rho_twin', 'mae_cer', 'top10', 'nll', 'cmi_mae']
def load_all():
    agg = collections.defaultdict(dict)
    for f in sorted(glob.glob('results/pred_*.npz')):
        name = f.split('pred_')[1].rsplit('_f', 1)[0]
        seed = int(f.split('_s')[-1][:-4])
        z = np.load(f)
        for k, v in metrics(f).items():
            agg[name][(k, seed)] = dict(v, ttrain=float(z['ttrain']), tinf=float(z['tinf']), nparam=int(z['nparam']))
    return agg
def cliff(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] > y[None]).sum() - (x[:, None] < y[None]).sum()) / (len(x) * len(y))
def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj
if __name__ == '__main__':
    agg = load_all()
    out = {}
    for name, d in agg.items():
        # average over seeds per kernel first, then mean/sd over kernels
        kern = collections.defaultdict(list)
        for (k, s), v in d.items():
            kern[k].append(v)
        per_k = {k: {key: float(np.mean([v[key] for v in L])) for key in KEYS} for k, L in kern.items()}
        out[name] = {key: (float(np.mean([p[key] for p in per_k.values()])), float(np.std([p[key] for p in per_k.values()]))) for key in KEYS}
        out[name]['per_kernel'] = per_k
        out[name]['n_runs'] = len(d)
        out[name]['ttrain'] = float(np.mean([v['ttrain'] for v in d.values()]))
        out[name]['tinf_ms'] = float(1000 * np.mean([v['tinf'] for v in d.values()]))
        out[name]['nparam'] = int(np.mean([v['nparam'] for v in d.values()]))
    # paired tests vs asunder on kernel-level means (seeds averaged), n = 8 held-out kernels
    tests = {}
    names = [n for n in out if n != 'asunder' and not n.startswith('_')]
    raw = []
    for metric in ['rho_J', 'rho_single', 'rho_twin']:
        for n in names:
            ks = sorted(set(out['asunder']['per_kernel']) & set(out[n]['per_kernel']))
            xs = [out['asunder']['per_kernel'][k][metric] for k in ks]; ys = [out[n]['per_kernel'][k][metric] for k in ks]
            p = wilcoxon(xs, ys).pvalue
            raw.append((metric, n, p, float(np.mean(np.array(xs) - np.array(ys))), float(cliff(xs, ys)), int(sum(x > y for x, y in zip(xs, ys))), len(xs)))
    for metric in ['rho_J', 'rho_single', 'rho_twin']:
        sub = [r for r in raw if r[0] == metric]
        adj = holm([r[2] for r in sub])
        for (mt, n, p, d, cd, wins, nn), pa in zip(sub, adj):
            tests[f'{mt}|{n}'] = dict(p=float(p), p_holm=float(pa), mean_diff=d, cliff=cd, wins=wins, n=nn)
    out['_tests_rhoJ'] = tests
    json.dump(out, open('analysis/pred_stats.json', 'w'), indent=1)
    for n in sorted(out, key=lambda n: -out[n]['rho_J'][0] if not n.startswith('_') else 0):
        if n.startswith('_'): continue
        print(f"{n:13s} " + ' '.join(f"{k}={out[n][k][0]:.3f}±{out[n][k][1]:.3f}" for k in ['rho_J', 'rho_single', 'rho_twin', 'mae_cer', 'top10', 'nll']), out[n]['n_runs'])
    for n, t in tests.items():
        print('test', n, {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()})
