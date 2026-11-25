"""Ablation summary: every variant against the seed-0 full model on the same folds, plus the seed spread of the full model."""
import sys, json, os, numpy as np
from evalpred import metrics
from train import KERNEL_ORDER

KEYS = ['rho_J', 'rho_single', 'rho_twin', 'top10', 'mae_cer', 'nll']
VARIANTS = ['asunder', 'no_corr', 'no_prox', 'no_pos', 'pool_readout']
import glob


def per_kernel(name, seed):
    out = {}
    for f in range(4):
        fn = f'results/pred_{name}_f{f}_s{seed}.npz'
        if os.path.exists(fn):
            out.update(metrics(fn))
    return out


res = {}
for v in VARIANTS:
    pk = per_kernel(v, 0)
    if len(pk) < 8:
        continue
    res[v] = {k: [float(np.mean([pk[kn][k] for kn in KERNEL_ORDER])), float(np.std([pk[kn][k] for kn in KERNEL_ORDER]))] for k in KEYS}
    res[v]['per_kernel_rhoJ'] = {kn: float(pk[kn]['rho_J']) for kn in KERNEL_ORDER}
full = res['asunder']
for v in res:
    if v == 'asunder':
        continue
    d = [res[v]['per_kernel_rhoJ'][kn] - full['per_kernel_rhoJ'][kn] for kn in KERNEL_ORDER]
    res[v]['delta_rhoJ'] = float(np.mean(d))
    res[v]['kernels_worse'] = int(sum(x < 0 for x in d))
# seed-to-seed spread of the full model (mean over kernels of each seed, then SD over seeds)
seed_means = {k: [] for k in KEYS}
for s in range(3):
    pk = per_kernel('asunder', s)
    if len(pk) == 8:
        for k in KEYS:
            seed_means[k].append(float(np.mean([pk[kn][k] for kn in KERNEL_ORDER])))
res['_seed_sd'] = {k: float(np.std(v, ddof=1)) if len(v) > 1 else None for k, v in seed_means.items()}
res['_seed_range'] = {k: [min(v), max(v)] if v else None for k, v in seed_means.items()}
Ps = [per_kernel('asunder', s) for s in range(3)]
if all(len(p) == 8 for p in Ps):
    res['_seed_sd_kernel'] = {kn: float(np.std([p[kn]['rho_J'] for p in Ps], ddof=1)) for kn in KERNEL_ORDER}
json.dump(res, open('analysis/ablation.json', 'w'), indent=1)
for v in VARIANTS:
    if v in res:
        print(v, {k: round(res[v][k][0], 3) for k in KEYS}, res[v].get('delta_rhoJ'), res[v].get('kernels_worse'))
print('seed sd', res['_seed_sd'])
print('seed range', res['_seed_range'])
