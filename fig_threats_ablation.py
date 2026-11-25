"""Fig. 6: ranking quality per threat (training and unseen threats). Fig. 7: ablation per kernel. Fig. 8: per-position risk maps."""
import sys, json, os, glob
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from train import KERNEL_ORDER
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5, 'mathtext.fontset': 'dejavuserif',
                     'axes.spines.top': False, 'axes.spines.right': False})


def per_threat(name, seeds=(0, 1, 2)):
    """Mean over kernels of the seed-averaged Spearman correlation for each of the six training threats."""
    acc = {kn: [] for kn in KERNEL_ORDER}
    for s in seeds:
        for f in range(4):
            fn = f'results/pred_{name}_f{f}_s{s}.npz'
            if not os.path.exists(fn):
                continue
            z = np.load(fn, allow_pickle=True)
            P, R, ker = z['P'], z['R'], z['ker']
            for kn in np.unique(ker):
                m = ker == kn
                acc[str(kn)].append([spearmanr(P[m][:, t, 2], R[m][:, t, 2]).correlation for t in range(6)])
    per_k = np.array([np.nanmean(acc[kn], 0) for kn in KERNEL_ORDER if acc[kn]])
    return per_k.mean(0), per_k.std(0)


# ---------------------------------------------------------------- Fig. 6
models = [('asunder', 'ASUNDER', 'o', 'black', '-'), ('gbm', 'GBM + exposure', 's', '0.45', '--'),
          ('gbm_noexp', 'GBM, no exposure', 'D', '0.65', ':'), ('gcn', 'GCN', 'v', '0.3', '-.')]
train_lab = ['0.6\nsingle', '1.2\nsingle', '2.0\nsingle', '0.6\ntwin', '1.2\ntwin', '2.0\ntwin']
tr = json.load(open('analysis/transfer.json'))
new_keys = ['0.9_1', '2.6_1', '0.9_2', '2.6_2', '1.2_3']
new_lab = ['0.9\nsingle', '2.6\nsingle', '0.9\ntwin', '2.6\ntwin', '1.2\nrandom\ndouble']
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={'width_ratios': [6, 5], 'wspace': 0.28})
ax = axs[0]
x = np.arange(6)
for i, (k, lab, mk, c, ls) in enumerate(models):
    mu, sd = per_threat(k)
    xx = x + (i - 1.5) * 0.09
    ax.errorbar(xx, mu, yerr=sd, fmt=mk, color=c, mfc=c if i % 2 == 0 else 'white', mec=c, ms=4, lw=0.8, capsize=1.5, label=lab)
    ax.plot(xx[:3], mu[:3], ls=ls, color=c, lw=0.7); ax.plot(xx[3:], mu[3:], ls=ls, color=c, lw=0.7)
ax.axvline(2.5, color='0.8', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(train_lab, fontsize=6.6)
ax.set_ylim(-0.1, 1.0); ax.set_ylabel('Spearman $\\rho_s$ of CER (held-out kernel)')
ax.set_xlabel('training threat: radius (tiles) and spot mode')
ax.set_title('(a)', fontsize=7.5, loc='left', x=-0.16, y=1.02)
ax.legend(ncol=2, frameon=False, fontsize=6.0, loc='lower center', bbox_to_anchor=(0.55, 1.03), handletextpad=0.3, columnspacing=1.2)
ax = axs[1]
x = np.arange(5)
series = [('ASUNDER', [np.mean([np.mean([tr[kn][f'asunder_s{s}'][t]['rho'] for s in range(3) if f'asunder_s{s}' in tr[kn]]) for kn in tr]) for t in new_keys],
           [np.std([np.mean([tr[kn][f'asunder_s{s}'][t]['rho'] for s in range(3) if f'asunder_s{s}' in tr[kn]]) for kn in tr]) for t in new_keys], 'o', 'black'),
          ('GBM, single-spot code', [np.mean([tr[kn]['gbm_m1'][t]['rho'] for kn in tr]) for t in new_keys],
           [np.std([tr[kn]['gbm_m1'][t]['rho'] for kn in tr]) for t in new_keys], 's', '0.45'),
          ('GBM, twin-spot code', [np.mean([tr[kn]['gbm_m2'][t]['rho'] for kn in tr]) for t in new_keys],
           [np.std([tr[kn]['gbm_m2'][t]['rho'] for kn in tr]) for t in new_keys], '^', '0.6')]
for i, (lab, mu, sd, mk, c) in enumerate(series):
    xx = x + (i - 1) * 0.12
    ax.errorbar(xx, mu, yerr=sd, fmt=mk, color=c, mfc=c if i != 2 else 'white', mec=c, ms=4, lw=0.8, capsize=1.5, label=lab)
ax.axvline(3.5, color='0.8', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(new_lab, fontsize=6.6)
ax.set_ylim(-0.1, 1.0)
ax.set_xlabel('unseen threat')
ax.set_title('(b)', fontsize=7.5, loc='left', x=-0.16, y=1.02)
ax.legend(ncol=2, frameon=False, fontsize=6.0, loc='lower center', bbox_to_anchor=(0.55, 1.03), handletextpad=0.3, columnspacing=1.2)
fig.savefig('figs/fig6_threats.png', dpi=600, bbox_inches='tight')

# ---------------------------------------------------------------- Fig. 7
ab = json.load(open('analysis/ablation.json'))
var = [('no_corr', 'no correspondence edges', 'o', 'black'), ('no_prox', 'no proximity edges', 's', '0.35'),
       ('no_pos', 'no encoder geometry', '^', '0.55'), ('pool_readout', 'pooled readout', 'D', '0.7')]
var = [v for v in var if v[0] in ab]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = np.arange(8)
sdk = ab.get('_seed_sd_kernel', {})
for j, kn in enumerate(KERNEL_ORDER):
    if kn in sdk:
        ax.fill_between([j - 0.42, j + 0.42], -2 * sdk[kn], 2 * sdk[kn], color='0.9', lw=0)
ax.axhline(0, color='0.5', lw=0.6)
for i, (k, lab, mk, c) in enumerate(var):
    d = [ab[k]['per_kernel_rhoJ'][kn] - ab['asunder']['per_kernel_rhoJ'][kn] for kn in KERNEL_ORDER]
    ax.plot(x + (i - 1.5) * 0.15, d, mk, color=c, mfc=c if i % 2 == 0 else 'white', mec=c, ms=4, label=lab, lw=0)
ax.set_xticks(x); ax.set_xticklabels(KERNEL_ORDER, fontsize=6.4, rotation=25, ha='right')
ax.set_ylabel('$\\Delta\\rho_s$ of $J$ relative to full model')
ax.legend(ncol=2, frameon=False, fontsize=5.9, loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.7)
fig.savefig('figs/fig7_ablation.png', dpi=600, bbox_inches='tight')

# ---------------------------------------------------------------- Fig. 8
if not os.path.exists('analysis/riskmap.json'):
    print('ok (no risk map)'); sys.exit(0)
rm = json.load(open('analysis/riskmap.json'))['ident_g2_m2']
nx, ny = len(rm['xs']), len(rm['ys'])
E = 100 * np.array(rm['emp']).reshape(ny, nx); P = 100 * np.array(rm['pred']).reshape(ny, nx)
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.2), gridspec_kw={'wspace': 0.75})
for ax, M, t in [(axs[0], E, '(a) measured'), (axs[1], P, '(b) predicted')]:
    im = ax.imshow(M, origin='lower', cmap='Greys', aspect='auto',
                   extent=[rm['xs'][0] - 0.5 * (rm['xs'][1] - rm['xs'][0]), rm['xs'][-1] + 0.5 * (rm['xs'][1] - rm['xs'][0]),
                           rm['ys'][0] - 0.5 * (rm['ys'][1] - rm['ys'][0]), rm['ys'][-1] + 0.5 * (rm['ys'][1] - rm['ys'][0])])
    ax.set_title(t, fontsize=7.2, loc='left')
    ax.set_xlabel('spot $x$ (tiles)')
    if ax is axs[0]:
        ax.set_ylabel('spot $y$ (tiles)')
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04); cb.ax.tick_params(labelsize=6); cb.set_label('CER (%)', fontsize=6.5)
fig.savefig('figs/fig8_riskmap.png', dpi=600, bbox_inches='tight')
print('ok')
