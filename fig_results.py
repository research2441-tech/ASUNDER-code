import sys, json, glob
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import KERNEL_ORDER
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7.5,'mathtext.fontset':'dejavuserif'})
ps = json.load(open('analysis/pred_stats.json'))
# ---------------- Fig 4: prediction per held-out kernel + predicted vs measured scatter
models = [('asunder', 'ASUNDER', '0.15', ''), ('gbm', 'GBM + exposure', '0.55', ''), ('gbm_noexp', 'GBM, no exposure', 'white', '////'),
          ('gcn', 'GCN', 'white', '....'), ('gat', 'GAT', 'white', 'xxxx'), ('gin', 'GIN', 'white', '\\\\\\\\')]
models = [m for m in models if m[0] in ps]
fig = plt.figure(figsize=(7.16, 2.7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.28)
ax = fig.add_subplot(gs[0, 0])
w = 0.8 / len(models)
for i, (key, lab, fc, hatch) in enumerate(models):
    v = [ps[key]['per_kernel'][k]['rho_J'] for k in KERNEL_ORDER]
    ax.bar(np.arange(8) + (i - (len(models) - 1) / 2) * w, v, w * 0.95, color=fc, edgecolor='black', hatch=hatch, lw=0.5, label=lab)
ax.set_xticks(range(8)); ax.set_xticklabels(KERNEL_ORDER, fontsize=6.6, rotation=20, ha='right')
ax.set_ylabel(r'Spearman $\rho$ of $J$ (held-out kernel)'); ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0)
ax.axhline(0, color='black', lw=0.5)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(ncol=3, frameon=False, fontsize=6.3, loc='upper left', bbox_to_anchor=(0.0, 1.17))
ax.set_title('(a)', fontsize=7.5, loc='left', x=-0.08, y=1.08)
ax = fig.add_subplot(gs[0, 1])
xs, ys = [], []
for f in glob.glob('results/pred_asunder_f*_s0.npz'):
    z = np.load(f)
    xs += list(z['R'][:, :, 2].max(1)); ys += list(z['P'][:, :, 2].max(1))
xs, ys = 100 * np.array(xs) + 0.01, 100 * np.array(ys) + 0.01
ax.scatter(xs, ys, s=2.5, c='black', alpha=0.35, lw=0)
lim = [0.008, 60]
ax.plot(lim, lim, 'k--', lw=0.6)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'measured $J$ (%)'); ax.set_ylabel(r'predicted $J$ (%)')
ax.set_title('(b)', fontsize=7.5, loc='left', x=-0.25, y=1.08)
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('figs/fig4_prediction.png', dpi=600, bbox_inches='tight')
# ---------------- Fig 5: search
ss = json.load(open('analysis/search_stats.json'))
meth = [('asunder', 'ASUNDER', '-', 'o', 'black'), ('gp_bo', 'GP-BO', '--', 's', '0.35'), ('reinforce', 'REINFORCE', '-.', '^', '0.35'),
        ('random', 'random', ':', 'v', '0.45'), ('asunder_screen', 'ASUNDER screen', '-', 'D', '0.65'), ('gbm_screen', 'GBM screen', '--', 'x', '0.65')]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.3))
ax = axs[0]
for key, lab, ls, mk, c in meth:
    cv = ss[key]['curve']
    ax.plot(np.arange(1, len(cv) + 1), cv, ls=ls, marker=mk, ms=2.6, color=c, lw=0.9, label=lab, markevery=3)
ax.set_yscale('log'); ax.set_xlabel('fault-injection campaigns used'); ax.set_ylabel('normalised regret of best found')
ax.legend(frameon=False, fontsize=6.2); ax.spines[['top', 'right']].set_visible(False)
ax.set_title('(a)', fontsize=7.5, loc='left', x=-0.2)
ax = axs[1]
pk = ss['_per_kernel']
order = ['asunder', 'gp_bo', 'reinforce', 'random', 'asunder_screen', 'gbm_screen', 'asunder_gp', 'diversity']
labels = ['ASUNDER', 'GP-BO', 'REINFORCE', 'random', 'ASUNDER screen', 'GBM screen', 'ASUNDER-GP', 'diversity']
mk = dict(zip(KERNEL_ORDER, ['o', 's', '^', 'v', 'D', 'P', 'X', '*']))
for i, k in enumerate(KERNEL_ORDER):
    if k not in pk: continue
    vals = [pk[k][m] + 0.005 for m in order]
    ax.plot(np.arange(len(order)) + (i - 3.5) * 0.075, vals, ls='none', marker=mk[k], ms=3.2, mfc='white' if i % 2 else 'black', mec='black', mew=0.6, label=k)
ax.set_yscale('log'); ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6.5)
ax.set_ylabel(r're-measured $J$ of returned design (%)')
ax.legend(ncol=4, frameon=False, fontsize=5.8, loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.8)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('(b)', fontsize=7.5, loc='left', x=-0.12, y=1.0)
fig.savefig('figs/fig5_search.png', dpi=600, bbox_inches='tight')
print('ok')
