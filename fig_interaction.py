import sys, json
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import KERNEL_ORDER
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7.5,'mathtext.fontset':'dejavuserif'})
rows = [json.loads(l) for k in KERNEL_ORDER for l in open(f'data/{k}.jsonl')]
rate = lambda r, t: r['labels'][t]['counts'][2] / r['labels'][t]['n']
ham = lambda r: sum(r['cfgA'][f] != r['cfgB'][f] for f in r['cfgA'])
rng = np.random.default_rng(0)
def ci(v):
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return np.mean(v), np.percentile(b, 2.5), np.percentile(b, 97.5)
FPS = ['none', 'split', 'g1', 'g2']
fig = plt.figure(figsize=(7.16, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.42)
stats = {}
for j, (t, ttl) in enumerate([('1.2_1', '(a) single spot, $r=1.2$'), ('1.2_2', '(b) twin spot, $r=1.2$')]):
    ax = fig.add_subplot(gs[0, j])
    for k, (div, fc, hatch) in enumerate([('ident', '0.25', ''), ('diverse', 'white', '////')]):
        m, lo, hi = [], [], []
        for fp in FPS:
            v = np.array([100 * rate(r, t) for r in rows if r['fp'] == fp and (ham(r) == 0) == (div == 'ident')])
            a, b, c = ci(v); m.append(a); lo.append(a - b); hi.append(c - a)
            stats[f'{t}_{fp}_{div}'] = (a, b, c, len(v))
        x = np.arange(4) + (k - 0.5) * 0.36
        ax.bar(x, m, 0.34, yerr=[lo, hi], color=fc, edgecolor='black', hatch=hatch, lw=0.6,
               error_kw=dict(lw=0.6, capsize=1.8), label='identical replicas' if div == 'ident' else 'diverse replicas')
    ax.set_xticks(range(4)); ax.set_xticklabels(FPS)
    ax.set_ylabel('critical error rate (%)'); ax.set_title(ttl, fontsize=7.5)
    ax.spines[['top', 'right']].set_visible(False)
    if j == 0:
        ax.legend(frameon=False, fontsize=6.5, loc='upper right')
ax = fig.add_subplot(gs[0, 2])
e0 = json.load(open('analysis/e0.json'))['rho_dbf']
T = ['0.6_1', '1.2_1', '2.0_1', '0.6_2', '1.2_2', '2.0_2']
M = np.array([[e0[k][t] for t in T] for k in KERNEL_ORDER])
im = ax.imshow(M, cmap='Greys', vmin=-0.2, vmax=1.0, aspect='auto')
for i in range(M.shape[0]):
    for jj in range(M.shape[1]):
        ax.text(jj, i, f'{M[i, jj]:.2f}', ha='center', va='center', fontsize=5.6, color='white' if M[i, jj] > 0.55 else 'black')
ax.set_xticks(range(6)); ax.set_xticklabels([f'{t.split("_")[0]}\n{"S" if t.endswith("1") else "T"}' for t in T], fontsize=6)
ax.set_yticks(range(8)); ax.set_yticklabels(KERNEL_ORDER, fontsize=6)
ax.set_title(r'(c) Spearman $\rho$, blind DBF vs spot', fontsize=7.5)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6)
fig.savefig('figs/fig3_interaction.png', dpi=600, bbox_inches='tight')
json.dump({k: list(map(float, v)) for k, v in stats.items()}, open('analysis/fig3_stats.json', 'w'), indent=1)
