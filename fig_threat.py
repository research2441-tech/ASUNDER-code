import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dmr import Design
from place import place
from fi import campaign
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7.5,'mathtext.fontset':'dejavuserif'})
REF = dict(u=1, alloc=1, share='le', fsm='bin', sched='asap')
DIV = dict(u=2, alloc=1, share='ded', fsm='onehot', sched='alap')
cases = [('(a) identical, no floorplan', REF, REF, 'none'), ('(b) identical, guard band g2', REF, REF, 'g2'),
         ('(c) diverse, guard band g2', REF, DIV, 'g2')]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.75))
out = []
for ax, (t, a, b, fp) in zip(axs, cases):
    d = Design('fir4', a, b); pl = place(d.nl, fp, 0)
    inp = d.make_inputs(8192, 5)
    c1 = campaign(d, pl, 1.2, 1, 8192, 11, inp)['rates'][2]
    c2 = campaign(d, pl, 1.2, 2, 8192, 12, inp)['rates'][2]
    out.append((t, c1, c2))
    pos, own = pl['pos'], pl['owner']
    for o, mk, fc, lab in [('A', 'o', 'black', 'replica A'), ('B', '^', '0.65', 'replica B'), ('W', 's', 'white', 'comparator W')]:
        m = own == o
        ax.scatter(pos[m, 0], pos[m, 1], s=4 if o != 'W' else 7, marker=mk, c=fc, edgecolors='black' if o == 'W' else 'none',
                   linewidths=0.4, label=lab, zorder=2)
    cx, cy = 1.2 + 0.1 * pl['W'], pl['H'] * 0.55
    ax.add_patch(Circle((cx, cy), 1.2, fill=False, lw=1.0, ls='-', zorder=3))
    if fp != 'none':
        ax.add_patch(Circle((cx + pl['offB'], cy), 1.2, fill=False, lw=1.0, ls='--', zorder=3))
        ax.annotate('', xy=(cx + pl['offB'] - 1.25, cy), xytext=(cx + 1.25, cy),
                    arrowprops=dict(arrowstyle='->', lw=0.7, ls=':'))
        ax.text(cx + pl['offB'] / 2 + 0.6, cy - 0.75, r'$\Delta$', ha='center', fontsize=7.5, bbox=dict(fc='white', ec='none', pad=0.3), zorder=4)
    ax.set_xlim(-1, pl['W']); ax.set_ylim(-1.2, pl['H'] + 0.2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(t, fontsize=7.5, pad=3)
    ax.set_xlabel(f'CER single {100*c1:.2f}%   twin {100*c2:.2f}%', fontsize=7)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=7, markerscale=2.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.2)
fig.savefig('figs/fig2_threat.png', dpi=600, bbox_inches='tight')
print(out)
