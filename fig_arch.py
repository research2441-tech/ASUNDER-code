import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif'})
fig, ax = plt.subplots(figsize=(7.16, 4.1))
ax.set_xlim(-4, 101); ax.set_ylim(-1, 57); ax.axis('off')
W, H = 17.0, 15.0
def box(x, y, title, lines, fill='white'):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0,rounding_size=1.0', fc=fill, ec='black', lw=0.9))
    ax.text(x + W/2, y + H - 1.0, title, ha='center', va='top', fontsize=6.5, fontweight='bold', linespacing=1.1)
    ax.text(x + W/2, y + 1.2, '\n'.join(lines), ha='center', va='bottom', fontsize=6.2, linespacing=1.3)
def arr(pts, ls='-', label=None, lpos=None, ha='center'):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color='black', lw=0.9, ls=ls)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=8, lw=0.9, ls=ls,
                                 color='black', shrinkA=0, shrinkB=0))
    if label:
        ax.text(lpos[0], lpos[1], label, fontsize=6.3, ha=ha, va='center', style='italic',
                bbox=dict(fc='white', ec='none', pad=0.4))
X = [1, 21.5, 42, 62.5, 83]
YT, YB = 34, 8
box(X[0], YT, 'Directive pair', [r'replica A: $\theta_A$', r'replica B: $\theta_B$', r'floorplan: $\varphi$'], '#eeeeee')
box(X[1], YT, 'HLS + synthesis', ['schedule, bind,', 'FSM emission,', 'LUT4/FF netlists', 'A, B and W'])
box(X[2], YT, 'Placement', ['quadratic solve,', 'bisection', 'legaliser, guard', r'band from $\varphi$'])
box(X[3], YT, 'Cross-replica\ngraph', ['netlist edges,', 'proximity edges,', 'correspondence', 'edges'])
box(X[4], YT, 'Typed-attention\nencoder', [r'$L$ layers,', 'one projection', 'set per edge', 'type'])
box(X[0], YB, 'Training loss', ['multinomial NLL,', 'binomial terms,', 'relative critical', r'deviance $\mathcal{L}_C$'], '#eeeeee')
box(X[1], YB, 'Spot-fault\ncampaigns', ['single and twin', r'spots of radius $r$,', 'outcome counts', 'and CMI'])
box(X[2], YB, 'Budgeted search', ['ensemble LCB,', 'few-shot', 'recalibration,', 'area/latency caps'])
box(X[3], YB, 'Outcome law', [r'$\hat{\pi}_S,\hat{\pi}_D,\hat{\pi}_C,\hat{\pi}_H$', r'and $\widehat{\mathrm{CMI}}$', r'per threat $\tau$'])
box(X[4], YB, 'Spot-integral\nreadout', [r'soft hit weights', r'$\omega_v(\mathbf{c};r)$, owner', 'pooling, mean', 'over spots'])
ym = YT + H/2
for i in range(4):
    arr([(X[i] + W, ym), (X[i + 1], ym)])
# encoder -> readout
arr([(X[4] + W/2, YT), (X[4] + W/2, YB + H)], label=r'$\mathbf{h}_v^{(L)}$', lpos=(X[4] + W/2 + 4.2, 28.5))
yb = YB + H/2
arr([(X[4], yb), (X[3] + W, yb)])
arr([(X[3], yb), (X[2] + W, yb)])
arr([(X[2], yb), (X[1] + W, yb)])
arr([(X[1], yb), (X[0] + W, yb)])
# search -> directive pair (proposals) through the gap between rows
xs = X[2] + W/2
arr([(xs, YB + H), (xs, 28.0), (X[0] + W/2, 28.0), (X[0] + W/2, YT)],
    label=r'proposals $(\theta_A,\theta_B,\varphi)$', lpos=(26.5, 28.0))
# outcome law -> loss, routed below the bottom row
xo = X[3] + W/2
arr([(xo, YB), (xo, 3.5), (X[0] + W/2, 3.5), (X[0] + W/2, YB)],
    label=r'predictions $\hat{\pi}$, $\widehat{\mathrm{CMI}}$', lpos=(40, 3.5))
# reverse-mode gradient, dashed, left margin then over the top row into the encoder
arr([(X[0], YB + H/2), (-2.5, YB + H/2), (-2.5, 53.5), (X[4] + W/2, 53.5), (X[4] + W/2, YT + H)], ls=(0, (4, 2.5)),
    label=r'reverse-mode gradient $\nabla_{\Theta}\mathcal{L}$ (training only)', lpos=(48, 53.5))
fig.savefig('figs/fig1_framework.png', dpi=600, bbox_inches='tight')
