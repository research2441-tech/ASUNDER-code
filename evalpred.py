import sys, glob, json, numpy as np
from scipy.stats import spearmanr
THR = ['0.6_1','1.2_1','2.0_1','0.6_2','1.2_2','2.0_2']
def metrics(f):
    z = np.load(f, allow_pickle=True)
    P, R, CM, AN, CMI, ker = z['P'], z['R'], z['CM'], z['AN'], z['CMI'], z['ker']
    out = {}
    for k in np.unique(ker):
        m = ker == k
        pc, rc = P[m][:, :, 2], R[m][:, :, 2]
        J_p, J_r = pc.max(1), rc.max(1)
        rho_t = [spearmanr(pc[:, t], rc[:, t]).correlation for t in range(6)]
        n = 4096
        nll = -(R[m] * n * np.log(P[m] + 1e-6)).sum(-1).mean() / n
        cmi_p = CM[m] / np.maximum(AN[m], 1e-6) if AN[m].max() > 0 else CM[m]
        k10 = max(1, int(0.1 * m.sum()))
        top_true = set(np.argsort(J_r)[:k10]); top_pred = set(np.argsort(J_p)[:k10])
        out[k] = dict(mae_cer=100 * np.abs(pc - rc).mean(), mae_all=100 * np.abs(P[m] - R[m]).mean(),
                      rho_cer=np.nanmean(rho_t), rho_J=spearmanr(J_p, J_r).correlation, nll=nll,
                      cmi_mae=np.abs(np.clip(cmi_p, 0, 1) - CMI[m]).mean(), top10=len(top_true & top_pred) / k10,
                      rho_single=np.nanmean(rho_t[:3]), rho_twin=np.nanmean(rho_t[3:]))
    return out
if __name__ == '__main__':
    for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'results/pred_*.npz')):
        for k, v in metrics(f).items():
            print(f.split('/')[-1], k, {a: round(float(b), 3) for a, b in v.items()})
