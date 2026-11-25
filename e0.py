import json, numpy as np
from scipy.stats import spearmanr
from train import KERNEL_ORDER
T=['0.6_1','1.2_1','2.0_1','0.6_2','1.2_2','2.0_2']
rows=[]
for kn in KERNEL_ORDER:
    for l in open(f'data/{kn}.jsonl'):
        r=json.loads(l); rows.append(r)
def rate(r,t,o=2): L=r['labels'][t]; return L['counts'][o]/L['n']
def ham(r): return sum(r['cfgA'][f]!=r['cfgB'][f] for f in r['cfgA'])
out={}
# 1. rank agreement DBF vs spatial
rho={}
for kn in KERNEL_ORDER:
    R=[r for r in rows if r['kernel']==kn]
    d=np.array([rate(r,'dbf') for r in R])
    rho[kn]={t: spearmanr(d,[rate(r,t) for r in R]).correlation for t in T}
    rho[kn]['J']=spearmanr(d,[max(rate(r,t) for t in T) for r in R]).correlation
out['rho_dbf']=rho
print('Spearman DBF-CER vs spatial CER:')
for kn in KERNEL_ORDER: print(kn, {k: round(v,2) for k,v in rho[kn].items()})
print('mean', {t: round(np.mean([rho[k][t] for k in KERNEL_ORDER]),2) for t in T+['J']})
# 2. floorplan x identical/diverse
print('\nmean CER (%) by floorplan and diversity (all kernels)')
tab={}
for fp in ['none','split','g1','g2']:
    for div in ['ident','diverse']:
        S=[r for r in rows if r['fp']==fp and ((ham(r)==0)==(div=='ident'))]
        tab[f'{fp}_{div}']={t: 100*np.mean([rate(r,t) for r in S]) for t in T}
        tab[f'{fp}_{div}']['n']=len(S)
        tab[f'{fp}_{div}']['J']=100*np.mean([max(rate(r,t) for t in T) for r in S])
        tab[f'{fp}_{div}']['cmi12_2']=np.mean([r['labels']['1.2_2']['cm']/max(r['labels']['1.2_2']['anyw'],1) for r in S])
        print(fp,div,len(S),{k: round(v,3) for k,v in tab[f'{fp}_{div}'].items()})
out['fp_div']=tab
# 3. per-kernel stats
print('\nper kernel: mean/median/min J, REF J (none), LUT range')
pk={}
for kn in KERNEL_ORDER:
    R=[r for r in rows if r['kernel']==kn]
    J=np.array([max(rate(r,t) for t in T) for r in R])
    ref=[r for r in R if r['idx']==0][0]
    pk[kn]=dict(n=len(R), J_mean=100*J.mean(), J_med=100*np.median(J), J_min=100*J.min(), J_max=100*J.max(),
                J_ref=100*max(rate(ref,t) for t in T), lut_min=min(r['n_lut'] for r in R), lut_max=max(r['n_lut'] for r in R),
                ff_min=min(r['n_ff'] for r in R), ff_max=max(r['n_ff'] for r in R), lat_min=min(r['lat'] for r in R), lat_max=max(r['lat'] for r in R),
                cells_min=min(r['n_cell'] for r in R), cells_max=max(r['n_cell'] for r in R),
                S=100*np.mean([rate(r,'1.2_1',0) for r in R]), D=100*np.mean([rate(r,'1.2_1',1) for r in R]), H=100*np.mean([rate(r,'1.2_1',3) for r in R]))
    print(kn, {k: round(v,3) if isinstance(v,float) else v for k,v in pk[kn].items()})
out['per_kernel']=pk
# 4. hamming vs CER twin
print('\nSpearman Hamming distance vs CER (twin r1.2) by floorplan')
hd={}
for fp in ['none','split','g1','g2']:
    S=[r for r in rows if r['fp']==fp]
    hd[fp]=spearmanr([ham(r) for r in S],[rate(r,'1.2_2') for r in S]).correlation
print({k: round(v,3) for k,v in hd.items()})
out['ham_rho']=hd
json.dump(out, open('analysis/e0.json','w'), indent=1, default=float)
