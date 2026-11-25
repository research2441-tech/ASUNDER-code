import json, random, sys, time, os
import numpy as np
from kernels import KERNELS
from hls import all_configs, cfg_key
from dmr import Design
from place import place, FLOORPLANS
from fi import campaign

THREATS = [(0.6, 1), (1.2, 1), (2.0, 1), (0.6, 2), (1.2, 2), (2.0, 2)]
REF = dict(u=1, alloc=1, share='le', fsm='bin', sched='asap')
NL = int(os.environ.get('ASUNDER_LANES', 4096))

def evaluate(kn, a, b, fp, seed, n_lanes=NL, threats=THREATS, dbf=True):
    d = Design(kn, a, b)
    pl = place(d.nl, fp, 0)
    inp = d.make_inputs(n_lanes, seed)
    rec = dict(kernel=kn, A=cfg_key(a), B=cfg_key(b), cfgA=a, cfgB=b, fp=fp, seed=seed,
               lat=d.lat, latA=d.rA.latency(), latB=d.rB.latency(), n_lut=len(d.nl.luts), n_ff=len(d.nl.ffs),
               n_cell=len(pl['cells']), W=pl['W'], H=pl['H'], area=pl['area'], hpwl=pl['hpwl'], labels={})
    for i, (r, m) in enumerate(threats):
        res = campaign(d, pl, r, m, n_lanes, seed * 31 + i, inp)
        rec['labels'][f"{r}_{m}"] = dict(counts=res['counts'].tolist(), n=res['n'], cm=res['cm'], anyw=res['anyw'])
    if dbf:
        res = campaign(d, pl, 0, 0, n_lanes, seed * 31 + 99, inp, model='dbf')
        rec['labels']['dbf'] = dict(counts=res['counts'].tolist(), n=res['n'], cm=res['cm'], anyw=res['anyw'])
    return rec

def designs_for(kn, n=None, seed=0):
    n = n or int(os.environ.get('ASUNDER_N', 300))
    cf = all_configs(KERNELS[kn]); rng = random.Random(seed + sum(map(ord, kn)))
    out = [(REF, REF, fp) for fp in FLOORPLANS]
    seen = set((cfg_key(a), cfg_key(b), fp) for a, b, fp in out)
    while len(out) < n:
        a = rng.choice(cf)
        b = a if rng.random() < 0.2 else rng.choice(cf)
        fp = rng.choice(FLOORPLANS)
        key = (cfg_key(a), cfg_key(b), fp)
        if key in seen: continue
        seen.add(key); out.append((a, b, fp))
    return out

if __name__ == '__main__':
    kns = sys.argv[1].split(',')
    os.makedirs('data', exist_ok=True)
    for kn in kns:
        fn = f'data/{kn}.jsonl'
        done = sum(1 for _ in open(fn)) if os.path.exists(fn) else 0
        ds = designs_for(kn)
        t0 = time.time()
        with open(fn, 'a') as f:
            for i, (a, b, fp) in enumerate(ds):
                if i < done: continue
                rec = evaluate(kn, a, b, fp, seed=1000 + i)
                rec['idx'] = i
                f.write(json.dumps(rec) + '\n'); f.flush()
                if i % 25 == 0: print(kn, i, f"{time.time()-t0:.0f}s", flush=True)
