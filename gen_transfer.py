"""Labels for threats never used in training: interpolated/extrapolated radius and a random double spot."""
import json, sys, os
from dmr import Design
from place import place
from fi import campaign
from train import KERNEL_ORDER
NEW = [(0.9, 1), (0.9, 2), (2.6, 1), (2.6, 2), (1.2, 3)]
N, LANES = int(os.environ.get('ASUNDER_NTRANSFER', 100)), int(os.environ.get('ASUNDER_LANES_T', 2048))
if __name__ == '__main__':
    for kn in sys.argv[1].split(','):
        out = {}
        for l in open(f'data/{kn}.jsonl'):
            r = json.loads(l)
            if r['idx'] >= N:
                continue
            d = Design(kn, r['cfgA'], r['cfgB']); pl = place(d.nl, r['fp'], 0)
            inp = d.make_inputs(LANES, 77000 + r['idx'])
            lab = {}
            for i, (rr, m) in enumerate(NEW):
                res = campaign(d, pl, rr, m, LANES, 88000 + 10 * r['idx'] + i, inp)
                lab[f'{rr}_{m}'] = dict(counts=res['counts'].tolist(), n=res['n'], cm=res['cm'], anyw=res['anyw'])
            out[r['idx']] = lab
        json.dump(out, open(f'data/transfer_{kn}.json', 'w'))
        print(kn, len(out), flush=True)
