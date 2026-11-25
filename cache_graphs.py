import json, pickle, sys, os
import numpy as np
from dmr import Design
from place import place
from graph import build_graph
from hls import cfg_key
THREATS = ['0.6_1','1.2_1','2.0_1','0.6_2','1.2_2','2.0_2']
def build(rec):
    d = Design(rec['kernel'], rec['cfgA'], rec['cfgB'])
    pl = place(d.nl, rec['fp'], 0)
    g = build_graph(d, pl)
    g['labels'] = rec['labels']; g['meta'] = {k: rec[k] for k in rec if k not in ('labels',)}
    return g
if __name__ == '__main__':
    os.makedirs('graphs', exist_ok=True)
    for kn in sys.argv[1].split(','):
        recs = [json.loads(l) for l in open(f'data/{kn}.jsonl')]
        gs = [build(r) for r in recs]
        pickle.dump(gs, open(f'graphs/{kn}.pkl', 'wb'))
        print(kn, len(gs), flush=True)
