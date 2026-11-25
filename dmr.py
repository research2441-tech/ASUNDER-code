"""Compose a DMR design (replica A, replica B, comparator wrapper) and run campaigns."""
import random
import numpy as np
from kernels import KERNELS, random_inputs, input_layout, output_layout, pack
from hls import Replica, WRAPPER, cfg_key
from netlist import synthesize, Netlist, lanes_to_words, words_to_lanes

WORK = 'work'


def replica_json(kname, cfg):
    k = KERNELS[kname]
    r = Replica(k, cfg)
    js = synthesize(r.verilog('rep'), 'rep', f"{WORK}/{kname}", cfg_key(cfg))
    return r, js


def wrapper_json(kname):
    k = KERNELS[kname]
    wout = sum(w for _, _, w in output_layout(k))
    return synthesize(WRAPPER.format(W1=wout - 1), 'wrap', f"{WORK}/{kname}", 'wrap')


class Design:
    def __init__(self, kname, cfgA, cfgB):
        self.kname, self.k = kname, KERNELS[kname]
        self.cfgA, self.cfgB = cfgA, cfgB
        self.rA, jsA = replica_json(kname, cfgA)
        self.rB, jsB = replica_json(kname, cfgB)
        jsW = wrapper_json(kname)
        nl = Netlist()
        nl.add_module(jsA, 'rep', 'A')
        din = nl.ports[('A', 'din')]
        nl.add_module(jsB, 'rep', 'B', bind={'din': din})
        nl.add_module(jsW, 'wrap', 'W', bind={'oa': nl.ports[('A', 'dout')], 'ob': nl.ports[('B', 'dout')],
                                               'da': nl.ports[('A', 'done')], 'db': nl.ports[('B', 'done')]})
        nl.compile()
        self.nl = nl
        self.lat = max(self.rA.latency(), self.rB.latency())
        self.T = self.lat + 3

    def make_inputs(self, n_lanes, seed):
        rng = random.Random(seed)
        lay = input_layout(self.k)
        outl = output_layout(self.k)
        din = self.nl.ports[('A', 'din')]
        bits = np.zeros((len(din), n_lanes), dtype=bool)
        gold = np.zeros((sum(w for _, _, w in outl), n_lanes), dtype=bool)
        for l in range(n_lanes):
            I = random_inputs(self.k, rng)
            v, nb = pack(lay, I)
            for b in range(nb):
                bits[b, l] = (v >> b) & 1
            O = self.k['golden'](I)
            g, ng = pack(outl, O)
            for b in range(ng):
                gold[b, l] = (g >> b) & 1
        words = lanes_to_words(bits)
        return {din[b]: words[b] for b in range(len(din))}, gold

    def run(self, inputs, ff_flip=None, lut_flip=None, t_inj=None):
        nl = self.nl
        V, _ = nl.simulate(inputs, self.T, ff_flip, lut_flip, t_inj)
        q = words_to_lanes(V[nl.ports[('W', 'q')]])
        alarm = words_to_lanes(V[nl.ports[('W', 'alarm')]])[0]
        valid = words_to_lanes(V[nl.ports[('W', 'valid')]])[0]
        oa = words_to_lanes(V[nl.ports[('A', 'dout')]])
        ob = words_to_lanes(V[nl.ports[('B', 'dout')]])
        return q, alarm, valid, oa, ob

    def check(self, n_lanes=128, seed=1):
        inp, gold = self.make_inputs(n_lanes, seed)
        q, alarm, valid, oa, ob = self.run(inp)
        ok = valid.all() and (~alarm).all() and (q == gold).all()
        return ok, dict(valid=valid.mean(), alarm=alarm.mean(), qerr=(q != gold).any(0).mean(),
                        aerr=(oa != gold).any(0).mean(), berr=(ob != gold).any(0).mean())


if __name__ == '__main__':
    import sys, time
    from hls import all_configs
    names = sys.argv[1:] or list(KERNELS)
    for kn in names:
        cfgs = all_configs(KERNELS[kn])
        rng = random.Random(3)
        for trial in range(3):
            a, b = rng.choice(cfgs), rng.choice(cfgs)
            t0 = time.time()
            d = Design(kn, a, b)
            t1 = time.time()
            ok, info = d.check()
            t2 = time.time()
            print(kn, cfg_key(a), cfg_key(b), 'LUT', len(d.nl.luts), 'FF', len(d.nl.ffs), 'lat', d.rA.latency(), d.rB.latency(),
                  'OK' if ok else 'FAIL', info, f"syn {t1 - t0:.1f}s sim {t2 - t1:.2f}s", flush=True)
