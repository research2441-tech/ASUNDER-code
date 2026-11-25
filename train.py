"""Training and evaluation utilities for the surrogate models."""
import pickle, time, copy
import numpy as np
import torch
from model import Asunder, GraphBaseline, sample_spots, stratified_spots, loss_fn, collate

THREATS = [(0.6, 1), (1.2, 1), (2.0, 1), (0.6, 2), (1.2, 2), (2.0, 2)]
TKEY = lambda t: f"{t[0]}_{t[1]}"
KERNEL_ORDER = ['aes_ark', 'aes_mix', 'present', 'speck', 'crc16', 'fir4', 'dct4', 'cordic']
FOLDS = [('aes_ark', 'crc16'), ('aes_mix', 'fir4'), ('present', 'dct4'), ('speck', 'cordic')]


def to_torch(g):
    if '_t' in g:
        return g['_t']
    t = dict(x=torch.tensor(g['x']), pos=torch.tensor(g['pos']), owner=torch.tensor(g['owner']),
             e_net=torch.tensor(g['e_net']), e_prox=torch.tensor(g['e_prox']), a_prox=torch.tensor(g['a_prox']),
             e_corr=torch.tensor(g['e_corr']), a_corr=torch.tensor(g['a_corr']),
             W=g['W'], H=g['H'], offB=g['offB'])
    g['_t'] = t
    return t


def labels_of(g, threats):
    L = [g['labels'][TKEY(t)] for t in threats]
    return dict(counts=torch.tensor([l['counts'] for l in L], dtype=torch.float32),
                n=torch.tensor([l['n'] for l in L], dtype=torch.float32),
                cm=torch.tensor([l['cm'] for l in L], dtype=torch.float32),
                anyw=torch.tensor([l['anyw'] for l in L], dtype=torch.float32))


def load(kernels):
    out = []
    for k in kernels:
        out += pickle.load(open(f'graphs/{k}.pkl', 'rb'))
    return out


def make_model(spec):
    kind = spec.get('kind', 'asunder')
    if kind == 'asunder':
        return Asunder(**{k: v for k, v in spec.items() if k not in ('kind', 'name')})
    return GraphBaseline(kind)


def run_epoch(model, graphs, opt, rng, threats, G=36, bs=8):
    model.train()
    idx = rng.permutation(len(graphs))
    tot = 0.0
    for b0 in range(0, len(idx), bs):
        opt.zero_grad()
        loss = 0.0
        bidx = idx[b0:b0 + bs]
        gl = [to_torch(graphs[i]) for i in bidx]
        B = collate(gl)
        H = model.encode(B)
        for j, i in enumerate(bidx):
            g = gl[j]
            h = H[B['ptr'][j]:B['ptr'][j + 1]]
            sp = sample_spots(g, threats, G, rng)
            P, pcm, pan, _ = model.readout(g, h, threats, sp)
            loss = loss + loss_fn(P, pcm, pan, labels_of(graphs[i], threats))
        loss = loss / len(idx[b0:b0 + bs])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += float(loss.detach()) * len(bidx)
    return tot / len(idx)


@torch.no_grad()
def predict(model, graphs, threats=THREATS, G=144, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    P = np.zeros((len(graphs), len(threats), 4))
    CM = np.zeros((len(graphs), len(threats)))
    AN = np.zeros((len(graphs), len(threats)))
    for b0 in range(0, len(graphs), 16):
        gl = [to_torch(gr) for gr in graphs[b0:b0 + 16]]
        B = collate(gl)
        H = model.encode(B)
        for j, g in enumerate(gl):
            h = H[B['ptr'][j]:B['ptr'][j + 1]]
            sp = stratified_spots(g, threats, G, rng)
            p, pcm, pan, _ = model.readout(g, h, threats, sp)
            P[b0 + j], CM[b0 + j], AN[b0 + j] = p.numpy(), pcm.numpy(), pan.numpy()
    return P, CM, AN


@torch.no_grad()
def val_loss(model, graphs, threats, seed=7):
    model.eval()
    rng = np.random.default_rng(seed)
    tot = 0.0
    for b0 in range(0, len(graphs), 16):
        grs = graphs[b0:b0 + 16]
        gl = [to_torch(gr) for gr in grs]
        B = collate(gl)
        H = model.encode(B)
        for j, g in enumerate(gl):
            h = H[B['ptr'][j]:B['ptr'][j + 1]]
            sp = stratified_spots(g, threats, 64, rng)
            P, pcm, pan, _ = model.readout(g, h, threats, sp)
            tot += float(loss_fn(P, pcm, pan, labels_of(grs[j], threats)))
    return tot / len(graphs)


def fit(spec, train_g, val_g, threats=THREATS, epochs=30, seed=0, lr=2e-3, patience=8, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(spec)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad = 1e9, None, 0
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        tl = run_epoch(model, train_g, opt, rng, threats)
        vl = val_loss(model, val_g, threats)
        sched.step()
        hist.append((tl, vl))
        if verbose:
            print(f"  ep {ep} train {tl:.4f} val {vl:.4f} {time.time() - t0:.0f}s", flush=True)
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def truth(graphs, threats=THREATS):
    R = np.zeros((len(graphs), len(threats), 4))
    CMI = np.zeros((len(graphs), len(threats)))
    for i, g in enumerate(graphs):
        for j, t in enumerate(threats):
            l = g['labels'][TKEY(t)]
            R[i, j] = np.array(l['counts']) / l['n']
            CMI[i, j] = l['cm'] / max(l['anyw'], 1)
    return R, CMI


def split_val(graphs, frac=0.12, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    nv = int(len(graphs) * frac)
    return [graphs[i] for i in idx[nv:]], [graphs[i] for i in idx[:nv]]
