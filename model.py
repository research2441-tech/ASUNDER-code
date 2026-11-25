"""Cross-replica typed-attention encoder with a spot-integral readout, plus graph baselines."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import F_DIM

TAU = 0.15
NOUT = 4


def scatter_softmax(score, index, n):
    # score [E, H]
    mx = torch.full((n, score.shape[1]), -1e30, device=score.device)
    mx = mx.scatter_reduce(0, index[:, None].expand_as(score), score, reduce='amax', include_self=True)
    ex = torch.exp(score - mx[index])
    den = torch.zeros((n, score.shape[1]), device=score.device).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


class TypedAttention(nn.Module):
    def __init__(self, h, heads, etypes):
        super().__init__()
        self.h, self.H, self.dh = h, heads, h // heads
        self.etypes = etypes          # dict name -> attr dim
        self.q = nn.ModuleDict({t: nn.Linear(h, h, bias=False) for t in etypes})
        self.k = nn.ModuleDict({t: nn.Linear(h, h, bias=False) for t in etypes})
        self.v = nn.ModuleDict({t: nn.Linear(h, h, bias=False) for t in etypes})
        self.ab = nn.ModuleDict({t: nn.Sequential(nn.Linear(a, 16), nn.ReLU(), nn.Linear(16, heads))
                                 for t, a in etypes.items() if a > 0})
        self.o = nn.Linear(h, h)
        self.n1, self.n2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))

    def forward(self, x, edges, gates=None):
        n = x.shape[0]
        out = torch.zeros_like(x)
        for t in self.etypes:
            if t not in edges or edges[t][0].shape[1] == 0:
                continue
            e, a = edges[t]
            s, d = e[0], e[1]
            q = self.q[t](x)[d].view(-1, self.H, self.dh)
            k = self.k[t](x)[s].view(-1, self.H, self.dh)
            v = self.v[t](x)[s].view(-1, self.H, self.dh)
            sc = (q * k).sum(-1) / math.sqrt(self.dh)
            if t in self.ab:
                sc = sc + self.ab[t](a)
            al = scatter_softmax(sc, d, n)
            msg = (al[..., None] * v).reshape(-1, self.h)
            agg = torch.zeros_like(x).index_add_(0, d, msg)
            g = 1.0 if gates is None else gates[t]
            out = out + g * agg
        x = self.n1(x + self.o(out))
        return self.n2(x + self.ff(x))


def edge_attr(t, a):
    if t == 'prox':
        return torch.cat([a / 2.5, torch.exp(-a ** 2)], 1)
    if t == 'corr':
        return torch.cat([a[:, :1], a[:, 1:] / 5.0, torch.exp(-a[:, 1:] ** 2)], 1)
    return a


class Asunder(nn.Module):
    """Variants:
       use_corr / use_prox / use_pos : input ablations
       typed : separate parameters per edge type (else one merged type)
       readout : 'spot' (spot-integral) or 'pool' (global pooling with threat concatenated)"""

    def __init__(self, h=64, heads=4, layers=3, use_corr=True, use_prox=True, use_pos=True,
                 typed=True, readout='spot'):
        super().__init__()
        self.cfg = dict(use_corr=use_corr, use_prox=use_prox, use_pos=use_pos, typed=typed, readout=readout)
        et = {'net': 0, 'rnet': 0}
        if use_prox:
            et['prox'] = 2
        if use_corr:
            et['corr'] = 5
        if not typed:
            et = {'all': 6}
        self.etypes = et
        self.inp = nn.Linear(F_DIM, h)
        self.layers = nn.ModuleList([TypedAttention(h, heads, et) for _ in range(layers)])
        self.phi = nn.Linear(h, h)
        if readout == 'spot':
            self.head = nn.Sequential(nn.Linear(4 * h + 5, 2 * h), nn.GELU(), nn.Linear(2 * h, h), nn.GELU(),
                                      nn.Linear(h, NOUT + 2))
        else:
            self.head = nn.Sequential(nn.Linear(6 * h + 2, 2 * h), nn.GELU(), nn.Linear(2 * h, h), nn.GELU(),
                                      nn.Linear(h, NOUT + 2))

    def prep_edges(self, g):
        E = {}
        net = g['e_net']
        E['net'] = (net, None)
        E['rnet'] = (net.flip(0), None)
        if self.cfg['use_prox']:
            E['prox'] = (g['e_prox'], edge_attr('prox', g['a_prox']))
        if self.cfg['use_corr']:
            E['corr'] = (g['e_corr'], edge_attr('corr', g['a_corr']))
        if not self.cfg['typed']:
            es, aa = [], []
            for t, (e, a) in E.items():
                es.append(e)
                z = torch.zeros(e.shape[1], 6)
                if t == 'prox':
                    z[:, :2] = a
                elif t == 'corr':
                    z[:, 2:] = a[:, :4] if a.shape[1] >= 4 else F.pad(a, (0, 4 - a.shape[1]))
                aa.append(z)
            E = {'all': (torch.cat(es, 1), torch.cat(aa, 0))}
        return E

    def encode(self, g, gates=None):
        x = g['x']
        if not self.cfg['use_pos']:
            x = x.clone()
            x[:, 31:33] = 0
        h = self.inp(x)
        E = self.prep_edges(g)
        for l in self.layers:
            h = l(h, E, gates)
        return h

    def readout(self, g, h, threats, spots):
        """Vectorised over threats. threats: list of (r, m); spots: list of [G, 2, 2] tensors
        (single-spot threats repeat the centre). Returns P [T,4], p_cm [T], p_any [T], per-spot P [T,G,4]."""
        z = self.phi(h)
        own = g['owner']
        T = len(threats)
        if self.cfg['readout'] == 'spot':
            G = spots[0].shape[0]
            sp = torch.stack(spots)                                   # [T,G,2,2]
            d = torch.cdist(sp.reshape(-1, 2), g['pos']).view(T * G, 2, -1).min(1).values
            rv = torch.tensor([t[0] for t in threats]).repeat_interleave(G)[:, None]
            mv = torch.tensor([t[1] - 1.0 for t in threats]).repeat_interleave(G)[:, None]
            w = torch.sigmoid((rv - d) / TAU)                         # [T*G, n]
            feats, cnts = [], []
            for o in range(3):
                mk = own == o
                wo = w[:, mk]
                c = wo.sum(1, keepdim=True)
                feats.append(wo @ z[mk] / (c + 1e-3))
                cnts.append(torch.log1p(c))
            f = torch.cat(feats + [feats[0] * feats[1]] + cnts + [rv, mv], 1)
            lo = self.head(f).view(T, G, -1)
            p = torch.softmax(lo[..., :NOUT], -1)
            return p.mean(1), torch.sigmoid(lo[..., NOUT]).mean(1), torch.sigmoid(lo[..., NOUT + 1]).mean(1), p
        pools = []
        for o in range(3):
            mk = own == o
            pools += [z[mk].mean(0), z[mk].max(0).values]
        pooled = torch.cat(pools)[None].expand(T, -1)
        tv = torch.tensor([[t[0], t[1] - 1.0] for t in threats])
        lo = self.head(torch.cat([pooled, tv], 1))
        return torch.softmax(lo[:, :NOUT], 1), torch.sigmoid(lo[:, NOUT]), torch.sigmoid(lo[:, NOUT + 1]), None


class GraphBaseline(nn.Module):
    """Netlist-only GNN in the style of earlier fault-vulnerability predictors (GCN/GAT/GIN),
    global pooling and the threat parameters concatenated before the head."""

    def __init__(self, kind='gcn', h=64, layers=3):
        super().__init__()
        self.kind = kind
        keep = [3, 4] + list(range(5, 22)) + list(range(22, 29)) + [29, 30, 34]
        self.keep = keep
        self.inp = nn.Linear(len(keep), h)
        self.ws = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        if kind == 'gat':
            self.att = nn.ModuleList([nn.Linear(2 * h, 1) for _ in range(layers)])
        if kind == 'gin':
            self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)) for _ in range(layers)])
            self.eps = nn.Parameter(torch.zeros(layers))
        self.head = nn.Sequential(nn.Linear(2 * h + 2, h), nn.ReLU(), nn.Linear(h, NOUT + 2))

    def encode(self, g, gates=None):
        x = self.inp(g['x'][:, self.keep])
        e = g['e_net']
        s = torch.cat([e[0], e[1]])
        d = torch.cat([e[1], e[0]])
        n = x.shape[0]
        deg = torch.ones(n).index_add_(0, d, torch.ones(d.shape[0]))
        for i, W in enumerate(self.ws):
            if self.kind == 'gcn':
                nrm = (deg[s] * deg[d]).rsqrt()[:, None]
                agg = torch.zeros_like(x).index_add_(0, d, nrm * x[s]) + x / deg[:, None]
                x = F.relu(W(agg))
            elif self.kind == 'gat':
                hx = W(x)
                sc = F.leaky_relu(self.att[i](torch.cat([hx[d], hx[s]], 1)), 0.2)
                al = scatter_softmax(sc, d, n)
                x = F.elu(torch.zeros_like(hx).index_add_(0, d, al * hx[s]) + hx)
            else:
                agg = torch.zeros_like(x).index_add_(0, d, x[s])
                x = self.mlps[i]((1 + self.eps[i]) * x + agg)
        return x

    def readout(self, g, h, threats, spots):
        pooled = torch.cat([h.mean(0), h.max(0).values])[None].expand(len(threats), -1)
        tv = torch.tensor([[t[0], t[1] - 1.0] for t in threats])
        lo = self.head(torch.cat([pooled, tv], 1))
        return torch.softmax(lo[:, :NOUT], 1), torch.sigmoid(lo[:, NOUT]), torch.sigmoid(lo[:, NOUT + 1]), None


def sample_spots(g, threats, G, rng):
    W, H, off = g['W'], g['H'], g['offB']
    res = []
    for r, m in threats:
        if m == 1:
            cx = rng.uniform(-0.5, W - 0.5, G)
            cy = rng.uniform(-0.5, H - 0.5, G)
            sp = np.stack([np.stack([cx, cy], 1)] * 2, 1)
        else:
            cx = rng.uniform(-0.5, W - 0.5 - off, G)
            cy = rng.uniform(-0.5, H - 0.5, G)
            sp = np.stack([np.stack([cx, cy], 1), np.stack([cx + off, cy], 1)], 1)
        res.append(torch.tensor(sp, dtype=torch.float32))
    return res


def stratified_spots(g, threats, G, rng):
    """Jittered-grid spot centres (lower variance than i.i.d. sampling)."""
    W, H, off = g['W'], g['H'], g['offB']
    s = int(np.sqrt(G))
    res = []
    for r, m in threats:
        span = (W - off) if m == 2 else W
        u = (np.arange(s)[:, None] + rng.uniform(0, 1, (s, s))) / s
        v = (np.arange(s)[None, :] + rng.uniform(0, 1, (s, s))) / s
        cx = -0.5 + u.ravel() * span
        cy = -0.5 + v.ravel() * H
        if m == 1:
            sp = np.stack([np.stack([cx, cy], 1)] * 2, 1)
        else:
            sp = np.stack([np.stack([cx, cy], 1), np.stack([cx + off, cy], 1)], 1)
        res.append(torch.tensor(sp, dtype=torch.float32))
    return res


def loss_fn(P, pcm, pany, lab, lam_c=1.0):
    """P [T,4]; lab: dict tensors counts [T,4], n [T], cm [T], anyw [T]."""
    c, n = lab['counts'], lab['n']
    eps = 1e-6
    nll = -(c * torch.log(P + eps)).sum(1) / n
    nll_cm = -(lab['cm'] * torch.log(pcm + eps) + (n - lab['cm']) * torch.log(1 - pcm + eps)) / n
    nll_an = -(lab['anyw'] * torch.log(pany + eps) + (n - lab['anyw']) * torch.log(1 - pany + eps)) / n
    rc = c[:, 2] / n
    pc = P[:, 2]
    dev = 2 * (rc * torch.log((rc + eps) / (pc + eps)) + (1 - rc) * torch.log((1 - rc + eps) / (1 - pc + eps)))
    dev = dev / (rc + 0.005)
    return (nll + 0.5 * (nll_cm + nll_an) + lam_c * dev).mean()


def collate(gs):
    """Merge several graphs into one disjoint graph for the encoder."""
    xs, pos, own, en, ep, ap, ec, ac, ptr = [], [], [], [], [], [], [], [], [0]
    for g in gs:
        o = ptr[-1]
        xs.append(g['x']); pos.append(g['pos']); own.append(g['owner'])
        en.append(g['e_net'] + o); ep.append(g['e_prox'] + o); ap.append(g['a_prox'])
        ec.append(g['e_corr'] + o); ac.append(g['a_corr'])
        ptr.append(o + g['x'].shape[0])
    return dict(x=torch.cat(xs), pos=torch.cat(pos), owner=torch.cat(own), e_net=torch.cat(en, 1),
                e_prox=torch.cat(ep, 1), a_prox=torch.cat(ap), e_corr=torch.cat(ec, 1), a_corr=torch.cat(ac),
                ptr=ptr)
