"""E1/E2: cross-kernel prediction experiments (grouped folds, two kernels held out per fold)."""
import sys, os, json, time, pickle
import numpy as np
import torch
from train import load, fit, predict, truth, FOLDS, KERNEL_ORDER, THREATS, split_val

SPECS = {
    'asunder': dict(kind='asunder', h=32),
    'no_corr': dict(kind='asunder', h=32, use_corr=False),
    'no_prox': dict(kind='asunder', h=32, use_prox=False),
    'no_pos': dict(kind='asunder', h=32, use_pos=False, use_prox=False),
    'untyped': dict(kind='asunder', h=32, typed=False),
    'pool_readout': dict(kind='asunder', h=32, readout='pool'),
    'gcn': dict(kind='gcn'), 'gat': dict(kind='gat'), 'gin': dict(kind='gin'),
}
N_TRAIN_PER_KERNEL = int(os.environ.get('ASUNDER_NTRAIN', 180))


def train_set(fold, seed):
    held = FOLDS[fold]
    rng = np.random.default_rng(1000 + fold)
    tr = []
    for k in KERNEL_ORDER:
        if k in held:
            continue
        gs = load([k])
        idx = rng.permutation(len(gs))[:N_TRAIN_PER_KERNEL]
        tr += [gs[i] for i in idx]
    return split_val(tr, 0.12, seed=fold)


def run(name, fold, seed, epochs=None):
    epochs = epochs or int(os.environ.get('ASUNDER_EPOCHS', 20))
    torch.set_num_threads(1)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    out = f'results/pred_{name}_f{fold}_s{seed}.npz'
    if os.path.exists(out):
        return
    tr, va = train_set(fold, seed)
    t0 = time.time()
    model, hist = fit(SPECS[name], tr, va, epochs=epochs, seed=seed, patience=6)
    ttrain = time.time() - t0
    torch.save(model.state_dict(), f'models/{name}_f{fold}_s{seed}.pt')
    te = load(list(FOLDS[fold]))
    t0 = time.time()
    P, CM, AN = predict(model, te)
    tinf = (time.time() - t0) / len(te)
    R, CMI = truth(te)
    ker = np.array([g['meta']['kernel'] for g in te])
    np.savez(out, P=P, CM=CM, AN=AN, R=R, CMI=CMI, ker=ker, idx=np.array([g['meta']['idx'] for g in te]),
             ttrain=ttrain, tinf=tinf, hist=np.array(hist), nparam=sum(p.numel() for p in model.parameters()))
    print(name, fold, seed, f'train {ttrain:.0f}s inf {tinf * 1000:.1f}ms', flush=True)


if __name__ == '__main__':
    jobs = [j.split(':') for j in sys.argv[1:]]
    for name, fold, seed in jobs:
        run(name, int(fold), int(seed))
