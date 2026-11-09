"""Train the face-kind head (stratamodels.runtimes.facekind): logistic regression on CLIP embeddings of labelled face crops.

    .venv/bin/python scripts/face_kind_train.py --crops crops.npz [--key crops] [--bgr] --labels labels.json [--model ViT-B-32:laion2b_s34b_b79k]
        [--out ~/models/face-kind/head-v1.npz]

crops.npz holds uint8 (N, H, W, 3) face crops cut with the runtime's margin (0.3); labels.json is a list of N kinds, one per crop: real,
statue, picture, toy, animal, or other (not a face: a back of a head, a pattern). Several apps' labels can be joined: pass --crops and
--labels more than once. Prints the 5-fold real-or-not AUC and the share of real faces kept / others dropped at 0.5, then writes W, b,
classes. Labels are the apps' private pictures' crops: they stay on this machine, only the head (a few kB of weights) is written."""
import argparse, json, os, sys, types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from stratamodels.runtimes import clip


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit(E, Y, k, C):
    """Multinomial logistic regression with an L2 penalty 1/(2C) on the weights (as scikit-learn's C), by L-BFGS in torch: (W (k, D), b (k,))."""
    import torch
    X = torch.from_numpy(np.asarray(E, np.float32)).double(); t = torch.from_numpy(np.asarray(Y)); W = torch.zeros(k, X.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(X @ W.T + b, t, reduction='sum') + (W ** 2).sum() / (2 * C); loss.backward(); return loss
    opt.step(closure); return W.detach().numpy(), b.detach().numpy()


def auc(pos, score):
    """Area under the ROC curve (ties count half)."""
    from scipy.stats import rankdata
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1; return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--crops', action='append', required=True); ap.add_argument('--labels', action='append', required=True)
    ap.add_argument('--key'); ap.add_argument('--bgr', action='store_true'); ap.add_argument('--model', default='ViT-B-32:laion2b_s34b_b79k'); ap.add_argument('--C', type=float, default=2.0)
    ap.add_argument('--out', default=os.path.expanduser('~/models/face-kind/head-v1.npz')); ap.add_argument('--models-dir', default=os.path.expanduser('~/models')); a = ap.parse_args()
    if len(a.crops) != len(a.labels): sys.exit('give one --labels per --crops')
    X, y = [], []
    for c, l in zip(a.crops, a.labels):
        arr = clip._array(dict(path=c, key=a.key, bgr=a.bgr)); lab = json.load(open(l))
        if len(lab) != len(arr): sys.exit(f'{l}: {len(lab)} labels for {len(arr)} crops')
        X.append(arr); y += lab
    h = clip.load(types.SimpleNamespace(name=a.model, options={}), a.models_dir)
    E = np.concatenate([clip._encode_images(h, [np.ascontiguousarray(f) for f in arr], 'crop') for arr in X]); y = np.array(y)          # each set's crops may be another size: embed set by set, join the vectors
    if 'real' not in set(y): sys.exit("no 'real' labels")
    classes = sorted(set(y)); Y = np.array([classes.index(v) for v in y]); r = y == 'real'
    folds = np.zeros(len(y), int); rng = np.random.default_rng(0)
    for c in range(len(classes)):
        ix = rng.permutation(np.where(Y == c)[0]); folds[ix] = np.arange(len(ix)) % 5
    p = np.zeros((len(y), len(classes)))
    for f in range(5): W, b = fit(E[folds != f], Y[folds != f], len(classes), a.C); p[folds == f] = softmax(E[folds == f] @ W.T + b)
    pr = p[:, classes.index('real')]
    print(f'{len(y)} crops ({r.sum()} real); 5-fold real-or-not AUC {auc(r, pr):.3f}; at 0.5 keeps {np.mean(pr[r] >= .5):.1%} of real faces, drops {np.mean(pr[~r] < .5):.1%} of the rest; kind accuracy {np.mean(p.argmax(1) == Y):.1%}')
    W, b = fit(E, Y, len(classes), a.C); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, W=W.astype(np.float32), b=b.astype(np.float32), classes=np.array(classes), model=a.model, n=len(y)); print('wrote', a.out)


if __name__ == '__main__':
    main()
