#!/usr/bin/env python3
"""
Fit the per-layer activation basis on the TRAIN windows only.

One basis per layer, shared by all 128 experts in that layer -- which is the
whole reason the idea is cheap. Stored as the top KMAX eigenvectors so any
k <= KMAX is a prefix, plus the mean, which the engine folds into the per-
projection bias for free.

  KMAX=960 python3 fit_basis.py    ->  acts/basis.npz
"""

import json
import os

import mlx.core as mx
import numpy as np

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")


def main():
    kmax = int(os.environ.get("KMAX", "960"))
    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    n_per = max(w["window"] for w in wins) + 1
    train = [(w["lo"], w["hi"]) for w in wins if w["window"] < n_per - 1]
    L = man["n_layers"]
    V = np.zeros((L, man["d"], kmax), dtype=np.float32)
    MU = np.zeros((L, man["d"]), dtype=np.float32)
    for l in range(L):
        X = np.load(f"{ACTS}/x_layer{l:02d}.npy", mmap_mode="r")
        Xtr = np.concatenate([X[a:b] for a, b in train])
        mu = Xtr.mean(axis=0)
        A = mx.array(Xtr) - mx.array(mu)
        C = A.T @ A
        mx.eval(C)
        w_, v = np.linalg.eigh(np.array(C, copy=True).astype(np.float64))
        V[l] = np.ascontiguousarray(v[:, ::-1][:, :kmax]).astype(np.float32)
        MU[l] = mu
        print(f"  L{l:02d} fitted on {Xtr.shape[0]:,} rows", flush=True)
    np.savez(f"{ACTS}/basis.npz", V=V, mu=MU, kmax=kmax,
             n_train=sum(b - a for a, b in train))
    print(f"wrote {ACTS}/basis.npz  {V.nbytes/2**20:.0f} MiB")


if __name__ == "__main__":
    main()
