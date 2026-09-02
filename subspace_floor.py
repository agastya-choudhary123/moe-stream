#!/usr/bin/env python3
"""
The floor: the best a rank-k shared basis can possibly do, for any choice of
basis.

PCA minimizes the error in x. What the engine cares about is the error in W x,
and those differ -- the weights may attenuate exactly the directions PCA throws
away. So "PCA is not good enough at k" is a weaker statement than "no shared
rank-k factorization is good enough at k", and only the second one closes the
question.

The second one is computable. Store W Q per expert and compute z = R^T u: the
expert block then applies W M u with M = Q R^T of rank k, and M is shared by
every expert in the layer. Minimizing

    sum_e E || W_e (I - M) u ||^2      over rank-k M

is reduced-rank regression. With G = sum_e W_e^T W_e and C = E[u u^T],
A = G^(1/2), B = C^(1/2), the error is || A (I - M) B ||_F, and the optimal M
is A^-1 [A B]_k B^-1, whose residual is the discarded singular values of A B:

    min error^2 (k) = sum_{i>k} sigma_i^2

That is a floor no basis of that rank can beat, PCA or otherwise. It is
reported here relative to the block's own pre-activation output energy, so it
is comparable to the end-to-end relative errors in subspace_q1b.py.

G is estimated from a sample of experts; expert usage on this model is close to
uniform (HANDOFF: Gini 0.262, no unused experts), so a uniform sample is
representative.

  LAYERS=18 NEXP=24 python3 subspace_floor.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_q1b import Experts

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
KS = [128, 200, 256, 320, 448, 576, 720, 960, 1440, 1920, 2400]


def main():
    layers = [int(s) for s in os.environ.get("LAYERS", "18").split(",")]
    nexp = int(os.environ.get("NEXP", "24"))
    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    n_per = max(w["window"] for w in wins) + 1
    ex = Experts()
    rng = np.random.default_rng(0)
    out = {}

    for L in layers:
        t0 = time.perf_counter()
        X = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        tr = np.concatenate([X[w["lo"]:w["hi"]] for w in wins
                             if w["window"] < n_per - 1])
        te = np.concatenate([X[w["lo"]:w["hi"]] for w in wins
                             if w["window"] == n_per - 1])
        mu = mx.array(tr.mean(axis=0))
        Utr = mx.array(tr) - mu
        Ute = mx.array(te) - mu
        C = (Utr.T @ Utr) / Utr.shape[0]
        mx.eval(C)

        picks = rng.choice(ex.n_experts, nexp, replace=False)
        G = mx.zeros((2880, 2880), dtype=mx.float32)
        sig = 0.0            # E ||W x + b||^2, the denominator that matters
        for e in picks:
            W = ex.get(L, int(e))
            for p in ("gate_proj", "up_proj"):
                Wp, bp = W[p]
                G = G + Wp.T @ Wp
                Y = mx.array(te) @ Wp.T + bp
                sig += float(mx.sum(Y * Y)) / te.shape[0]
            mx.eval(G)
        G = G / nexp
        sig /= nexp

        Cn = np.array(C, copy=True).astype(np.float64)
        Gn = np.array(G, copy=True).astype(np.float64)
        # symmetric square roots
        wc, Vc = np.linalg.eigh(Cn)
        B = (Vc * np.sqrt(np.clip(wc, 0, None))) @ Vc.T
        wg, Vg = np.linalg.eigh(Gn)
        A = (Vg * np.sqrt(np.clip(wg, 0, None))) @ Vg.T
        s = np.linalg.svd(A @ B, compute_uv=False)
        tail = np.concatenate([np.cumsum(s[::-1] ** 2)[::-1][1:], [0.0]])

        # the same quantities for the actual PCA basis, for the gap
        wcov, Vpca = np.linalg.eigh(np.array((Ute.T @ Ute) / Ute.shape[0],
                                             copy=True).astype(np.float64))
        r = dict(sig=sig, ks={})
        for k in KS:
            r["ks"][k] = dict(floor=float(np.sqrt(tail[k - 1] / sig)))
        out[L] = r
        print(f"L{L:02d}  {time.perf_counter()-t0:5.1f}s   "
              f"E||Wx+b||^2 = {sig:.4g}", flush=True)
        for k in KS:
            print(f"    k={k:<5} best possible rel error {r['ks'][k]['floor']:.4f}")
        json.dump({str(k): v for k, v in out.items()},
                  open(f"{ACTS}/floor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
