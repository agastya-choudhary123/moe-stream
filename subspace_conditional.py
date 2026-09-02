#!/usr/bin/env python3
"""
The two activation-subspace schemes the shared-basis floor does NOT cover.

`subspace_floor.py` settles one question completely: for a rank-k map shared by
all 128 experts in a layer, no basis beats ~0.42 relative error at k=320. That
is the scheme as originally proposed, and it is dead. It says nothing about:

  A. A PER-EXPERT basis. The floor used the marginal activation covariance, but
     routing partitions the activation space -- an expert only ever sees the
     tokens routed to it, and that conditional distribution can be much narrower
     than the pooled one. The per-genre numbers hint at exactly this: a basis fit
     on one genre retains 71-75% at k=320 where a cross-genre basis retains
     33-46%. Routing is a sharper conditioning than genre.

     It is not obviously more expensive, either, because Q is shared between gate
     and up (they read the same x). Per expert you store Q, W_g Q and W_u Q --
     three [2880, k] against two [2880, 2880]. At 4 bits that breaks even at
     k=1922, so k=320 would be 0.17x on gate+up.

  B. An OUTPUT-side basis on down_proj. Every expert's output lands in the same
     residual stream. If those outputs share a low-dimensional subspace, then
     W_down ~ Q_out (Q_out^T W_down) with Q_out shared per layer, storing
     [k, 2880] per expert. down_proj is a third of the blob and none of the
     input-side result applies to it.

The sample-count trap that produced the original 98% is avoided by construction:
every comparison here is at EQUAL sample count and evaluated on held-out tokens.
Test A in particular compares a conditional basis against a marginal basis fit on
the same number of samples, so neither can win on degrees of freedom.

  LAYERS=4,18,35 NEXP=24 python3 subspace_conditional.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.gpt_oss import swiglu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_q1b import Experts, router_logits

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
KS = [64, 128, 200, 256, 320, 448, 576, 720, 960]


def basis(X, mu, kmax):
    """Top-kmax eigenvectors of the centered second moment of X."""
    A = mx.array(X) - mu
    C = A.T @ A
    mx.eval(C)
    w, V = np.linalg.eigh(np.array(C, copy=True).astype(np.float64))
    return mx.array(np.ascontiguousarray(V[:, ::-1][:, :kmax]).astype(np.float32))


def retention(V, Xe, mu, ks):
    """Energy of held-out Xe retained by the top-k columns of V."""
    U = mx.array(Xe) - mu
    Z = U @ V
    mx.eval(Z)
    z = np.array(Z).astype(np.float64) ** 2
    tot = float(mx.sum(U * U))
    cum = np.cumsum(z.sum(axis=0))
    return {k: float(cum[k - 1] / tot) for k in ks if k <= V.shape[1]}


def main():
    layers = [int(s) for s in os.environ.get("LAYERS", "4,18,35").split(",")]
    nexp = int(os.environ.get("NEXP", "24"))
    kmax = max(KS)
    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    n_per = max(w["window"] for w in wins) + 1
    tr_rows = np.concatenate([np.arange(w["lo"], w["hi"]) for w in wins
                              if w["window"] < n_per - 1])
    te_rows = np.concatenate([np.arange(w["lo"], w["hi"]) for w in wins
                              if w["window"] == n_per - 1])
    routes = np.load(f"{ACTS}/routes.npy", mmap_mode="r")
    b = np.load(f"{ACTS}/basis.npz")
    ex = Experts()
    rng = np.random.default_rng(0)
    out = {}

    for L in layers:
        t0 = time.perf_counter()
        X = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        rt = np.array(routes[L])
        mu = mx.array(b["mu"][L])
        Vshared = mx.array(np.ascontiguousarray(b["V"][L][:, :kmax]))

        # ---- A. per-expert conditional basis, at equal sample count ----------
        picks = rng.choice(ex.n_experts, nexp, replace=False)
        cond, marg, shared, ns = [], [], [], []
        for e in picks:
            tr_e = tr_rows[np.any(rt[tr_rows] == e, axis=1)]
            te_e = te_rows[np.any(rt[te_rows] == e, axis=1)]
            if len(tr_e) < 64 or len(te_e) < 32:
                continue
            n = len(tr_e)
            Xtr_e = np.array(X[np.sort(tr_e)])
            Xte_e = np.array(X[np.sort(te_e)])
            # marginal basis at the SAME sample count -- the fair control
            same = np.sort(rng.choice(tr_rows, n, replace=False))
            Vc = basis(Xtr_e, mu, kmax)
            Vm = basis(np.array(X[same]), mu, kmax)
            cond.append(retention(Vc, Xte_e, mu, KS))
            marg.append(retention(Vm, Xte_e, mu, KS))
            shared.append(retention(Vshared, Xte_e, mu, KS))
            ns.append(n)
        agg = lambda rs: {k: float(np.mean([r[k] for r in rs])) for k in KS}
        A = dict(n_experts=len(ns), n_train_per_expert=float(np.mean(ns)),
                 conditional=agg(cond), marginal_same_n=agg(marg),
                 shared_full=agg(shared))

        # ---- B. output-side basis on down_proj -------------------------------
        # y_e = h @ W_down^T for the tokens actually routed to e, pooled over
        # experts: does one Q_out per layer span every expert's realized output?
        def outputs(rows_src, cap):
            Y = []
            for e in picks:
                r = rows_src[np.any(rt[rows_src] == e, axis=1)]
                if len(r) == 0:
                    continue
                r = np.sort(rng.choice(r, min(cap, len(r)), replace=False))
                xs = mx.array(np.array(X[r]))
                W = ex.get(L, int(e))
                Wg, bg = W["gate_proj"]
                Wu, bu = W["up_proj"]
                Wd, _ = W["down_proj"]
                h = swiglu(xs @ Wu.T + bu, xs @ Wg.T + bg)
                y = h @ Wd.T                    # bias stays exact, excluded
                mx.eval(y)
                Y.append(np.array(y))
            return np.concatenate(Y)

        Ytr = outputs(tr_rows, 96)
        Yte = outputs(te_rows, 48)
        zero = mx.zeros((2880,), dtype=mx.float32)
        Vy = basis(Ytr, zero, kmax)             # uncentered: the map is linear
        B = dict(n_train=int(Ytr.shape[0]), n_eval=int(Yte.shape[0]),
                 retention=retention(Vy, Yte, zero, KS))

        out[L] = dict(A=A, B=B)
        print(f"L{L:02d}  {time.perf_counter()-t0:5.1f}s  "
              f"{A['n_experts']} experts, {A['n_train_per_expert']:.0f} train "
              f"tokens each", flush=True)
        print("      A. input-side basis, retention on held-out tokens of that expert")
        print(f"      {'k':>5} {'per-expert':>11} {'marginal(=n)':>13} "
              f"{'shared(16k)':>12}")
        for k in KS:
            print(f"      {k:>5} {A['conditional'][k]*100:10.1f}% "
                  f"{A['marginal_same_n'][k]*100:12.1f}% "
                  f"{A['shared_full'][k]*100:11.1f}%")
        print("      B. output-side basis on down_proj, one Q_out for the layer")
        print("      " + "  ".join(f"k={k}:{B['retention'][k]*100:.1f}%"
                                   for k in (128, 320, 576, 960)))
        json.dump({str(k): v for k, v in out.items()},
                  open(f"{ACTS}/conditional.json", "w"), indent=1)


if __name__ == "__main__":
    main()
