#!/usr/bin/env python3
"""
Q1: where does k saturate, on a diverse corpus, with the basis held out?

The number this replaces -- "90% at k=200, 98% at k=320" -- was fitted on 420
samples. A rank-k subspace fitted on N samples explains the training set almost
perfectly once k approaches N, so at k=320/N=420 that measurement was mostly
reporting its own degrees of freedom. Two things fix it: many more samples than
k, and evaluating on documents the basis never saw.

Three splits, in increasing order of how much they ask of the idea:

  self   fit and evaluate on the same rows        (the optimistic number)
  doc    fit on 2 windows/genre, evaluate on the third
  genre  fit on 7 genres, evaluate on the held-out 8th

and a convergence sweep (fit on 1/2/4/8/16 windows, evaluate on the same
held-out rows) that shows directly how much of the old 98% was sample count.

Retention is computed in closed form. For an orthonormal basis Q,

    sum_i ||Q^T (x_i - mu)||^2 / sum_i ||x_i - mu||^2  =  tr(Q^T C Q) / tr(C)

with C the (centered or uncentered) second-moment matrix of the *evaluation*
rows, so no data has to be projected: one 2880x2880 matrix per split.

Centering is free in the engine and so is on by default: y = W x =
W mu + (W Q)(Q^T (x - mu)), and W mu is a [2880] vector per expert that folds
into the `bias` component every gpt-oss projection already carries.

  python3 subspace_q1.py            # all layers, doc split
  LAYERS=0,9,18,27,35 GENRE=1 python3 subspace_q1.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
KS = [64, 128, 160, 200, 256, 320, 384, 448, 512, 576, 640, 720, 800, 960,
      1152, 1440, 1920]
TARGETS = [0.90, 0.95, 0.98, 0.99, 0.995]


def moments(X, center):
    """Return (C, n) with C the second-moment (or covariance) matrix, fp32 GPU."""
    A = mx.array(X)
    if center:
        A = A - A.mean(axis=0, keepdims=True)
    C = (A.T @ A)
    mx.eval(C)
    return np.array(C, copy=True), X.shape[0]


def retention(C_fit, C_eval, ks):
    """Eigenvectors of C_fit, energy of C_eval retained by the top-k of them."""
    w, V = np.linalg.eigh(C_fit.astype(np.float64))
    V = V[:, ::-1]                                  # descending eigenvalue order
    E = np.einsum("ij,jk,ki->i", V.T, C_eval.astype(np.float64), V)
    tot = np.trace(C_eval.astype(np.float64))
    cum = np.cumsum(E) / tot
    return {k: float(cum[k - 1]) for k in ks if k <= len(cum)}, cum


def k_for(cum, targets):
    out = {}
    for t in targets:
        idx = np.searchsorted(cum, t) + 1
        out[t] = int(idx) if idx <= len(cum) else None
    return out


def main():
    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    seq = man["seq"]
    genres = sorted({w["genre"] for w in wins})
    layers = [int(s) for s in os.environ.get(
        "LAYERS", ",".join(str(i) for i in range(man["n_layers"]))).split(",")]
    center = os.environ.get("CENTER", "1") == "1"
    do_genre = os.environ.get("GENRE", "1") == "1"

    rows_of = {}
    for w in wins:
        rows_of.setdefault((w["genre"], w["window"]), (w["lo"], w["hi"]))
    n_win_per_genre = max(w["window"] for w in wins) + 1
    print(f"{len(wins)} windows, {len(genres)} genres x {n_win_per_genre}, "
          f"{len(wins)*seq:,} rows/layer, center={center}")

    def sel(X, keys):
        return np.concatenate([X[lo:hi] for lo, hi in
                               (rows_of[k] for k in keys)], axis=0)

    train_keys = [(g, i) for g in genres for i in range(n_win_per_genre - 1)]
    test_keys = [(g, n_win_per_genre - 1) for g in genres]

    results = {}
    for L in layers:
        t0 = time.perf_counter()
        X = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        Xtr, Xte = sel(X, train_keys), sel(X, test_keys)
        Ctr, ntr = moments(Xtr, center)
        Cte, nte = moments(Xte, center)

        r = {}
        cur, cum = retention(Ctr, Cte, KS)
        r["doc"] = dict(ret=cur, k_at=k_for(cum, TARGETS), n_fit=ntr, n_eval=nte)
        cur, cum = retention(Cte, Cte, KS)
        r["self"] = dict(ret=cur, k_at=k_for(cum, TARGETS), n_fit=nte, n_eval=nte)

        # convergence: how much of a high retention number is just sample count
        conv = {}
        for nw in (1, 2, 4, 8, 16):
            keys = train_keys[:nw]
            Cs, ns = moments(sel(X, keys), center)
            c, cu = retention(Cs, Cte, KS)
            conv[ns] = dict(ret=c, k_at=k_for(cu, TARGETS))
        r["convergence"] = conv

        if do_genre:
            g_out = {}
            for g in genres:
                fit = [k for k in rows_of if k[0] != g]
                ev = [k for k in rows_of if k[0] == g]
                Cf, nf = moments(sel(X, fit), center)
                Ce, ne = moments(sel(X, ev), center)
                c, cu = retention(Cf, Ce, KS)
                cs, cus = retention(Ce, Ce, KS)
                g_out[g] = dict(ret=c, k_at=k_for(cu, TARGETS),
                                self_ret=cs, self_k_at=k_for(cus, TARGETS),
                                n_fit=nf, n_eval=ne)
            r["genre"] = g_out

        results[L] = r
        d = r["doc"]["ret"]
        print(f"L{L:02d} {time.perf_counter()-t0:5.1f}s  held-out doc: "
              f"k=200 {d.get(200,0)*100:5.2f}%  k=320 {d.get(320,0)*100:5.2f}%  "
              f"k=720 {d.get(720,0)*100:5.2f}%  |  k@95% "
              f"{r['doc']['k_at'][0.95]}  k@98% {r['doc']['k_at'][0.98]}  "
              f"k@99% {r['doc']['k_at'][0.99]}", flush=True)
        json.dump(dict(center=center, ks=KS, targets=TARGETS, genres=genres,
                       results={str(k): v for k, v in results.items()}),
                  open(f"{ACTS}/q1_retention{'_c' if center else ''}.json", "w"))


if __name__ == "__main__":
    main()
