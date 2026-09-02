#!/usr/bin/env python3
"""
The subspace floor, re-priced in KL instead of in activation energy.

`subspace_floor.py` asked: what is the smallest achievable ||W (I - M) u||?
That is the wrong question if most of the activation energy is causally inert,
which is what the J-space result suggests. The right question is what the
projection costs the model's *predictions*, and to second order that is

    dKL  ~  1/2 * E[ du^T F du ],    du = (I - P) u,   F = E[g g^T]

with g = d NLL / d (expert-path input), measured by `jspace_fisher.py`. So the
metric on the error changes from the identity to F, and minimizing
`tr((I-P) C (I-P) F)` over rank-k P is the same reduced-rank regression as
before with G := F. One substitution, a completely different objective.

Two things are reported:

  validation -- the KL this predicts for the PCA basis at k=320, against the
  4.34 nats that `subspace_q2.py` actually measured end to end. A second-order
  predictor that cannot reproduce a measured number does not get to price
  anything.

  headroom -- the KL of the Fisher-optimal rank-k basis. If that is small where
  the PCA basis was catastrophic, the activation-subspace idea reopens and the
  next step is to build that basis and run it end to end.

  python3 jspace_floor.py
"""

import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
KS = [64, 128, 200, 320, 448, 576, 720, 960, 1440, 1920]


def sqrtm_psd(A):
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def main():
    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    n_per = max(w["window"] for w in wins) + 1
    b = np.load(f"{ACTS}/basis.npz")
    fj = json.load(open(f"{ACTS}/jspace_fisher.json"))
    out = {}

    for Ls in fj:
        L = int(Ls)
        n = fj[Ls]["n"]
        F = np.load(f"{ACTS}/fisher_L{L:02d}.npy").astype(np.float64) / n
        # A Fisher estimated from n samples has rank <= n out of 2880, so an
        # "optimal" basis chosen against it will happily discard every direction
        # the estimate never saw. Choosing on one half and pricing on the other
        # is the only version of this number that means anything.
        Fe = np.load(f"{ACTS}/fisher_even_L{L:02d}.npy").astype(np.float64)
        Fo = np.load(f"{ACTS}/fisher_odd_L{L:02d}.npy").astype(np.float64)
        Fe /= max(Fe.trace(), 1e-30) / F.trace()
        Fo /= max(Fo.trace(), 1e-30) / F.trace()
        X = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        te = np.concatenate([X[w["lo"]:w["hi"]] for w in wins
                             if w["window"] == n_per - 1])
        mu = mx.array(b["mu"][L])
        U = mx.array(te) - mu
        C = np.array((U.T @ U) / U.shape[0], copy=True).astype(np.float64)

        # cost of a projection P, in nats: 1/2 tr((I-P) C (I-P) F)
        total = 0.5 * float(np.trace(C @ F))

        # the PCA basis actually used end to end
        V = np.ascontiguousarray(b["V"][L]).astype(np.float64)
        pca = {}
        for k in KS:
            if k > V.shape[1]:
                continue
            Q = V[:, :k]
            R = np.eye(2880) - Q @ Q.T
            pca[k] = 0.5 * float(np.trace(R @ C @ R @ F))

        # the Fisher-optimal rank-k map, by reduced-rank regression
        def optimal(Ffit, Fscore):
            A = sqrtm_psd(Ffit)
            B = sqrtm_psd(C)
            Uu, ss, Vt = np.linalg.svd(A @ B)
            # M_k = A^-1 [A B]_k B^-1; score its residual under Fscore
            Ai = np.linalg.pinv(A, rcond=1e-10)
            Bi = np.linalg.pinv(B, rcond=1e-10)
            res = {}
            for k in KS:
                Mk = Ai @ (Uu[:, :k] * ss[:k]) @ Vt[:k] @ Bi
                R = np.eye(2880) - Mk
                res[k] = 0.5 * float(np.trace(R @ C @ R.T @ Fscore))
            return res

        opt = optimal(F, F)                    # in-sample: chooses and prices
        opt_ho = optimal(Fe, Fo)               # held out: the honest one

        out[L] = dict(total_nats=total, pca=pca, optimal=opt,
                      optimal_heldout=opt_ho)
        print(f"L{L:02d}  full projection away would cost {total:8.3f} nats")
        print(f"      {'k':>5} {'PCA basis':>14} {'Fisher-opt':>13} "
              f"{'Fisher-opt held out':>21}")
        for k in KS:
            if k in pca:
                print(f"      {k:>5} {pca[k]:11.4f} nat {opt[k]:10.4f} nat "
                      f"{opt_ho[k]:16.4f} nat")
        json.dump({str(k): v for k, v in out.items()},
                  open(f"{ACTS}/jspace_floor.json", "w"), indent=1)

    print("\nvalidation: subspace_q2.py measured KL 4.34 nats end to end for the "
          "PCA basis at k=320 (mean over 8 held-out windows, all 36 layers "
          "projected at once).")


if __name__ == "__main__":
    main()
