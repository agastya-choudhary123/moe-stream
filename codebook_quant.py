#!/usr/bin/env python3
"""
Codebook (vector) quantization of gpt-oss-120b experts -- does it beat affine?

Bytes per token is the only remaining lever on the 120b (io_saturation.py: the
engine runs at 93% of its achievable byte floor, the SSD at 94% of its rate).
Every skew-exploiting scheme is dead because the load-balancing loss flattened
both expert usage (Gini 0.262) and the gate weights (0.35/0.26/0.21/0.18, see
gate_skew.py). What load balancing does NOT touch is structure *inside* an
expert, which is what vector quantization exploits.

Affine 4-bit spends 4 bits per weight plus a bf16 scale and bias per group of
64 -- 4.5 bits/weight all in. Product quantization splits each group into
subvectors of width d and stores an index into a K-entry codebook, costing
log2(K)/d bits/weight plus one fp16 scale per group. The question is whether
that buys the same quality for fewer bits, because the codebook can place its
entries where the weights actually are instead of on a uniform lattice.

Note what is NOT assumed: no bf16 original (gpt-oss was released natively in
MXFP4 and no wider original exists anywhere), and no claim that some experts
matter more than others.

Ground truth is the dequantized 4-bit weight -- that IS the model. Error is
measured the way every other probe here measures it: the expert block's output
on HELD-OUT tokens routed to that expert, never on the tokens any codebook was
fit on.

  LAYERS=18 NEXP=4 python3 codebook_quant.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.gpt_oss import swiglu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_q1b import Experts, GS, BITS

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")

# (subvector width d, codebook size K). bits/weight = log2(K)/d + 16/GS.
CONFIGS = [
    (2, 64),                # 3.25 b/w
    (2, 128),               # 3.75
    (2, 256),               # 4.25
    (4, 4096),              # 3.25, wider subvector at the same rate
]
AFFINE = [3, 4]             # baselines; 4 is the model as it ships


# Rows per distance tile, chosen so the [chunk, K] distance block stays at
# TILE floats (32 MB fp32) whatever K is. The old fixed 1<<16 was written for
# K=64; at K=4096 it asks for a 1 GB block, and the [chunk, K, d] form it fed
# asked for 4.3 GB, which is what OOM-killed the machine.
TILE = 1 << 23


def _tile(K, cap=1 << 16):
    return max(1, min(cap, TILE // K))


def kmeans(V, K, iters=12, seed=0, sub=120_000):
    """Lloyd on a subsample. Returns [K, d] centroids."""
    rng = np.random.default_rng(seed)
    S = V if len(V) <= sub else V[rng.choice(len(V), sub, replace=False)]
    C = S[rng.choice(len(S), K, replace=False)].astype(np.float32).copy()
    d = S.shape[1]
    chunk = _tile(K)
    for _ in range(iters):
        idx = np.empty(len(S), dtype=np.int32)
        c2 = np.einsum("kj,kj->k", C, C)
        for s in range(0, len(S), chunk):
            ch = S[s:s + chunk].astype(np.float32, copy=False)
            # ||ch - C||^2 expanded; the ||ch||^2 term is constant along the
            # argmin axis, so it is dropped (same trick as assign()). Only a
            # [chunk, K] block is ever live, and it is updated in place.
            dist = ch @ C.T
            dist *= -2.0
            dist += c2
            idx[s:s + chunk] = dist.argmin(1)
        cnt = np.bincount(idx, minlength=K).astype(np.float32)
        for j in range(d):
            C[:, j] = np.bincount(idx, weights=S[:, j].astype(np.float64),
                                  minlength=K) / np.maximum(cnt, 1)
        empty = cnt == 0
        if empty.any():                      # respawn dead centroids
            C[empty] = S[rng.choice(len(S), int(empty.sum()), replace=False)]
    return C


def assign(V, C, chunk=None):
    """Nearest centroid for every row of V, on the GPU."""
    Cm = mx.array(C)
    c2 = mx.sum(Cm * Cm, axis=1)
    chunk = chunk or _tile(len(C), cap=1 << 18)
    out = []
    for s in range(0, len(V), chunk):
        ch = mx.array(V[s:s + chunk])
        out.append(mx.argmin(-2.0 * (ch @ Cm.T) + c2, axis=1))
        mx.eval(out[-1])        # force each tile to free before the next
    return mx.concatenate(out)


def pq(W, d, K, seed=0):
    """Product-quantize [out, in] fp32. Per-group max-abs scale, then VQ."""
    out, inp = W.shape
    G = inp // GS
    Wg = np.array(W, copy=False).reshape(out, G, GS)
    scale = np.abs(Wg).max(axis=2, keepdims=True)
    scale = np.maximum(scale, 1e-8).astype(np.float32)
    scale = scale.astype(np.float16).astype(np.float32)     # fp16 on disk
    V = (Wg / scale).reshape(-1, d).astype(np.float32)
    C = kmeans(V, K, seed=seed)
    idx = assign(V, C)
    Wq = mx.array(C)[idx].reshape(out, G, GS) * mx.array(scale)
    return Wq.reshape(out, inp)


def qdq(A, bits):
    w, s, b = mx.quantize(A, group_size=GS, bits=bits)
    return mx.dequantize(w, s, b, group_size=GS, bits=bits).astype(mx.float32)


def main():
    layers = [int(s) for s in os.environ.get("LAYERS", "18").split(",")]
    nexp = int(os.environ.get("NEXP", "4"))

    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    n_per = max(w["window"] for w in wins) + 1
    te_rows = np.concatenate([np.arange(w["lo"], w["hi"]) for w in wins
                              if w["window"] == n_per - 1])
    routes = np.load(f"{ACTS}/routes.npy", mmap_mode="r")
    ex = Experts()
    rng = np.random.default_rng(0)

    schemes = ([(f"affine {b}b", ("affine", b), b + 32.0 / GS) for b in AFFINE] +
               [(f"PQ d={d} K={K}", ("pq", (d, K)), np.log2(K) / d + 16.0 / GS)
                for d, K in CONFIGS])
    schemes.sort(key=lambda s: s[2])

    results = {}
    for L in layers:
        X = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        rt = np.array(routes[L])
        picks = rng.choice(ex.n_experts, nexp, replace=False)
        errs = {n: [] for n, _, _ in schemes}
        t0 = time.perf_counter()
        used = 0

        for e in picks:
            te_e = te_rows[np.any(rt[te_rows] == e, axis=1)]
            if len(te_e) < 32:
                continue
            used += 1
            W = ex.get(L, int(e))
            xs = mx.array(np.array(X[np.sort(te_e)]))

            def block(Wg_, bg_, Wu_, bu_, Wd_, bd_):
                h = swiglu(xs @ Wu_.T + bu_, xs @ Wg_.T + bg_)
                return h @ Wd_.T + bd_

            (Wg, bg), (Wu, bu), (Wd, bd) = (W["gate_proj"], W["up_proj"],
                                            W["down_proj"])
            y0 = block(Wg, bg, Wu, bu, Wd, bd)
            nrm = float(mx.sqrt(mx.sum(y0 ** 2)))

            for name, (kind, arg), _ in schemes:
                if kind == "affine":
                    q = lambda A: qdq(A, arg)
                else:
                    dd, KK = arg
                    q = lambda A: pq(A, dd, KK)
                y = block(q(Wg), bg, q(Wu), bu, q(Wd), bd)
                errs[name].append(float(mx.sqrt(mx.sum((y - y0) ** 2))) / nrm)

        print(f"\nL{L:02d}  {used} experts, {time.perf_counter()-t0:.0f}s"
              f"   (held-out output error vs the 4-bit model)")
        print(f"  {'scheme':<16} {'bits/wt':>8} {'vs 4.5b':>8} {'out err':>10}")
        base = 4.0 + 32.0 / GS
        for name, _, bpw in schemes:
            if not errs[name]:
                continue
            print(f"  {name:<16} {bpw:8.2f} {bpw/base:7.2f}x "
                  f"{np.mean(errs[name]):10.4f}")
        results[L] = {n: float(np.mean(v)) for n, v in errs.items() if v}

    json.dump(results, open("codebook_quant.json", "w"), indent=1)
    print("\nwrote codebook_quant.json")


if __name__ == "__main__":
    main()
