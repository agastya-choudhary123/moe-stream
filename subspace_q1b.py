#!/usr/bin/env python3
"""
Q1b + Q3: what the projection does to the actual expert output, and whether the
factors survive being stored in 4 bits.

Variance retention is a proxy. The quantity that matters is the error in the
vector the residual stream receives, after a clamped SwiGLU that is not linear
and after the top-4 weighted sum. So this runs the real thing: real 4-bit
experts read out of experts.bin, real routing, real held-out activations.

    y_true   the expert block on the unmodified activation
    y_proj   gate/up on  mu + Q Q^T (x - mu).  Exactly equal to storing the
             factors W Q and computing z = Q^T (x - mu), since
             W (Q Q^T u) = (W Q)(Q^T u).  This is the bf16-factor case: bf16
             factors are *more* precise than today's weights, which are 4-bit,
             so the only new error is the projection itself.
    y_q4     the same with the factors W Q actually quantized to 4 bits --
             double quantization on top of the model's existing 4-bit weights.
             Prefixes are exact here: group_size 64 divides every k tested and
             groups run along k, so quantizing W Q once and truncating equals
             truncating and then quantizing.

down_proj is never projected: its input is the post-SwiGLU intermediate, which
is expert-specific, so no basis can be shared across experts.

Reference scale, from HANDOFF.md: two correct implementations of a layer
disagree at rel 2.0e-3; requantizing 4-bit weights to 3-bit, which visibly
damages the model, measured rel 2.6e-1.

  LAYERS=0,9,18,27,35 N=512 python3 subspace_q1b.py
"""

import json
import os
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.gpt_oss import swiglu

HERE = os.path.dirname(os.path.abspath(__file__))
ACTS = os.path.join(HERE, "acts")
MODEL = os.path.join(HERE, "model-120b")
KS = [128, 192, 256, 320, 384, 448, 576, 704, 960, 1408]
GS, BITS = 64, 4


class Experts:
    """Random access to one expert's dequantized weights in experts.bin."""

    def __init__(self):
        idx = json.load(open(os.path.join(MODEL, "experts_index.json")))
        self.blob = idx["blob_bytes"]
        self.n_experts = idx["n_experts"]
        self.comp = {f"{c['proj']}.{c['part']}": c for c in idx["components"]}
        self.fd = os.open(os.path.join(MODEL, "experts.bin"), os.O_RDONLY)

    def _part(self, base, name):
        c = self.comp[name]
        raw = os.pread(self.fd, c["nbytes"], base + c["offset"])
        if c["dtype"] == "U32":
            return mx.array(np.frombuffer(raw, dtype=np.uint32).copy()
                            ).reshape(c["shape"])
        a = mx.view(mx.array(np.frombuffer(raw, dtype=np.uint16).copy()),
                    mx.bfloat16)
        return a.reshape(c["shape"])

    def get(self, layer, e):
        """(W, bias) per projection; W dequantized to fp32, shaped [out, in]."""
        base = (layer * self.n_experts + e) * self.blob
        out = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            W = mx.dequantize(self._part(base, f"{p}.weight"),
                              self._part(base, f"{p}.scales"),
                              self._part(base, f"{p}.biases"),
                              group_size=GS, bits=BITS).astype(mx.float32)
            out[p] = (W, self._part(base, f"{p}.bias").astype(mx.float32))
        return out


def router_logits(layer, X):
    w = mx.load(os.path.join(MODEL, "nonexpert.safetensors"))
    p = f"model.layers.{layer}.mlp.router."
    W = mx.dequantize(w[p + "weight"], w[p + "scales"], w[p + "biases"],
                      group_size=GS, bits=BITS).astype(mx.float32)
    return X @ W.T + w[p + "bias"].astype(mx.float32)


def qdq(A):
    w, s, b = mx.quantize(A, group_size=GS, bits=BITS)
    return mx.dequantize(w, s, b, group_size=GS, bits=BITS).astype(mx.float32)


def main():
    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    n_per = max(w["window"] for w in wins) + 1
    layers = [int(s) for s in os.environ.get("LAYERS", "0,9,18,27,35").split(",")]
    N = int(os.environ.get("N", "512"))
    kq4 = int(os.environ.get("KQ4", "576"))     # largest k evaluated with 4-bit
    ks_q4 = [k for k in KS if k <= kq4]
    rng = np.random.default_rng(0)

    b = np.load(f"{ACTS}/basis.npz")
    test = [(w["lo"], w["hi"]) for w in wins if w["window"] == n_per - 1]
    ex = Experts()
    routes_all = np.load(f"{ACTS}/routes.npy", mmap_mode="r")
    results = {}

    for L in layers:
        t0 = time.perf_counter()
        Xall = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        te_rows = np.concatenate([np.arange(a, b_) for a, b_ in test])
        pick = np.sort(rng.choice(te_rows, N, replace=False))
        X = mx.array(np.array(Xall[pick]))
        mu = mx.array(b["mu"][L])
        V = mx.array(np.ascontiguousarray(b["V"][L]))

        g = router_logits(L, X)
        inds = mx.argpartition(g, kth=-4, axis=-1)[:, -4:]
        sc = mx.softmax(mx.take_along_axis(g, inds, axis=-1), axis=-1, precise=True)
        mx.eval(inds, sc)
        inds_np = np.array(inds)
        rec = np.array(routes_all[L][pick])
        agree = float(np.mean([len(set(a) & set(c)) / 4
                               for a, c in zip(inds_np, rec)]))

        U = X - mu
        XP = {k: mu + (U @ V[:, :k]) @ V[:, :k].T for k in KS}
        Z = {k: U @ V[:, :k] for k in ks_q4}
        mx.eval(*XP.values(), *Z.values())

        y_true = mx.zeros((N, 2880), dtype=mx.float32)
        y_proj = {k: mx.zeros((N, 2880), dtype=mx.float32) for k in KS}
        y_q4 = {k: mx.zeros((N, 2880), dtype=mx.float32) for k in ks_q4}

        for e in range(ex.n_experts):
            hit = np.nonzero(inds_np == e)
            if hit[0].size == 0:
                continue
            rows = mx.array(hit[0].astype(np.int32))
            wgt = sc[rows, mx.array(hit[1].astype(np.int32))][:, None]
            W = ex.get(L, e)
            Wg, bg = W["gate_proj"]
            Wu, bu = W["up_proj"]
            Wd, bd = W["down_proj"]

            def block(gate, up):
                return (swiglu(up, gate) @ Wd.T + bd) * wgt

            y_true = y_true.at[rows].add(
                block(X[rows] @ Wg.T + bg, X[rows] @ Wu.T + bu))
            for k in KS:
                xp = XP[k][rows]
                y_proj[k] = y_proj[k].at[rows].add(
                    block(xp @ Wg.T + bg, xp @ Wu.T + bu))

            # the compressed store: factors W Q in 4 bits, plus W mu folded into
            # the bias every projection already carries
            Fg, Fu = qdq(Wg @ V[:, :kq4]), qdq(Wu @ V[:, :kq4])
            og, ou = bg + mu @ Wg.T, bu + mu @ Wu.T
            for k in ks_q4:
                z = Z[k][rows]
                y_q4[k] = y_q4[k].at[rows].add(
                    block(z @ Fg[:, :k].T + og, z @ Fu[:, :k].T + ou))
            mx.eval(y_true, *y_proj.values(), *y_q4.values())

        nrm = float(mx.sqrt(mx.sum(y_true ** 2)))

        def err(y):
            d = float(mx.sqrt(mx.sum((y - y_true) ** 2))) / nrm
            cos = float(mx.sum(y * y_true) /
                        (mx.sqrt(mx.sum(y * y)) * mx.sqrt(mx.sum(y_true ** 2))))
            return d, cos

        r = dict(route_agree=agree, n=N,
                 proj={k: err(y_proj[k]) for k in KS},
                 q4={k: err(y_q4[k]) for k in ks_q4})
        results[L] = r
        print(f"L{L:02d}  {time.perf_counter()-t0:5.1f}s  routing reproduced "
              f"{agree*100:.1f}%")
        for k in KS:
            q = (f"4-bit factors rel {r['q4'][k][0]:.4f} cos {r['q4'][k][1]:.5f}"
                 if k in ks_q4 else "")
            print(f"    k={k:<5} bf16 factors rel {r['proj'][k][0]:.4f} "
                  f"cos {r['proj'][k][1]:.5f}   {q}", flush=True)
        json.dump({str(kk): v for kk, v in results.items()},
                  open(f"{ACTS}/q1b_output_error.json", "w"), indent=1)


if __name__ == "__main__":
    main()
