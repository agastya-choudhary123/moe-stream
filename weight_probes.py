#!/usr/bin/env python3
"""
Three low-level structure probes that nothing in this project has looked at.

Each is cheap, each is a different KIND of structure from the subspace work, and
each would shrink the blob by a different mechanism.

1. GATE/UP PAIRING. SwiGLU neuron j computes swish(gate_j . x) * (up_j . x + 1),
   so gate_j and up_j are two vectors attached to the same unit. If up_j were
   close to a rescaling of gate_j, you would store one vector and a scalar and
   halve gate+up -- two thirds of the blob. Nothing here has checked whether the
   two halves of a SwiGLU unit are related.

2. NIBBLE ENTROPY. The only LOSSLESS axis available. The store is 4-bit affine
   quantized, and if the 16 codes are not used uniformly then the file has real
   entropy below 4 bits per weight and is compressible with no quality cost at
   all. Also checks the scales and biases, which are bf16 and may be using far
   less than 16 bits of range.

3. PERMUTATION EQUIVALENCE. HANDOFF.md records that distinct experts are
   mutually orthogonal (cosine +0.002) and concludes there is no cross-expert
   redundancy. That conclusion is stronger than the evidence: cosine is not
   permutation invariant, and two experts that compute the same function with
   their hidden units in a different order would look exactly that orthogonal.
   Neuron alignment is a real phenomenon in the weight-symmetry literature. If
   experts were permutation-related you would store one expert plus a
   permutation (about 4 KB against 14 MB).

  LAYERS=18 python3 weight_probes.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_q1b import Experts

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model-120b")


def entropy(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    layers = [int(s) for s in os.environ.get("LAYERS", "18").split(",")]
    ex = Experts()
    idx = json.load(open(os.path.join(MODEL, "experts_index.json")))
    rng = np.random.default_rng(0)

    for L in layers:
        print(f"=== layer {L} " + "=" * 56)

        # ---- 1. gate/up pairing --------------------------------------------
        cos_pair, cos_rand, r1 = [], [], []
        for e in rng.choice(ex.n_experts, 8, replace=False):
            W = ex.get(L, int(e))
            G, U = W["gate_proj"][0], W["up_proj"][0]
            gn = G / mx.sqrt(mx.sum(G * G, axis=1, keepdims=True))
            un = U / mx.sqrt(mx.sum(U * U, axis=1, keepdims=True))
            c = mx.sum(gn * un, axis=1)                     # neuron j with its pair
            perm = np.random.default_rng(1).permutation(2880)
            c2 = mx.sum(gn * un[mx.array(perm)], axis=1)    # against a random other
            mx.eval(c, c2)
            cos_pair.append(float(mx.mean(mx.abs(c))))
            cos_rand.append(float(mx.mean(mx.abs(c2))))
            # best per-neuron rescaling up_j ~ a_j gate_j, residual energy
            a = mx.sum(G * U, axis=1) / mx.sum(G * G, axis=1)
            res = U - a[:, None] * G
            r1.append(float(mx.sum(res * res) / mx.sum(U * U)))
        print(f"1. gate/up pairing: |cos| paired {np.mean(cos_pair):.4f} vs "
              f"shuffled {np.mean(cos_rand):.4f}")
        print(f"   best up_j = a_j * gate_j leaves "
              f"{np.mean(r1)*100:.1f}% of up's energy unexplained "
              f"(0% would halve gate+up)")

        # ---- 2. lossless headroom ------------------------------------------
        comps = {f"{c['proj']}.{c['part']}": c for c in idx["components"]}
        nib = np.zeros(16, dtype=np.int64)
        hi_scale = np.zeros(256, dtype=np.int64)
        lo_scale = np.zeros(256, dtype=np.int64)
        for e in rng.choice(ex.n_experts, 16, replace=False):
            base = (L * ex.n_experts + int(e)) * ex.blob
            c = comps["gate_proj.weight"]
            raw = np.frombuffer(os.pread(ex.fd, c["nbytes"], base + c["offset"]),
                                dtype=np.uint8)
            nib += np.bincount(raw & 0xF, minlength=16)
            nib += np.bincount(raw >> 4, minlength=16)
            c = comps["gate_proj.scales"]
            s = np.frombuffer(os.pread(ex.fd, c["nbytes"], base + c["offset"]),
                              dtype=np.uint8)
            hi_scale += np.bincount(s[1::2], minlength=256)   # bf16 high byte
            lo_scale += np.bincount(s[0::2], minlength=256)   # bf16 low byte
        h_nib = entropy(nib)
        print(f"2. lossless headroom: 4-bit codes carry {h_nib:.3f} bits of "
              f"entropy (max 4.000) -> {(1-h_nib/4)*100:.1f}% of the weight "
              f"bytes are redundant")
        print(f"   scale bf16: high byte {entropy(hi_scale):.2f} bits, low byte "
              f"{entropy(lo_scale):.2f} bits (max 8.00 each)")
        wshare = comps["gate_proj.weight"]["nbytes"] * 3 / ex.blob
        print(f"   weights are {wshare*100:.0f}% of the blob, so a perfect "
              f"entropy coder saves ~{(1-h_nib/4)*wshare*100:.1f}% of it")

        # ---- 3. permutation equivalence between experts ---------------------
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            print("3. permutation test skipped: scipy not available")
            continue
        a, b = [int(v) for v in rng.choice(ex.n_experts, 2, replace=False)]
        Wa = ex.get(L, a)
        Wb = ex.get(L, b)
        # a neuron's identity is its (gate, up) input pair
        A = mx.concatenate([Wa["gate_proj"][0], Wa["up_proj"][0]], axis=1)
        B = mx.concatenate([Wb["gate_proj"][0], Wb["up_proj"][0]], axis=1)
        An = A / mx.sqrt(mx.sum(A * A, axis=1, keepdims=True))
        Bn = B / mx.sqrt(mx.sum(B * B, axis=1, keepdims=True))
        S = np.array((An @ Bn.T).astype(mx.float32)).astype(np.float64)
        t0 = time.perf_counter()
        ri, ci = linear_sum_assignment(-S)
        matched = S[ri, ci]
        base_nrm = float(mx.sum(A * A))
        resid = float(mx.sum((A - B[mx.array(ci.astype(np.int32))]) ** 2)) / base_nrm
        print(f"3. permutation alignment ({time.perf_counter()-t0:.1f}s, "
              f"experts {a} and {b}):")
        print(f"   best-matched neuron cosine: mean {matched.mean():.4f}, "
              f"max {matched.max():.4f}")
        print(f"   ||W_a - P W_b||^2 / ||W_a||^2 = {resid:.4f} "
              f"(2.0 = unrelated, 0 = identical up to permutation)")


if __name__ == "__main__":
    main()
