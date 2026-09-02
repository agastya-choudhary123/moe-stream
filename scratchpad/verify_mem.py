#!/usr/bin/env python3
"""Peak-memory + correctness check for the codebook_quant.py distance fix.

1. brute-force [n, K, d] distances vs the expanded form, small case, exact
   assignment agreement.
2. pq() over every CONFIG on a real layer-18 expert, reporting peak RSS and
   peak MLX buffer use after each.
"""
import os
import resource
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
from codebook_quant import CONFIGS, assign, kmeans, pq
from subspace_q1b import Experts


def rss_gb():
    # darwin reports ru_maxrss in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def check_exact():
    rng = np.random.default_rng(0)
    for d, K in [(2, 64), (4, 4096)]:
        V = rng.standard_normal((5000, d)).astype(np.float32)
        C = kmeans(V, K, iters=3, sub=5000)
        got = np.array(assign(V, C))
        ref = ((V[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        # ties are possible; compare achieved distance, not index
        dg = ((V - C[got]) ** 2).sum(1)
        dr = ((V - C[ref]) ** 2).sum(1)
        ok = np.allclose(dg, dr, atol=1e-5)
        print(f"  d={d} K={K}: assignment matches brute force: {ok} "
              f"(idx equal {100*np.mean(got == ref):.2f}%)")
        assert ok


def main():
    print("exactness vs brute-force distances")
    check_exact()

    L = int(os.environ.get("L", "18"))
    ex = Experts()
    W = ex.get(L, 0)["gate_proj"][0]
    print(f"\npq() peak memory, layer {L} expert 0 gate_proj {tuple(W.shape)}")
    print(f"  {'config':<16} {'rows':>10} {'peak RSS GB':>12} {'peak MLX GB':>12}")
    print(f"  {'baseline':<16} {'':>10} {rss_gb():12.2f} "
          f"{mx.get_peak_memory()/1e9:12.2f}")
    for d, K in CONFIGS:
        mx.reset_peak_memory()
        Wq = pq(W, d, K)
        mx.eval(Wq)
        rel = float(mx.sqrt(mx.sum((Wq - W) ** 2) / mx.sum(W ** 2)))
        del Wq
        print(f"  d={d} K={K:<10} {W.size // d:10d} {rss_gb():12.2f} "
              f"{mx.get_peak_memory()/1e9:12.2f}   relerr {rel:.4f}")


if __name__ == "__main__":
    main()
