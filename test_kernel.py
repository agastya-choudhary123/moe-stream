#!/usr/bin/env python3
"""
Correctness and isolated timing for the fused MoE expert kernel.

Correctness is measured against an fp32 reference built by dequantizing the
same slots and running the MLP in full precision -- not against MLX's
gather_qmm path, which is itself bf16 and carries its own error. Both the
fused kernel and the MLX path are then scored against that reference, so the
question the test answers is "is the kernel at least as accurate as what it
replaces", not "does it match MLX bit for bit".
"""

import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PF_SLOTS", "16")
import engine_v3 as E
import kernels as K

D_MODEL, D_FF, K_EXPERTS = 2048, 768, 8


def load_pool(n_slots=16, layer=0, experts=range(8)):
    pool = E.ExpertPool(E.MODEL_DIR, n_slots=n_slots, n_workers=4)
    slots = pool.acquire(layer, list(experts))
    return pool, mx.array(np.asarray(slots, dtype=np.uint32))


def mlx_path(pool, idx, x, scores):
    """Exactly the expert block engine_v3 runs today."""
    from mlx_lm.models.activations import swiglu
    v = pool.views
    xe = mx.expand_dims(x, (-2, -3))
    common = dict(rhs_indices=idx, transpose=True,
                  group_size=pool.group_size, bits=pool.bits)
    x_up = mx.gather_qmm(xe, v["up_proj.weight"], scales=v["up_proj.scales"],
                         biases=v["up_proj.biases"], **common)
    x_gate = mx.gather_qmm(xe, v["gate_proj.weight"], scales=v["gate_proj.scales"],
                           biases=v["gate_proj.biases"], **common)
    h = swiglu(x_gate, x_up)
    y = mx.gather_qmm(h, v["down_proj.weight"], scales=v["down_proj.scales"],
                      biases=v["down_proj.biases"], **common)
    return (y.squeeze(-2) * scores[..., None]).sum(axis=-2)


def fp32_reference(pool, slot_list, x, scores):
    v, g, b = pool.views, pool.group_size, pool.bits
    xf = x.reshape(-1).astype(mx.float32)
    acc = mx.zeros((D_MODEL,), dtype=mx.float32)
    for i, s in enumerate(slot_list):
        def deq(name):
            return mx.dequantize(v[f"{name}.weight"][s], v[f"{name}.scales"][s],
                                 v[f"{name}.biases"][s], group_size=g,
                                 bits=b).astype(mx.float32)
        gate = deq("gate_proj") @ xf
        up = deq("up_proj") @ xf
        h = (gate * mx.sigmoid(gate)) * up
        acc = acc + float(scores[i]) * (deq("down_proj") @ h)
    return acc


def rel_err(a, b):
    a = np.array(a.astype(mx.float32), dtype=np.float64)
    b = np.array(b.astype(mx.float32), dtype=np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def main():
    nsg = int(os.environ.get("NSG", "32"))
    pool, idx = load_pool()
    slot_list = [int(s) for s in np.array(idx)]
    pool_u32 = pool.staging

    mx.random.seed(0)
    x = (mx.random.normal((D_MODEL,)) * 0.5).astype(mx.bfloat16)
    scores = mx.softmax(mx.random.normal((K_EXPERTS,)), axis=-1).astype(mx.float32)

    ref = fp32_reference(pool, slot_list, x, scores)
    mx.eval(ref)

    y_mlx = mlx_path(pool, idx.reshape(1, 1, K_EXPERTS),
                     x.reshape(1, 1, D_MODEL), scores.astype(mx.bfloat16))
    mx.eval(y_mlx)

    y_k = K.moe_expert_mlp(pool_u32, idx, x, scores, nsg=nsg,
                           verbose=os.environ.get("VERBOSE") == "1")
    mx.eval(y_k)

    e_mlx = rel_err(y_mlx.reshape(-1), ref)
    e_k = rel_err(y_k, ref)
    print(f"slots {slot_list}  nsg {nsg} ({nsg*32} threads/tg)")
    print(f"  max|.| reference   {float(mx.abs(ref).max()):.4f}")
    print(f"  MLX gather_qmm     rel err {e_mlx:.3e}")
    print(f"  fused metal kernel rel err {e_k:.3e}")
    ok = e_k <= max(e_mlx, 1e-6) * 1.5
    print(f"  -> kernel {'at least as accurate' if ok else 'WORSE'} as MLX path")

    # isolated timing: median over repeats, each fully evaluated
    def timeit(fn, n=60):
        for _ in range(5):
            mx.eval(fn())
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t)
        return sorted(ts)

    xb = x.reshape(1, 1, D_MODEL)
    ib = idx.reshape(1, 1, K_EXPERTS)
    sb = scores.astype(mx.bfloat16)
    tm = timeit(lambda: mlx_path(pool, ib, xb, sb))
    tk = timeit(lambda: K.moe_expert_mlp(pool_u32, idx, x, scores, nsg=nsg))
    n = len(tm)
    print(f"\n  MLX path   median {tm[n//2]*1e6:8.1f} us   "
          f"(p10 {tm[n//10]*1e6:.1f}, p90 {tm[9*n//10]*1e6:.1f})")
    print(f"  fused      median {tk[n//2]*1e6:8.1f} us   "
          f"(p10 {tk[n//10]*1e6:.1f}, p90 {tk[9*n//10]*1e6:.1f})")
    print(f"  speedup    {tm[n//2]/tk[n//2]:.2f}x")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
