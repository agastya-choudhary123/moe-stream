#!/usr/bin/env python3
"""
Byte-verify the slot pool, especially past the old int32 overflow point.

This is the test that did not exist, and its absence cost the project a
corrupted model in both engines: the flat-buffer pool silently returned the
wrong scales/biases for every slot beyond 2^31 / (blob/2) -- slot 307 on
gpt-oss-120b's 600-slot default, slot 1619 on Qwen3-30B's 3072-slot default.
The weights were right the whole time, so nothing crashed and the output stayed
plausible.

What is checked, for slots spread across the whole pool including the far end:

  1. every component of every filled slot matches a direct pread, byte for byte
  2. the views are LIVE -- refilling a slot with a different expert changes what
     the view returns (a copy would silently freeze)
  3. no component array has >= 2^31 elements, which is where MLX shapes wrap

  python3 verify_pool.py model-120b 700
"""

import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_v3 import ExpertPool


def part_bytes(fd, idx, blob, layer, expert, c):
    return os.pread(fd, c["nbytes"],
                    (layer * idx["n_experts"] + expert) * blob + c["offset"])


def as_np(a):
    return np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a)


def ref_np(raw, c):
    if c["dtype"] == "U32":
        return np.frombuffer(raw, dtype=np.uint32).reshape(c["shape"])
    v = mx.view(mx.array(np.frombuffer(raw, dtype=np.uint16).copy()), mx.bfloat16)
    return np.array(v.astype(mx.float32)).reshape(c["shape"])


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else "model-120b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 700
    idx = json.load(open(os.path.join(md, "experts_index.json")))
    blob = idx["blob_bytes"]
    old_limit = 2 ** 31 // (blob // 2)
    print(f"{md}: {n} slots x {blob/1e6:.2f} MB; the flat-buffer pool went wrong "
          f"past slot {old_limit}")

    p = ExpertPool(md, n_slots=n)
    print(f"  pool {p.pool_bytes/2**30:.2f} GiB in {len(p.parts)} arrays, "
          f"largest {max(a.size for _, a, _ in p.parts):,} elements")
    fd = os.open(os.path.join(md, "experts.bin"), os.O_RDONLY)
    comps = sorted(idx["components"], key=lambda c: c["offset"])

    test = sorted({0, 1, old_limit - 1, old_limit, old_limit + 1, n // 2,
                   n - 2, n - 1} & set(range(n)))
    bad = 0
    for s in test:
        L, e = 7, s % idx["n_experts"]
        p._read(s, L, e)
        for c in comps:
            got = as_np(p.views[f"{c['proj']}.{c['part']}"][s])
            ref = ref_np(part_bytes(fd, idx, blob, L, e, c), c)
            if not np.array_equal(got, ref):
                bad += 1
                print(f"  MISMATCH slot {s} {c['proj']}.{c['part']}")
        print(f"  slot {s:>5}: all {len(comps)} components match")

    # liveness: the same slot, a different expert
    s = n - 1
    first = as_np(p.views["gate_proj.scales"][s]).copy()
    p._read(s, 9, 11)
    ref = ref_np(part_bytes(fd, idx, blob, 9, 11,
                            [c for c in comps if c["part"] == "scales"][0]),
                 [c for c in comps if c["part"] == "scales"][0])
    now = as_np(p.views["gate_proj.scales"][s])
    live = np.array_equal(now, ref) and not np.array_equal(now, first)
    print(f"  views are live (track buffer writes): {live}")
    over = [nm for nm, a, _ in p.parts if a.size >= 2 ** 31]
    print(f"  arrays at risk of int32 wrap: {over or 'none'}")
    ok = bad == 0 and live and not over
    print("PASS" if ok else f"FAIL ({bad} mismatched components)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
