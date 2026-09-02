#!/usr/bin/env python3
"""
Structural verification of the gpt-oss-120b expert store.

verify_repack.py checks the Qwen3 store byte-for-byte against the source
shards. That is not available here: repack_gptoss.py deletes each shard the
moment it is consumed, because holding the 65.8 GB download and the 64.6 GB
store at once needs 130 GB and there are ~90. So this verifies what can still
be verified without the source, which is enough to catch the failure modes that
actually happen in a streaming repack:

  a component never written (shard deleted before its layer completed)
  a component written at the wrong offset (blob geometry off by one)
  a torn or partial write

All three show up as weights that are zero, constant, or statistically absurd.
Real 4-bit affine weights have a characteristic signature: nibbles roughly
uniform over 0..15, scales small and positive-ish, dequantized values roughly
zero-mean with std in the low hundredths. Checking every one of the 4608
experts is ~64 GB of reading, so this samples by default and takes --all for
the full sweep.
"""

import argparse
import ctypes
import fcntl
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ext", "build"))
import mlx.core as mx

MODEL_DIR = os.path.expanduser("~/Desktop/moe-stream/model-120b")


def bf16(raw):
    return mx.view(mx.array(raw.view(np.uint16)), mx.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every expert (~64 GB read)")
    ap.add_argument("--per-layer", type=int, default=4)
    ap.add_argument("--dir", default=MODEL_DIR)
    a = ap.parse_args()

    idx = json.load(open(os.path.join(a.dir, "experts_index.json")))
    blob, NL, NE = idx["blob_bytes"], idx["n_layers"], idx["n_experts"]
    comps = {(c["proj"], c["part"]): c for c in idx["components"]}
    gs, bits = idx["quantization"]["group_size"], idx["quantization"]["bits"]

    written = set(tuple(x) for x in idx.get("written", []))
    missing = [(l, p, q) for l in range(NL) for (p, q) in comps
               if (l, p, q) not in written]
    print(f"store: {NL} layers x {NE} experts, blob {blob:,} B "
          f"({blob//idx['page_size']} pages)")
    print(f"index: {len(written)}/{NL*len(comps)} components marked written")
    if missing:
        print(f"  !! {len(missing)} components NEVER WRITTEN, e.g. {missing[:4]}")

    path = os.path.join(a.dir, "experts.bin")
    want = blob * NL * NE
    have = os.path.getsize(path)
    print(f"file:  {have:,} B, expected {want:,} "
          f"{'OK' if have == want else '!! SIZE MISMATCH'}")
    du = os.statvfs(a.dir)
    print(f"       {os.popen(f'du -sh {path}').read().split()[0]} actually on disk")

    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)
    buf = (ctypes.c_char * blob)()

    layers = range(NL)
    bad, checked = [], 0
    print(f"\nchecking {'every expert' if a.all else f'{a.per_layer}/layer'}:")
    for l in layers:
        es = range(NE) if a.all else np.linspace(0, NE - 1, a.per_layer).astype(int)
        stats = []
        for e in es:
            os.preadv(fd, [buf], (l * NE + int(e)) * blob)
            raw = np.frombuffer(buf, dtype=np.uint8)
            checked += 1
            for (proj, part), c in comps.items():
                seg = raw[c["offset"]:c["offset"] + c["nbytes"]]
                if not seg.any():
                    bad.append((l, int(e), proj, part, "ALL ZERO"))
                    continue
                if part == "weight":
                    q = seg.view(np.uint32)
                    nib = np.concatenate([(q >> (4 * i)) & 0xF for i in range(8)])
                    # real 4-bit weights spread over the codebook; a torn write
                    # or wrong offset collapses that spread
                    if nib.max() == nib.min():
                        bad.append((l, int(e), proj, part, "CONSTANT NIBBLES"))
                    stats.append(nib.mean())
                elif part == "scales":
                    s = np.array(bf16(seg).astype(mx.float32)).reshape(c["shape"])
                    if not np.isfinite(s).all():
                        bad.append((l, int(e), proj, part, "NON-FINITE SCALES"))
                    # NO magnitude threshold here, deliberately. Two attempts
                    # at one produced only false positives: gpt-oss scales grow
                    # smoothly with depth (down_proj mean |scale| 0.008 at layer
                    # 0, 0.38 at 15, ~2.2 at 28-32, ~4.0 at 33-34), so any fixed
                    # cutoff lands mid-distribution somewhere and flags healthy
                    # experts -- 57 of them at >10, then layers 33/34 at ">5 for
                    # 10% of rows". Dequantized std tracks the same smooth curve
                    # (7.1, 7.8, 6.9, 6.4, 8.0, 10.6, 9.5 across 28-34), which is
                    # physiology, not corruption. Zero / constant / non-finite
                    # are the checks that actually catch a bad write.
        if l % 6 == 0 or l == NL - 1:
            m = np.mean(stats) if stats else float("nan")
            print(f"   layer {l:>2}: nibble mean {m:5.2f} (7.5 = uniform)  "
                  f"{'ok' if not bad else str(len(bad)) + ' issues so far'}",
                  flush=True)
    os.close(fd)

    print(f"\nchecked {checked} experts, {checked*blob/1e9:.1f} GB read")
    if bad:
        print(f"!! {len(bad)} problems, first 10:")
        for b in bad[:10]:
            print("   ", b)
        sys.exit(1)
    print("PASS: every sampled component non-zero, nibbles spread, scales finite")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
