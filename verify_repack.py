#!/usr/bin/env python3
"""
Verify the repacked store is byte-identical to the original, then measure it.

A silent corruption here would surface much later as garbage tokens, so this
compares raw bytes for randomly chosen experts against the source shards
rather than trusting the repack loop.
"""

import json
import os
import random
import struct
import time
from glob import glob

import numpy as np

OUT = os.path.expanduser("~/Desktop/moe-stream/model")
SRC = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/snapshots")
F_NOCACHE, F_RDAHEAD = 48, 45
N_CHECK = 12


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main():
    idx = json.load(open(os.path.join(OUT, "experts_index.json")))
    blob_bytes = idx["blob_bytes"]
    n_exp = idx["n_experts"]
    layout = idx["components"]

    snap = sorted(glob(os.path.join(SRC, "*")))[0]
    tensors = {}
    for sp in sorted(glob(os.path.join(snap, "*.safetensors"))):
        hdr, ds = read_header(sp)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            tensors[name] = (sp, ds + s, ds + e)

    path = os.path.join(OUT, "experts.bin")
    size = os.path.getsize(path)
    expect = blob_bytes * idx["n_layers"] * n_exp
    print(f"experts.bin : {size/2**30:.2f} GB  "
          f"({'size ok' if size == expect else 'SIZE MISMATCH'})")

    fd = os.open(path, os.O_RDONLY)
    ok = True
    random.seed(0)
    picks = [(random.randrange(idx["n_layers"]), random.randrange(n_exp))
             for _ in range(N_CHECK)]
    picks += [(0, 0), (idx["n_layers"] - 1, n_exp - 1)]   # edges

    print(f"\nchecking {len(picks)} experts byte-for-byte ...")
    for li, ex in picks:
        off = (li * n_exp + ex) * blob_bytes
        blob = os.pread(fd, blob_bytes, off)
        assert len(blob) == blob_bytes
        for c in layout:
            got = blob[c["offset"]:c["offset"] + c["nbytes"]]
            nm = f"model.layers.{li}.mlp.switch_mlp.{c['proj']}.{c['part']}"
            sp, s, _ = tensors[nm]
            sfd = os.open(sp, os.O_RDONLY)
            try:
                want = os.pread(sfd, c["nbytes"], s + ex * c["nbytes"])
            finally:
                os.close(sfd)
            if got != want:
                print(f"  MISMATCH layer {li} expert {ex} "
                      f"{c['proj']}.{c['part']}")
                ok = False
    print(f"  {'ALL MATCH' if ok else 'CORRUPTION DETECTED'}")

    # alignment
    bad = [(l, e) for l in range(idx["n_layers"]) for e in range(n_exp)
           if ((l * n_exp + e) * blob_bytes) % idx["page_size"]]
    print(f"  page-aligned offsets: {len(bad) == 0}")

    # throughput on the real store
    os.close(fd)
    fd = os.open(path, os.O_RDONLY)
    os.fcntl = __import__("fcntl")
    os.fcntl.fcntl(fd, F_NOCACHE, 1)
    os.fcntl.fcntl(fd, F_RDAHEAD, 0)
    n = 200
    offs = [random.randrange(0, (size - blob_bytes) // blob_bytes) * blob_bytes
            for _ in range(n)]
    for o in offs[:8]:
        os.pread(fd, blob_bytes, o)
    t0 = time.perf_counter()
    for o in offs:
        os.pread(fd, blob_bytes, o)
    dt = time.perf_counter() - t0
    bw = n * blob_bytes / dt / 2**20
    print(f"\nrandom expert reads (qd1): {bw:,.0f} MB/s, "
          f"{dt/n*1e3:.2f} ms/expert")
    per_tok = idx["n_layers"] * 8 * blob_bytes / 2**20
    print(f"  {per_tok:,.0f} MB/token fully cold -> {bw/per_tok:.2f} tok/s floor")
    os.close(fd)

    import mlx.core as mx
    w = mx.load(os.path.join(OUT, "nonexpert.safetensors"))
    nb = sum(v.nbytes for v in w.values())
    print(f"\nnonexpert.safetensors: {len(w)} tensors, {nb/2**30:.2f} GB")
    print(f"  has routers: {sum(1 for k in w if k.endswith('mlp.gate.weight'))}"
          f" of {idx['n_layers']} layers")


if __name__ == "__main__":
    main()
