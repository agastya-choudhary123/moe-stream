#!/usr/bin/env python3
"""
Phase 3: repack Qwen3-30B-A3B into an expert-contiguous weight store.

On disk MLX stores experts stacked per layer: switch_mlp.gate_proj.weight has
shape [128, 768, 256]. Expert e is a contiguous slice of that tensor, but its
*full* set of nine components (gate/up/down x weight/scales/biases) is spread
across nine separate tensors. Fetching one expert natively means nine reads,
six of them 48 KB -- and Phase 0b measured small reads at a fraction of peak.

This rewrites the model so one expert is one contiguous 2.53 MB blob, which is
exactly 162 pages, so every blob starts page-aligned with no padding. One
pread per expert, straight into an MLX staging buffer.

Outputs:
  experts.bin           15.2 GB, blob (layer, expert) at (layer*128+expert)*2654208
  experts_index.json    component layout within a blob
  nonexpert.safetensors ~1 GB, attention/norms/embeddings/routers for MLX
"""

import json
import os
import struct
import sys
import time
from glob import glob

import numpy as np

SRC = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/snapshots")
OUT = os.path.expanduser("~/Desktop/moe-stream/model")
PAGE = 16384

PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
COMPONENTS = [(p, c) for p in PROJS for c in PARTS]

DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "F16": 2, "BF16": 2, "U16": 2,
               "U8": 1, "I8": 1}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def main():
    snap = sorted(glob(os.path.join(SRC, "*")))[0]
    shards = sorted(glob(os.path.join(snap, "*.safetensors")))
    if not shards:
        sys.exit(f"no shards under {snap}")
    cfg = json.load(open(os.path.join(snap, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    n_experts = cfg["num_experts"]

    os.makedirs(OUT, exist_ok=True)

    # index every tensor: name -> (shard path, absolute byte start, dtype, shape)
    tensors = {}
    for sp in shards:
        hdr, data_start = read_header(sp)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            tensors[name] = (sp, data_start + s, data_start + e,
                             meta["dtype"], meta["shape"])

    # ---- work out the blob layout from layer 0 -----------------------------
    layout, blob_bytes = [], 0
    for proj, part in COMPONENTS:
        nm = f"model.layers.0.mlp.switch_mlp.{proj}.{part}"
        _, s, e, dt, shape = tensors[nm]
        per = (e - s) // n_experts
        layout.append({
            "proj": proj, "part": part, "dtype": dt,
            "shape": shape[1:], "offset": blob_bytes, "nbytes": per,
        })
        blob_bytes += per

    print(f"experts      : {n_layers} layers x {n_experts}")
    print(f"blob size    : {blob_bytes:,} bytes ({blob_bytes/2**20:.2f} MB)")
    print(f"page aligned : {blob_bytes % PAGE == 0} "
          f"({blob_bytes/PAGE:.0f} pages)")
    if blob_bytes % PAGE:
        sys.exit("blob size is not a multiple of the page size; "
                 "padding would be required and the reader assumes none")

    total = blob_bytes * n_layers * n_experts
    free = os.statvfs(OUT).f_bavail * os.statvfs(OUT).f_frsize
    print(f"output       : {total/2**30:.2f} GB  (free {free/2**30:.0f} GB)")
    if free < total * 1.05:
        sys.exit("not enough free disk")

    # ---- write experts.bin, one layer at a time ----------------------------
    # Read the nine stacked tensors for a layer (~324 MB), then emit its 128
    # blobs sequentially. Keeps peak memory at one layer and both the reads and
    # the writes mostly sequential.
    out_path = os.path.join(OUT, "experts.bin")
    t0 = time.perf_counter()
    written = 0
    with open(out_path, "wb", buffering=1 << 24) as out:
        for li in range(n_layers):
            stacks = {}
            for proj, part in COMPONENTS:
                nm = f"model.layers.{li}.mlp.switch_mlp.{proj}.{part}"
                sp, s, e, _, _ = tensors[nm]
                fd = os.open(sp, os.O_RDONLY)
                try:
                    buf = os.pread(fd, e - s, s)
                finally:
                    os.close(fd)
                if len(buf) != e - s:
                    sys.exit(f"short read on {nm}")
                stacks[(proj, part)] = buf

            for ex in range(n_experts):
                for comp in layout:
                    per = comp["nbytes"]
                    blob = stacks[(comp["proj"], comp["part"])]
                    out.write(blob[ex * per:(ex + 1) * per])
                    written += per

            done = li + 1
            el = time.perf_counter() - t0
            print(f"  layer {done:>2}/{n_layers}  {written/2**30:6.2f} GB  "
                  f"{written/el/2**20:,.0f} MB/s", end="\r", flush=True)

    print(f"\nwrote {written/2**30:.2f} GB in {time.perf_counter()-t0:.0f}s")
    if written != total:
        sys.exit(f"size mismatch: wrote {written}, expected {total}")

    # ---- index -------------------------------------------------------------
    index = {
        "model": "mlx-community/Qwen3-30B-A3B-4bit",
        "n_layers": n_layers, "n_experts": n_experts,
        "blob_bytes": blob_bytes, "page_size": PAGE,
        "offset_formula": "(layer * n_experts + expert) * blob_bytes",
        "quantization": cfg["quantization"],
        "components": layout,
    }
    with open(os.path.join(OUT, "experts_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # ---- non-expert weights for MLX to load normally ------------------------
    import mlx.core as mx
    keep = {}
    for sp in shards:
        for k, v in mx.load(sp).items():
            if "switch_mlp" not in k:
                keep[k] = v
    nb = sum(v.nbytes for v in keep.values())
    mx.save_safetensors(os.path.join(OUT, "nonexpert.safetensors"), keep)
    print(f"non-expert   : {len(keep)} tensors, {nb/2**30:.2f} GB")

    for extra in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json", "merges.txt",
                  "vocab.json"):
        src = os.path.join(snap, extra)
        if os.path.exists(src):
            with open(src, "rb") as a, open(os.path.join(OUT, extra), "wb") as b:
                b.write(a.read())

    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
