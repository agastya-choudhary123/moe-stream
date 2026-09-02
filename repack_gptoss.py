#!/usr/bin/env python3
"""
Repack gpt-oss-120b-4bit into an expert-contiguous weight store, streaming.

The model is 65.8 GB and the store it produces is 64.6 GB. Holding both needs
130 GB against ~90 GB free, so this never does: it pulls ONE shard (<=5.4 GB),
writes every expert component that shard contains straight to its final offset,
deletes the shard, and moves on. Peak extra disk is the growing store plus one
shard.

Layout follows repack.py's idea with gpt-oss geometry -- MLX stores experts
stacked per layer, so one expert's components are spread across twelve tensors;
here one expert becomes one contiguous blob:

    offset(layer, expert) = (layer * 128 + expert) * BLOB

Two differences from the Qwen3 repack that both caused bugs first time round:

  Each projection carries a real `bias` vector on top of the quantization
  `biases`. Twelve components per expert, not nine.

  **10 of the 36 layers have their expert tensors split across two shards.**
  Assembling a whole blob in memory and writing it once therefore does not
  work -- half the components are in a shard that has already been deleted.
  Instead each component is written independently to its own offset inside the
  blob, and progress is tracked per (layer, proj, part). A blob is complete
  when its twelve components have all landed, in whatever order the shards
  happened to arrive.

Payload is not a multiple of the 16 KB page, so blobs are padded up so every
blob stays page-aligned and `os.preadv` lands straight in the MLX staging
buffer without straddling.

Resumable: rerun after an interruption and it skips components already written.
Non-expert tensors (attention, norms, embeddings, routers, lm_head) are saved to
a per-shard partial file as each shard is consumed, NOT accumulated in memory
until the end -- otherwise an interruption loses them all and the restart has to
re-download shards whose expert data is already on disk, purely to recover ~1 GB
of tensors. They are merged into nonexpert.safetensors once every shard is done.
"""

import json
import os
import struct
import sys
import time

import numpy as np

REPO = "mlx-community/gpt-oss-120b-4bit"
OUT = os.path.expanduser("~/Desktop/moe-stream/model-120b")
PARTDIR = os.path.join(OUT, "_nonexpert_parts")
PAGE = 16384
PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases", "bias")
DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "F16": 2, "BF16": 2, "U16": 2,
               "U8": 1, "I8": 1}
NP_OF = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
         "U32": np.uint32, "I32": np.int32, "U8": np.uint8, "I8": np.int8}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def free_gb():
    s = os.statvfs(OUT if os.path.exists(OUT) else "/")
    return s.f_bavail * s.f_frsize / 1e9


def main():
    from huggingface_hub import hf_hub_download

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PARTDIR, exist_ok=True)
    t_start = time.time()

    meta = {}
    for f in ("config.json", "model.safetensors.index.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "chat_template.jinja"):
        try:
            p = hf_hub_download(REPO, f)
            meta[f] = p
            if f != "model.safetensors.index.json":
                dst = os.path.join(OUT, f)
                if not os.path.exists(dst):
                    with open(p, "rb") as a, open(dst, "wb") as b:
                        b.write(a.read())
        except Exception as e:
            print(f"  (skip {f}: {str(e)[:60]})")

    cfg = json.load(open(meta["config.json"]))
    NL, NE = cfg["num_hidden_layers"], cfg["num_local_experts"]
    wmap = json.load(open(meta["model.safetensors.index.json"]))["weight_map"]
    shards = sorted(set(wmap.values()))
    print(f"{REPO}: {NL} layers x {NE} experts, {len(shards)} shards, "
          f"{free_gb():.0f} GB free")

    idx_path = os.path.join(OUT, "experts_index.json")
    bin_path = os.path.join(OUT, "experts.bin")
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else None
    written = set(tuple(x) for x in index.get("written", [])) if index else set()
    if written:
        print(f"  resuming: {len(written)}/{NL*len(PROJS)*len(PARTS)} "
              f"components already written")

    nonexpert = {}
    fd = None
    try:
        for si, shard in enumerate(shards):
            keys = [k for k, sh in wmap.items() if sh == shard]
            ekeys = [k for k in keys if ".mlp.experts." in k]
            nkeys = [k for k in keys if ".mlp.experts." not in k]
            todo = []
            for k in ekeys:
                l = int(k.split("model.layers.")[1].split(".")[0])
                proj, part = k.rsplit(".", 2)[-2:]
                if (l, proj, part) not in written:
                    todo.append((k, l, proj, part))
            part_path = os.path.join(PARTDIR, shard.replace(".safetensors", "")
                                     + ".npz")
            if not todo and (not nkeys or os.path.exists(part_path)):
                continue

            t0 = time.time()
            print(f"\n[{si+1}/{len(shards)}] {shard}  free {free_gb():.0f} GB",
                  flush=True)
            path = hf_hub_download(REPO, shard)
            sz = os.path.getsize(path)
            dl = max(time.time() - t0, 1e-6)
            print(f"    {sz/1e9:.2f} GB in {dl:.0f}s ({sz/dl/1e6:.0f} MB/s)",
                  flush=True)

            hdr, data_off = read_header(path)

            if index is None:                       # plan the blob once
                comps, off = [], 0
                for proj in PROJS:
                    for part in PARTS:
                        probe = [k for k in wmap
                                 if k.endswith(f".mlp.experts.{proj}.{part}")]
                        if not probe:
                            continue
                        # shapes are identical across layers; find one we can see
                        seen = next((k for k in probe if k in hdr), None)
                        shape = hdr[seen]["shape"] if seen else None
                        dt = hdr[seen]["dtype"] if seen else None
                        if shape is None:
                            raise SystemExit(
                                f"cannot plan blob: {proj}.{part} not in shard 1")
                        per = int(np.prod(shape[1:])) * DTYPE_BYTES[dt]
                        comps.append(dict(proj=proj, part=part, dtype=dt,
                                          shape=list(shape[1:]), offset=off,
                                          nbytes=per))
                        off += per
                blob = -(-off // PAGE) * PAGE
                index = dict(model=REPO, n_layers=NL, n_experts=NE,
                             blob_bytes=blob, payload_bytes=off, page_size=PAGE,
                             components=comps,
                             quantization={"group_size": 64, "bits": 4},
                             offset_formula="(layer*n_experts+expert)*blob_bytes",
                             written=[])
                print(f"    blob {blob:,} B = {blob//PAGE} pages "
                      f"({off:,} payload + {blob-off:,} pad), "
                      f"store {blob*NL*NE/1e9:.1f} GB")
            blob = index["blob_bytes"]
            coff = {(c["proj"], c["part"]): c for c in index["components"]}

            if fd is None:
                fd = os.open(bin_path, os.O_RDWR | os.O_CREAT)
                os.ftruncate(fd, blob * NL * NE)

            src = np.memmap(path, dtype=np.uint8, mode="r")

            if nkeys and not os.path.exists(part_path):
                blobs = {}
                for k in nkeys:
                    info = hdr[k]
                    s0, e0 = info["data_offsets"]
                    blobs[k] = np.array(src[data_off + s0:data_off + e0])
                    blobs[k + "|meta"] = np.frombuffer(
                        json.dumps([info["dtype"], info["shape"]]).encode(),
                        dtype=np.uint8)
                np.savez(part_path, **blobs)   # survives an interruption

            for k, l, proj, part in todo:
                c = coff.get((proj, part))
                if c is None:
                    continue
                base = hdr[k]["data_offsets"][0] + data_off
                n = c["nbytes"]
                for e in range(NE):
                    os.pwrite(fd, src[base + e * n: base + (e + 1) * n].tobytes(),
                              (l * NE + e) * blob + c["offset"])
                written.add((l, proj, part))
            index["written"] = sorted(written)
            json.dump(index, open(idx_path, "w"), indent=1)

            del src
            # hf_hub_download returns a SYMLINK into blobs/. Resolve it BEFORE
            # unlinking -- realpath() of a dead symlink returns the link path
            # itself, so removing the link first strands the 5 GB blob forever.
            # That leaked 20 GB over four shards the first time round.
            real = os.path.realpath(path)
            os.remove(path)
            if real != path and os.path.exists(real):
                os.remove(real)
            full = len({l for l in range(NL)
                        if all((l, c["proj"], c["part"]) in written
                               for c in index["components"])})
            print(f"    +{len(todo)} components, {full}/{NL} layers complete, "
                  f"{time.time()-t0:.0f}s", flush=True)
    finally:
        if fd is not None:
            os.close(fd)

    nonexpert = {}
    from glob import glob as _glob
    for f in sorted(_glob(os.path.join(PARTDIR, "*.npz"))):
        z = np.load(f)
        for k in z.files:
            if k.endswith("|meta"):
                continue
            dt, shape = json.loads(bytes(z[k + "|meta"]).decode())
            nonexpert[k] = (z[k], dt, shape)
    if nonexpert:
        import mlx.core as mx
        arrs = {}
        for k, (raw, dt, shape) in nonexpert.items():
            a = mx.array(raw.view(NP_OF[dt]))
            if dt == "BF16":
                a = mx.view(a, mx.bfloat16)
            arrs[k] = a.reshape(shape)
        out = os.path.join(OUT, "nonexpert.safetensors")
        mx.save_safetensors(out, arrs)
        print(f"\nnonexpert.safetensors: {os.path.getsize(out)/1e9:.2f} GB, "
              f"{len(arrs)} tensors")

    full = len({l for l in range(NL)
                if all((l, c["proj"], c["part"]) in written
                       for c in index["components"])})
    print(f"\ndone in {(time.time()-t_start)/60:.1f} min -> {OUT}")
    print(f"complete layers: {full}/{NL}, free {free_gb():.0f} GB")


if __name__ == "__main__":
    main()
